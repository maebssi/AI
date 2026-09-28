"""설비 상태 에이전트 (추론).

세션(연속 프레임) 데이터를 받아 최신 시점의 위험도 4단계, 클래스 확률, 이상점수,
정상 대비 편차가 큰 신호(판정 근거)를 반환한다.
"""
import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import torch

from maebssi.config import MODEL_DIR, PROCESSED_DIR, STATE_CODES, STATE_NAMES
from maebssi.equipment.features import add_features
from maebssi.equipment.nets import FusionNet

SIGNAL_LABELS = {
    "NTC": "제어함 내부온도(NTC)", "CT1": "전류 CT1", "CT2": "주행모터 전류 CT2", "CT3": "전류 CT3",
    "CT4": "전류 CT4", "PM1.0": "분진 PM1.0", "PM2.5": "분진 PM2.5", "PM10": "분진 PM10",
    "ir_max": "열화상 최고온도", "ir_mean": "열화상 평균온도",
}
SMOOTH_FRAMES = 5  # 최근 N 프레임 확률 평균으로 단일 프레임 노이즈 완화


@dataclass
class Assessment:
    device_id: str
    kind: str
    frame_id: str
    timestamp: str
    state: int
    state_name: str
    state_code: str
    probabilities: dict
    anomaly_score: float
    anomaly_flag: bool
    top_signals: list = field(default_factory=list)
    symptom_tags: list = field(default_factory=list)
    latest_values: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    def summary(self) -> str:
        sig = ", ".join(f"{s['label']} {s['value']:.1f}(정상 중앙값 {s['normal_median']:.1f})"
                        for s in self.top_signals[:3])
        return (f"{self.device_id} 위험도 '{self.state_name}'(확률 {self.probabilities[self.state_name]:.0%}), "
                f"이상점수 {self.anomaly_score:.2f}. 주요 편차: {sig}")


class EquipmentAgent:
    def __init__(self, device: str | None = None):
        meta = json.loads((MODEL_DIR / "equipment_meta.json").read_text("utf8"))
        self.fcols = meta["feature_columns"]
        self.acols = meta["anomaly_columns"]
        self.athr = meta["anomaly_threshold"]
        self.ref = meta["normal_reference"]
        norm = meta["fusion_norm"]
        self.tab_mu, self.tab_sd = np.array(norm["tab_mu"]), np.array(norm["tab_sd"])
        self.img_mu, self.img_sd = norm["img_mu"], norm["img_sd"]
        self.gbm = joblib.load(MODEL_DIR / "equipment_gbm.joblib")
        self.iso = joblib.load(MODEL_DIR / "equipment_iforest.joblib")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = FusionNet(len(self.fcols)).to(self.device)
        self.net.load_state_dict(torch.load(MODEL_DIR / "equipment_fusion.pt", map_location=self.device))
        self.net.eval()

    @torch.no_grad()
    def predict_proba(self, feat: pd.DataFrame, thumbs: np.ndarray) -> np.ndarray:
        X = feat[self.fcols].to_numpy(np.float32)
        p_gbm = self.gbm.predict_proba(X)
        t = torch.tensor((X - self.tab_mu) / self.tab_sd, dtype=torch.float32, device=self.device)
        im = torch.tensor((thumbs.astype(np.float32) - self.img_mu) / self.img_sd,
                          device=self.device).unsqueeze(1)
        p_net = torch.softmax(self.net(t, im), 1).cpu().numpy()
        return (p_gbm + p_net) / 2

    def assess(self, frames: pd.DataFrame, thumbs: np.ndarray) -> Assessment:
        """frames: 한 설비 세션의 원시 프레임(시간순), thumbs: 같은 순서의 열화상 썸네일."""
        frames = frames.reset_index(drop=True).copy()
        frames["_row"] = np.arange(len(frames))
        feat = add_features(frames)
        order = feat["_row"].to_numpy()
        tail = feat.iloc[-SMOOTH_FRAMES:]
        proba = self.predict_proba(tail, thumbs[order[-SMOOTH_FRAMES:]]).mean(0)
        state = int(proba.argmax())
        last = feat.iloc[-1]
        anomaly = float(-self.iso.score_samples(last[self.acols].to_numpy(np.float32)[None])[0])

        ref = self.ref[last["kind"]]
        devs = []
        for sig, r in ref.items():
            z = (last[sig] - r["median"]) / r["iqr"]
            devs.append({"signal": sig, "label": SIGNAL_LABELS[sig], "value": float(last[sig]),
                         "normal_median": r["median"], "normal_p95": r["p95"], "robust_z": float(z)})
        devs.sort(key=lambda d: -abs(d["robust_z"]))
        # 정상 IQR 대비 3배 이상 벗어나고 정상 95백분위도 넘는 신호만 근거로 제시
        top = [d for d in devs if d["robust_z"] >= 3 and d["value"] > d["normal_p95"]][:5]

        return Assessment(
            device_id=str(last["device_id"]), kind=str(last["kind"]), frame_id=str(last["frame_id"]),
            timestamp=str(last["ts"]), state=state, state_name=STATE_NAMES[state],
            state_code=STATE_CODES[state],
            probabilities={STATE_NAMES[i]: float(p) for i, p in enumerate(proba)},
            anomaly_score=anomaly, anomaly_flag=anomaly > self.athr,
            top_signals=top, symptom_tags=symptom_tags(top),
            latest_values={k: float(last[k]) for k in SIGNAL_LABELS} | {"ex_temperature": float(last["ex_temperature"])},
        )


def symptom_tags(top_signals: list) -> list[str]:
    """편차 신호 → 증상 태그(지식 검색 질의에 사용)."""
    tags = []
    sigs = {s["signal"] for s in top_signals if s["robust_z"] > 0}
    if sigs & {"ir_max", "ir_mean", "NTC"}:
        tags.append("과열")
    if sigs & {"CT1", "CT2", "CT3", "CT4"}:
        tags.append("과전류")
    if sigs & {"PM1.0", "PM2.5", "PM10"}:
        tags.append("분진")
    return tags


@lru_cache(maxsize=1)
def get_agent() -> EquipmentAgent:
    return EquipmentAgent()


# ── 데모/시연용: 전처리된 Validation 세션을 실시간 스트림처럼 재생 ──────────────────────
@lru_cache(maxsize=1)
def _replay_store():
    df = pd.read_parquet(PROCESSED_DIR / "frames.parquet")
    df = df[df.split == "valid"].sort_values(["session_id", "ts"]).reset_index(drop=True)
    thumbs = np.load(PROCESSED_DIR / "thermal_valid.npy")
    return df, thumbs


def list_replay_sessions() -> pd.DataFrame:
    df, _ = _replay_store()
    return (df.groupby("session_id")
              .agg(device_id=("device_id", "first"), kind=("kind", "first"), frames=("frame_id", "size"),
                   max_state=("state", "max"), start=("ts", "min"))
              .reset_index())


def replay_window(session_id: str, upto: int | None = None, window: int = 120):
    """세션의 앞에서부터 upto 번째 프레임까지 중 최근 window 프레임을 반환 (upto=None → 끝까지)."""
    df, thumbs = _replay_store()
    s = df[df.session_id == session_id]
    if s.empty:
        raise KeyError(f"unknown session {session_id}")
    end = len(s) if upto is None else max(1, min(upto, len(s)))
    s = s.iloc[max(0, end - window):end]
    return s.drop(columns=["state"]), thumbs[s.thumb_idx.to_numpy()], s["state"].to_numpy()


def first_frame_reaching(session_id: str, state: int) -> int | None:
    """세션에서 정답 라벨이 state 이상이 되는 첫 프레임 번호(1-based) — 시나리오 구성용."""
    df, _ = _replay_store()
    s = df[df.session_id == session_id].reset_index(drop=True)
    idx = np.flatnonzero(s.state.to_numpy() >= state)
    return int(idx[0]) + 1 if len(idx) else None
