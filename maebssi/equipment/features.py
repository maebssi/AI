"""설비 상태 판정용 특징 생성.

- 프레임 원시값: 센서 8종, 외부환경 3종, 열화상 통계
- 시계열(인과적) 특징: 같은 세션 내 과거 W 프레임에 대한 이동평균·표준편차·기울기
  (미래 프레임이나 세션 경과시간은 쓰지 않는다 → 실시간 스트리밍에서도 동일하게 계산 가능)
"""
import numpy as np
import pandas as pd

from maebssi.config import EXTERNAL, SENSORS

IR_COLS = ["ir_mean", "ir_std", "ir_max", "ir_p50", "ir_p90", "ir_p99",
           "ir_hot_frac", "ir_center_mean", "ir_grad"]
TREND_COLS = SENSORS + ["ir_max", "ir_mean", "ir_p99"]
WINDOWS = (10, 30, 60)
BASE_COLS = SENSORS + EXTERNAL + IR_COLS + ["cumulative_operating_day", "equipment_history"]


def _slope(x: np.ndarray) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    t = np.arange(n) - (n - 1) / 2
    return float((t * (x - x.mean())).sum() / (t ** 2).sum())


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """session_id, ts 순으로 정렬된 df 에 특징 열을 추가해 반환."""
    df = df.sort_values(["session_id", "ts"]).reset_index(drop=True)
    out = {c: df[c].astype(float) for c in BASE_COLS}
    out["is_oht"] = (df["kind"] == "oht").astype(float)
    out["ir_minus_ambient"] = df["ir_max"] - df["ex_temperature"]
    out["ntc_minus_ambient"] = df["NTC"] - df["ex_temperature"]
    out["ct_total"] = df[["CT1", "CT2", "CT3", "CT4"]].sum(axis=1)
    out["ct_imbalance"] = df[["CT1", "CT2", "CT3", "CT4"]].std(axis=1)

    g = df.groupby("session_id", sort=False)
    for c in TREND_COLS:
        col = g[c]
        for w in WINDOWS:
            out[f"{c}_mean{w}"] = col.transform(lambda s: s.rolling(w, min_periods=1).mean())
            out[f"{c}_std{w}"] = col.transform(lambda s: s.rolling(w, min_periods=2).std()).fillna(0)
        out[f"{c}_slope30"] = col.transform(
            lambda s: s.rolling(30, min_periods=2).apply(_slope, raw=True)).fillna(0)
        out[f"{c}_dev60"] = df[c] - out[f"{c}_mean60"]
    feat = pd.DataFrame(out)
    return pd.concat([df, feat.add_prefix("f_")], axis=1)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("f_")]
