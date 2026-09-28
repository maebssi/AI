"""설비 상태 에이전트 모델 학습·평가.

모델 3종을 학습하고 Validation(학습에 없는 설비 agv17·18, oht17·18)에서 평가한다.
  1) GBM   : 센서·열화상 통계·시계열 특징 → HistGradientBoosting (클래스 가중)
  2) Fusion: 열화상 썸네일 CNN + 특징 MLP 멀티모달 신경망 (GPU)
  3) Ensemble: 1)+2) 확률 평균  ← 서비스 기본 모델
추가로 정상 구간만으로 학습한 IsolationForest 이상점수를 만든다(비지도 시계열 이상 탐지).

실행: python -m maebssi.equipment.train [--epochs 15]
"""
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             recall_score)
from sklearn.utils.class_weight import compute_sample_weight

from maebssi.config import MODEL_DIR, PROCESSED_DIR, REPORT_DIR, STATE_NAMES
from maebssi.equipment.features import add_features, feature_columns
from maebssi.equipment.nets import FusionNet

SEED = 42
ANOMALY_COLS = ["f_NTC_dev60", "f_CT1_dev60", "f_CT2_dev60", "f_CT3_dev60", "f_CT4_dev60",
                "f_ir_max_dev60", "f_ir_minus_ambient", "f_ntc_minus_ambient", "f_ct_imbalance",
                "f_NTC_slope30", "f_ir_max_slope30", "f_PM10_mean30", "f_PM2.5_mean30"]


def load():
    df = pd.read_parquet(PROCESSED_DIR / "frames.parquet")
    df = add_features(df)
    thumbs = {s: np.load(PROCESSED_DIR / f"thermal_{s}.npy") for s in ("train", "valid")}
    return df, thumbs


def metrics(y, pred) -> dict:
    return {
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "accuracy": float((y == pred).mean()),
        "recall_per_class": {STATE_NAMES[i]: float(r) for i, r in
                             enumerate(recall_score(y, pred, average=None, labels=[0, 1, 2, 3]))},
        # 경고·위험을 '경고 이상'으로 잡았는지 (정비 대응 관점의 핵심 지표)
        "recall_warning_or_above": float(((pred >= 2) & (y >= 2)).sum() / max((y >= 2).sum(), 1)),
        "false_alarm_rate_on_normal": float(((pred >= 2) & (y == 0)).sum() / max((y == 0).sum(), 1)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2, 3]).tolist(),
    }


def train_gbm(Xtr, ytr):
    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=48, min_samples_leaf=80,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=30, random_state=SEED)
    clf.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
    return clf


def train_fusion(tab_tr, img_tr, ytr, tab_va, img_va, epochs, device):
    torch.manual_seed(SEED)
    mu, sd = tab_tr.mean(0), tab_tr.std(0) + 1e-6
    img_mu, img_sd = float(img_tr.mean()), float(img_tr.std())
    norm_tab = lambda a: torch.tensor((a - mu) / sd, dtype=torch.float32)
    norm_img = lambda a: torch.tensor((a.astype(np.float32) - img_mu) / img_sd).unsqueeze(1)

    Ttr, Itr, Ytr = norm_tab(tab_tr), norm_img(img_tr), torch.tensor(ytr)
    Tva, Iva = norm_tab(tab_va), norm_img(img_va)
    model = FusionNet(tab_tr.shape[1]).to(device)
    w = torch.tensor(len(ytr) / (4 * np.bincount(ytr, minlength=4)), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    steps = epochs * int(np.ceil(len(Ytr) / 512))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Ytr))
        tot = 0.0
        for i in range(0, len(perm), 512):
            idx = perm[i:i + 512]
            t, im, y = Ttr[idx].to(device), Itr[idx].to(device), Ytr[idx].to(device)
            t = t + 0.05 * torch.randn_like(t)
            if torch.rand(1).item() < 0.5:
                im = im.flip(-1)
            loss = loss_fn(model(t, im), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        print(f"  fusion epoch {ep + 1}/{epochs} loss={tot / len(Ytr):.4f}", flush=True)
    norm = {"tab_mu": mu.tolist(), "tab_sd": sd.tolist(), "img_mu": img_mu, "img_sd": img_sd}
    return model, norm, predict_fusion(model, Tva, Iva, device)


@torch.no_grad()
def predict_fusion(model, T, I, device, bs=4096):
    model.eval()
    out = []
    for i in range(0, len(T), bs):
        out.append(torch.softmax(model(T[i:i + bs].to(device), I[i:i + bs].to(device)), 1).cpu().numpy())
    return np.concatenate(out)


def normal_reference(df_train: pd.DataFrame) -> dict:
    """정상(state 0) 구간의 설비종류별 주요 신호 분포 — 판정 근거 설명에 사용."""
    cols = ["NTC", "CT1", "CT2", "CT3", "CT4", "PM1.0", "PM2.5", "PM10", "ir_max", "ir_mean"]
    ref = {}
    for kind, g in df_train[df_train.state == 0].groupby("kind"):
        ref[kind] = {c: {"median": float(g[c].median()),
                         "iqr": float(g[c].quantile(.75) - g[c].quantile(.25)) or 1.0,
                         "p95": float(g[c].quantile(.95))} for c in cols}
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()
    t0 = time.time()
    df, thumbs = load()
    fcols = feature_columns(df)
    tr, va = df[df.split == "train"], df[df.split == "valid"]
    Xtr, ytr = tr[fcols].to_numpy(np.float32), tr.state.to_numpy()
    Xva, yva = va[fcols].to_numpy(np.float32), va.state.to_numpy()
    print(f"train={len(tr)} valid={len(va)} features={len(fcols)} ({time.time() - t0:.0f}s)")

    report = {"n_train": int(len(tr)), "n_valid": int(len(va)), "n_features": len(fcols),
              "train_devices": sorted(tr.device_id.unique().tolist()),
              "valid_devices": sorted(va.device_id.unique().tolist())}

    print("[1] GBM")
    gbm = train_gbm(Xtr, ytr)
    p_gbm = gbm.predict_proba(Xva)
    report["gbm"] = metrics(yva, p_gbm.argmax(1))
    print(json.dumps(report["gbm"], ensure_ascii=False, indent=1))

    print("[2] Fusion (thermal CNN + feature MLP)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, norm, p_fus = train_fusion(Xtr, thumbs["train"][tr.thumb_idx.to_numpy()], ytr,
                                    Xva, thumbs["valid"][va.thumb_idx.to_numpy()], args.epochs, device)
    report["fusion"] = metrics(yva, p_fus.argmax(1))
    print(json.dumps(report["fusion"], ensure_ascii=False, indent=1))

    print("[3] Ensemble")
    p_ens = (p_gbm + p_fus) / 2
    report["ensemble"] = metrics(yva, p_ens.argmax(1))
    print(json.dumps(report["ensemble"], ensure_ascii=False, indent=1))
    report["ensemble_classification_report"] = classification_report(
        yva, p_ens.argmax(1), target_names=list(STATE_NAMES.values()), output_dict=True)

    # 설비 종류별 성능
    report["ensemble_by_kind"] = {k: metrics(yva[m], p_ens.argmax(1)[m])
                                  for k in ("agv", "oht") for m in [(va.kind == k).to_numpy()]}

    print("[4] IsolationForest (normal-only)")
    normal = tr[tr.state == 0][ANOMALY_COLS].to_numpy(np.float32)
    iso = IsolationForest(n_estimators=300, max_samples=4096, random_state=SEED).fit(normal)
    score = -iso.score_samples(va[ANOMALY_COLS].to_numpy(np.float32))
    report["anomaly_mean_score_by_state"] = {STATE_NAMES[s]: float(score[yva == s].mean()) for s in range(4)}
    thr = float(np.quantile(-iso.score_samples(normal), 0.99))
    report["anomaly_threshold_p99_normal"] = thr

    joblib.dump(gbm, MODEL_DIR / "equipment_gbm.joblib")
    joblib.dump(iso, MODEL_DIR / "equipment_iforest.joblib")
    torch.save(net.state_dict(), MODEL_DIR / "equipment_fusion.pt")
    meta = {"feature_columns": fcols, "anomaly_columns": ANOMALY_COLS, "anomaly_threshold": thr,
            "fusion_norm": norm, "normal_reference": normal_reference(tr)}
    (MODEL_DIR / "equipment_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), "utf8")
    (REPORT_DIR / "equipment_eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf8")
    pd.DataFrame({"frame_id": va.frame_id, "device_id": va.device_id, "session_id": va.session_id,
                  "state": yva, "pred": p_ens.argmax(1), **{f"p{i}": p_ens[:, i] for i in range(4)},
                  "anomaly": score}).to_parquet(REPORT_DIR / "equipment_valid_predictions.parquet")
    print(f"done in {time.time() - t0:.0f}s → {MODEL_DIR}")


if __name__ == "__main__":
    main()
