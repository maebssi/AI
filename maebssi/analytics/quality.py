"""불량(설비 고장) 예측 + 원인 변수 분석 — AI4I 2020.

- 모델: HistGradientBoosting (물리 기반 파생변수 포함) vs 로지스틱 회귀 기준선
- 평가: 층화 5-fold CV AUC/PR-AUC + 20% 홀드아웃
- 원인 변수: 전역(순열 중요도) + 개별 건(가림 기법: 변수를 정상 중앙값으로 바꿨을 때 고장확률 감소량)
- 원인 적중률: 홀드아웃 고장 건의 1순위 원인 변수가 실제 고장 유형(TWF/HDF/PWF/OSF)을 규정하는 변수군에 속하는 비율

실행: python -m maebssi.analytics.quality
"""
import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from maebssi.analytics.db import connect
from maebssi.config import MODEL_DIR, REPORT_DIR

SEED = 42
RAW = ["air_temp_k", "process_temp_k", "rpm", "torque_nm", "tool_wear_min"]
LABELS = {
    "air_temp_k": "공기 온도", "process_temp_k": "공정 온도", "rpm": "회전 속도", "torque_nm": "토크",
    "tool_wear_min": "공구 마모 시간", "temp_diff_k": "공정-공기 온도차", "power_w": "기계 출력(토크×각속도)",
    "strain": "공구 마모×토크(부하 누적)", "type_code": "제품 등급",
}
# 고장 유형 → 그 유형을 규정하는 변수군 (AI4I 데이터 설명서의 고장 생성 규칙 기준)
FAILURE_DRIVERS = {
    "twf": {"tool_wear_min", "strain"},
    "hdf": {"temp_diff_k", "rpm", "process_temp_k", "air_temp_k"},
    "pwf": {"power_w", "torque_nm", "rpm"},
    "osf": {"strain", "tool_wear_min", "torque_nm", "type_code"},
}
FAILURE_NAMES = {"twf": "공구 마모 고장", "hdf": "방열 고장", "pwf": "전력 고장", "osf": "과부하 고장", "rnf": "무작위 고장"}
ACTIONS = {
    "tool_wear_min": "공구 교체 주기 점검(200분 이상 사용 공구 교체)",
    "strain": "공구 마모가 큰 상태에서 고토크 가공 회피 — 공구 교체 또는 가공 조건 완화",
    "temp_diff_k": "냉각 계통 점검(공정-공기 온도차 부족 → 방열 불량)",
    "rpm": "회전 속도 설정 확인(저속 운전 시 방열 저하, 과속·저속 모두 출력 이상)",
    "power_w": "출력이 정상 범위(약 3.5~9kW)를 벗어남 — 토크·속도 조합 조정",
    "torque_nm": "과도한 절삭 부하 — 이송량·절입량 조정",
    "process_temp_k": "공정 온도 관리(냉각수·환기)", "air_temp_k": "작업장 온도 관리",
    "type_code": "저등급(L) 제품은 과부하 한계가 낮음 — 가공 조건 보수적으로",
}


def load() -> pd.DataFrame:
    con = connect()
    df = pd.read_sql("SELECT * FROM machining", con)
    con.close()
    return add_features(df)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["temp_diff_k"] = df["process_temp_k"] - df["air_temp_k"]
    df["power_w"] = df["torque_nm"] * df["rpm"] * 2 * np.pi / 60
    df["strain"] = df["tool_wear_min"] * df["torque_nm"]
    df["type_code"] = df["product_type"].map({"L": 0, "M": 1, "H": 2}).astype(float)
    return df


FEATURES = RAW + ["temp_diff_k", "power_w", "strain", "type_code"]


def make_gbm():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=20, class_weight="balanced", random_state=SEED)


def make_baseline():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))


def local_causes(model, x: pd.DataFrame, reference: pd.Series, top: int = 3) -> list[dict]:
    """가림 기법: 변수 하나(파생변수 포함 원시 변수 묶음)를 정상 중앙값으로 되돌렸을 때 고장확률 감소량."""
    base = float(model.predict_proba(x[FEATURES])[0, 1])
    out = []
    for f in FEATURES:
        x2 = x.copy()
        x2[f] = reference[f]
        if f in RAW:  # 원시 변수를 바꾸면 파생변수도 함께 재계산
            x2 = add_features(x2.drop(columns=["temp_diff_k", "power_w", "strain", "type_code"]))
            x2[f] = reference[f]
        drop = base - float(model.predict_proba(x2[FEATURES])[0, 1])
        out.append({"feature": f, "label": LABELS[f], "value": float(x[f].iloc[0]),
                    "normal_median": float(reference[f]), "prob_drop": drop, "action": ACTIONS.get(f, "")})
    out.sort(key=lambda d: -d["prob_drop"])
    return [d for d in out if d["prob_drop"] > 0.01][:top]


def main():
    df = load()
    X, y = df[FEATURES], df["machine_failure"].to_numpy()
    report = {"n": len(df), "failure_rate": float(y.mean())}

    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for name, make in [("gbm", make_gbm), ("logistic_baseline", make_baseline), ("gbm_raw_only", make_gbm)]:
        cols = RAW + ["type_code"] if name == "gbm_raw_only" else FEATURES
        aucs, aps = [], []
        for tr, te in skf.split(X, y):
            m = make().fit(df.iloc[tr][cols], y[tr])
            p = m.predict_proba(df.iloc[te][cols])[:, 1]
            aucs.append(roc_auc_score(y[te], p))
            aps.append(average_precision_score(y[te], p))
        report[f"cv5_{name}"] = {"auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
                                 "pr_auc_mean": float(np.mean(aps))}
        print(name, report[f"cv5_{name}"])

    tr, te = train_test_split(df, test_size=0.2, stratify=y, random_state=SEED)
    model = make_gbm().fit(tr[FEATURES], tr["machine_failure"])
    p = model.predict_proba(te[FEATURES])[:, 1]
    yt = te["machine_failure"].to_numpy()
    thr = 0.5
    pred = p >= thr
    report["holdout"] = {"auc": float(roc_auc_score(yt, p)), "pr_auc": float(average_precision_score(yt, p)),
                         "recall@0.5": float(pred[yt == 1].mean()), "precision@0.5": float(yt[pred].mean()),
                         "n_test": int(len(te)), "n_fail_test": int(yt.sum())}
    print("holdout", report["holdout"])

    pi = permutation_importance(model, te[FEATURES], yt, scoring="roc_auc", n_repeats=10, random_state=SEED)
    report["global_importance"] = sorted(
        [{"feature": f, "label": LABELS[f], "auc_drop": float(v)} for f, v in zip(FEATURES, pi.importances_mean)],
        key=lambda d: -d["auc_drop"])

    reference = tr[tr.machine_failure == 0][FEATURES].median()
    hits, rows = [], []
    fails = te[(te.machine_failure == 1) & (te[list(FAILURE_DRIVERS)].sum(axis=1) >= 1)]
    for _, r in fails.iterrows():
        causes = local_causes(model, r.to_frame().T.astype({c: float for c in FEATURES}), reference)
        true_types = [t for t in FAILURE_DRIVERS if r[t] == 1]
        drivers = set().union(*(FAILURE_DRIVERS[t] for t in true_types))
        hit = bool(causes) and causes[0]["feature"] in drivers
        hits.append(hit)
        rows.append({"udi": int(r["udi"]), "types": true_types, "top_cause": causes[0]["feature"] if causes else None, "hit": hit})
    report["cause_top1_hit_rate"] = float(np.mean(hits))
    report["cause_eval_n"] = len(hits)
    report["cause_examples"] = rows[:15]
    print("cause top1 hit", report["cause_top1_hit_rate"], len(hits))

    joblib.dump({"model": model, "reference": reference}, MODEL_DIR / "quality_gbm.joblib")
    (REPORT_DIR / "quality_eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf8")


@lru_cache(maxsize=1)
def _bundle():
    return joblib.load(MODEL_DIR / "quality_gbm.joblib")


def predict(record: dict) -> dict:
    """단일 가공 조건(원시 5변수 + product_type) → 고장확률·원인 변수·권장 조치."""
    b = _bundle()
    x = add_features(pd.DataFrame([{**record, "product_type": record.get("product_type", "L")}]))
    prob = float(b["model"].predict_proba(x[FEATURES])[0, 1])
    return {"failure_probability": prob, "risk": "높음" if prob >= 0.5 else "주의" if prob >= 0.2 else "낮음",
            "causes": local_causes(b["model"], x, b["reference"]) if prob >= 0.2 else [],
            "derived": {k: float(x[k].iloc[0]) for k in ("temp_diff_k", "power_w", "strain")}}


if __name__ == "__main__":
    main()
