"""에이전트별 정량 평가 + 통합 시나리오 평가.

  python -m scripts.evaluate            # 전체
  python -m scripts.evaluate rag qa     # 일부

결과: artifacts/reports/evaluation.json
LLM 사용 여부와 무관하게 재현되도록 기본적으로 MAEBSSI_LLM=off 로 실행한다(--llm 으로 켬).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if "--llm" not in sys.argv:
    os.environ["MAEBSSI_LLM"] = "off"

from maebssi.config import REPORT_DIR  # noqa: E402


def load_jsonl(name):
    return [json.loads(l) for l in (ROOT / "eval" / name).read_text("utf8").splitlines() if l.strip()]


def eval_rag(name: str = "rag_eval.jsonl") -> dict:
    from maebssi.knowledge.rag import get_agent
    ka = get_agent()
    rows = load_jsonl(name)
    rec, mrr, refuse_ok, cite_prec, details = [], [], [], [], []
    for r in rows:
        ans = ka.answer(r["q"])
        docs = [h["chunk_id"].split("#")[0] for h in ans["hits"]]
        answerable = bool(r["gold"])
        refuse_ok.append(ans["refused"] != answerable)
        if answerable:
            hit = [d in r["gold"] for d in docs[:5]]
            rec.append(any(hit))
            mrr.append(1 / (hit.index(True) + 1) if any(hit) else 0)
            if not ans["refused"] and ans["citations"]:
                cited_docs = {h["chunk_id"].split("#")[0] for h in ans["hits"] if h["cite"] in ans["citations"]}
                cite_prec.append(len(cited_docs & set(r["gold"])) / len(cited_docs))
        details.append({"q": r["q"], "gold": r["gold"], "top_docs": docs[:3], "refused": ans["refused"]})
    ans_rows = [r for r in rows if r["gold"]]
    return {
        "n_answerable": len(ans_rows), "n_unanswerable": len(rows) - len(ans_rows),
        "recall@5": float(np.mean(rec)), "mrr@5": float(np.mean(mrr)),
        "refusal_accuracy": float(np.mean(refuse_ok)),
        "refusal_accuracy_unanswerable": float(np.mean([ok for ok, r in zip(refuse_ok, rows) if not r["gold"]])),
        "false_refusal_rate": float(np.mean([not ok for ok, r in zip(refuse_ok, rows) if r["gold"]])),
        "citation_precision": float(np.mean(cite_prec)) if cite_prec else None,
        "dense_retrieval": ka.emb is not None,
        "details": details,
    }


def eval_rag_holdout() -> dict:
    """임계값 보정에 쓰지 않은 질문으로 측정."""
    return eval_rag("rag_holdout.jsonl")


def eval_nl2sql() -> dict:
    """자연어→SQL 실행 정확도 (결과 집합 비교). 규칙 파서 기준(LLM off)."""
    from maebssi.analytics.db import connect
    from maebssi.analytics.nl2sql import rule_based
    con = connect()

    def rs(sql):
        return sorted(tuple(round(v, 6) if isinstance(v, float) else v for v in r) for r in con.execute(sql).fetchall())

    out = {}
    for name in ("nl2sql_eval.jsonl", "nl2sql_holdout.jsonl"):
        rows, errors = load_jsonl(name), []
        for r in rows:
            pred = rule_based(r["q"])
            try:
                ok = rs(pred) == rs(r["sql"])
            except Exception as e:
                ok, pred = False, f"{pred} ({e})"
            if not ok:
                errors.append({"q": r["q"], "pred": pred, "gold": r["sql"]})
        key = "tuned_set" if "eval" in name else "holdout_set"
        out[key] = {"n": len(rows), "execution_accuracy": 1 - len(errors) / len(rows), "errors": errors}
    return out


def eval_quality() -> dict:
    from maebssi.analytics import quality
    quality.main()
    r = json.loads((REPORT_DIR / "quality_eval.json").read_text("utf8"))
    return {k: r[k] for k in ("n", "failure_rate", "cv5_gbm", "cv5_logistic_baseline", "cv5_gbm_raw_only", "holdout",
                              "cause_top1_hit_rate", "cause_eval_n", "global_importance")}


def eval_graph() -> dict:
    from maebssi.knowledge.failure_graph import evaluate
    return evaluate()


def eval_qa_routing() -> dict:
    from maebssi.orchestrator.qa import route
    rows = load_jsonl("qa_routing.jsonl")
    ok = [route(r["q"]) == r["intent"] for r in rows]
    return {"n": len(rows), "accuracy": float(np.mean(ok)),
            "errors": [r for r, o in zip(rows, ok) if not o]}


def eval_detection() -> dict:
    """Validation 세션을 5프레임(초) 간격으로 스트리밍하며 '경고 이상' 최초 감지 시점을 라벨과 비교."""
    from maebssi.equipment.agent import first_frame_reaching, get_agent, list_replay_sessions, replay_window
    agent = get_agent()
    out = []
    for s in list_replay_sessions().itertuples():
        true_first = first_frame_reaching(s.session_id, 2)
        detected = None
        for upto in range(10, s.frames + 1, 5):
            f, t, _ = replay_window(s.session_id, upto, window=90)
            if agent.assess(f, t).state >= 2:
                detected = upto
                break
        out.append({"session_id": s.session_id, "true_first_warning": true_first, "detected_at": detected})
    pos = [o for o in out if o["true_first_warning"]]
    neg = [o for o in out if not o["true_first_warning"]]
    lags = [o["detected_at"] - o["true_first_warning"] for o in pos if o["detected_at"]]
    return {
        "n_sessions": len(out), "n_with_warning": len(pos),
        "session_detection_rate": float(np.mean([o["detected_at"] is not None for o in pos])) if pos else None,
        "false_alarm_sessions": int(sum(o["detected_at"] is not None for o in neg)),
        "median_lag_sec": float(np.median(lags)) if lags else None,
        "p90_abs_lag_sec": float(np.percentile(np.abs(lags), 90)) if lags else None,
        "early_detections": int(sum(l < 0 for l in lags)),
        "sessions": out,
    }


def eval_scenarios() -> dict:
    """대표 시나리오 '이상 발생 대응' 단위 성공률.
    성공 = (정답 위험도에 맞는 분기) ∧ (경고/위험이면 올바른 우선순위의 초안 + 인용 근거 + 승인 전 미발행 + 승인 후 발행)."""
    from maebssi.equipment.agent import first_frame_reaching, list_replay_sessions
    from maebssi.orchestrator.graph import resume_incident, start_incident
    from maebssi.workorder import tools as wo
    from maebssi.workorder.db import reset
    reset()
    results = []
    sessions = list_replay_sessions()
    for s in sessions[sessions.max_state >= 3].itertuples():
        for target in (0, 1, 2, 3):
            upto = first_frame_reaching(s.session_id, target)
            nxt = first_frame_reaching(s.session_id, target + 1) if target < 3 else s.frames + 1
            upto = min(upto + 20, nxt - 1)  # 해당 단계 진입 20초 후(다음 단계 전)
            r = start_incident(s.session_id, upto)
            pred = r["assessment"]["state"]
            checks = {"state_correct": pred == target}
            if target >= 2:
                checks["drafted"] = bool(r.get("wo_id"))
                if r.get("wo_id"):
                    d = wo.get_work_order(r["wo_id"])
                    checks["priority_correct"] = d["priority"] == ("P1" if target == 3 else "P2")
                    checks["has_citations"] = "[MNT-" in d["body"]
                    checks["not_issued_before_approval"] = d["status"] == "draft" and r["pending_approval"] is not None
                    approver = "생산관리자,정비파트장" if d["priority"] == "P1" else "정비파트장"
                    r2 = resume_incident(r["thread_id"], True, approver, "평가 승인")
                    checks["issued_after_approval"] = wo.get_work_order(r["wo_id"])["status"] == "issued"
            else:
                checks["no_work_order"] = not r.get("wo_id")
            results.append({"session_id": s.session_id, "target_state": target, "pred_state": pred,
                            "success": all(checks.values()), **checks})
    reset()  # 평가로 생긴 작업지시 정리
    by_state = {t: float(np.mean([r["success"] for r in results if r["target_state"] == t])) for t in range(4)}
    return {"n": len(results), "success_rate": float(np.mean([r["success"] for r in results])),
            "success_rate_by_state": by_state, "results": results}


TASKS = {"rag": eval_rag, "rag_holdout": eval_rag_holdout, "nl2sql": eval_nl2sql, "quality": eval_quality,
         "graph": eval_graph, "qa": eval_qa_routing, "detection": eval_detection, "scenario": eval_scenarios}


def main():
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(TASKS)
    path = REPORT_DIR / "evaluation.json"
    report = json.loads(path.read_text("utf8")) if path.exists() else {}
    eq = REPORT_DIR / "equipment_eval.json"
    if eq.exists():
        e = json.loads(eq.read_text("utf8"))
        report["equipment_classifier"] = {k: e[k] for k in ("gbm", "fusion", "ensemble", "ensemble_by_kind")}
    for n in names:
        t0 = time.time()
        report[n] = TASKS[n]()
        summary = {k: v for k, v in report[n].items() if not isinstance(v, (list, dict))}
        print(f"[{n}] {time.time() - t0:.0f}s", json.dumps(summary, ensure_ascii=False))
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), "utf8")
    print("saved", path)


if __name__ == "__main__":
    main()
