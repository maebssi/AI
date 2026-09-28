"""이상 발생 대응 시나리오 오케스트레이터 (LangGraph).

  monitor ──(정상)──────────────────────────────▶ END
     │ (관심)                                         ▲
     ├──────▶ watch ─────────────────────────────────┤
     │ (경고·위험)                                     │
     └──▶ diagnose ─▶ plan ─▶ draft ─▶ approval(⏸ 관리자) ─▶ END

- monitor : 설비 상태 에이전트 (센서+열화상 → 위험도 4단계)
- diagnose: 설비 지식 에이전트 (증상 태그로 원인·조치·안전 절차 검색, 인용 포함)
- plan    : 작업지시 도구 (정비 이력, 부품 재고, 생산 영향, 기술자 배정)
- draft   : 작업지시 초안 작성·저장 (status=draft)
- approval: interrupt 로 멈추고 관리자 승인/반려를 기다림 → 승인 시에만 발행·예약·격리 실행
모든 단계는 audit_log 에 에이전트 행동 로그로 남는다.
"""
import uuid
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from maebssi import llm
from maebssi.equipment.agent import get_agent as get_equipment_agent, replay_window
from maebssi.knowledge.failure_graph import failure_modes_for_symptoms
from maebssi.knowledge.rag import get_agent as get_knowledge_agent
from maebssi.workorder import tools as wo

TAG_QUERY = {
    "과열": "{kind} 열화상 온도 상승 과열 주요 원인과 점검 절차",
    "과전류": "{kind} 주행 모터 전류 급증 과전류 원인과 점검 절차",
    "분진": "{kind} 분진 농도 상승 원인과 조치",
}
KIND_KO = {"oht": "OHT", "agv": "AGV"}


class IncidentState(TypedDict, total=False):
    session_id: str
    upto: int | None
    assessment: dict
    knowledge: list
    failure_modes: list
    plan: dict
    wo_id: str
    draft: dict
    decision: dict
    outcome: str
    trace: list


def _trace(state, node, msg, **detail):
    wo.log_agent_step(f"agent:{node}", node, {"session_id": state.get("session_id"), "msg": msg, **detail})
    return (state.get("trace") or []) + [{"node": node, "msg": msg, "at": datetime.now().isoformat(timespec="seconds")}]


# ── 노드 ────────────────────────────────────────────────────────────────────────
def monitor(state: IncidentState) -> dict:
    frames, thumbs, _ = replay_window(state["session_id"], state.get("upto"))
    a = get_equipment_agent().assess(frames, thumbs)
    return {"assessment": a.to_dict(), "trace": _trace(state, "monitor", a.summary())}


def route(state: IncidentState) -> str:
    s = state["assessment"]["state"]
    return "diagnose" if s >= 2 else ("watch" if s == 1 else "normal")


def watch(state: IncidentState) -> dict:
    msg = f"{state['assessment']['device_id']} 관심 단계: 모니터링 주기 단축(1분→10초), 다음 정기 점검에 확인 항목 추가 [MNT-RSK-002]"
    return {"outcome": "watch", "trace": _trace(state, "watch", msg)}


def normal(state: IncidentState) -> dict:
    return {"outcome": "normal", "trace": _trace(state, "monitor", "정상 — 조치 불필요")}


def diagnose(state: IncidentState) -> dict:
    a = state["assessment"]
    kind = KIND_KO[a["kind"]]
    tags = a["symptom_tags"] or ["과열"]
    other = "AGV" if kind == "OHT" else "OHT"
    ka = get_knowledge_agent()
    results = []
    for tag in tags:
        r = ka.answer(TAG_QUERY[tag].format(kind=kind))
        if not r["refused"]:  # 다른 설비 종류 전용 항목 제외
            r["answer"] = "\n".join(ln for ln in r["answer"].splitlines() if f"({other})" not in ln)
        results.append({"topic": tag, **{k: r[k] for k in ("answer", "citations", "refused")}})
    policy = ka.answer(f"{a['state_name']} 단계 대응 시간과 승인자")
    results.append({"topic": "대응기준", **{k: policy[k] for k in ("answer", "citations", "refused")}})
    safety = ka.answer("정비 전 잠금 표지 LOTO 안전 절차")
    results.append({"topic": "안전", **{k: safety[k] for k in ("answer", "citations", "refused")}})
    # 공공 지침(KOSHA)의 정비·보수 절차 — 코퍼스에 있을 때만
    proc = ka.answer("정비보수 계획서 수립 시 에너지 차단 및 격리 계획")
    if not proc["refused"] and any("KOSHA" in c for c in proc["citations"]):
        results.append({"topic": "정비절차", **{k: proc[k] for k in ("answer", "citations", "refused")}})
    # 지식그래프: 증상 → 구동 전동기 고장모드 후보
    fms = failure_modes_for_symptoms(tags)
    cites = sorted({c for r in results for c in r["citations"]})
    msg = f"지식 검색 {len(results)}건, 인용 {len(cites)}개" + (f", 고장모드 후보 {len(fms)}개" if fms else "")
    return {"knowledge": results, "failure_modes": fms, "trace": _trace(state, "diagnose", msg, citations=cites)}


def plan(state: IncidentState) -> dict:
    a = state["assessment"]
    priority = wo.PRIORITY[a["state"]]
    tags = a["symptom_tags"] or ["과열"]
    est = max(wo.TAG_EST_HOURS[t] for t in tags)
    now = datetime.fromisoformat(a["timestamp"])
    impact = wo.production_impact(a["device_id"], now, est)
    if priority == "P1":
        start = now + timedelta(minutes=30)
    else:  # P2: SLA(24h) 안에서 반송 부하가 가장 낮은 시간대
        start = datetime.fromisoformat(impact["lowest_load_window"]["start"])
    parts = wo.recommend_parts(a["kind"], tags)
    tech = wo.find_technician(a["kind"], tags, start, est)
    p = {"priority": priority, "tags": tags, "est_hours": est, "planned_start": start.isoformat(sep=" "),
         "impact": impact, "parts": parts, "technician": tech,
         "history": wo.maintenance_history(a["device_id"], 5, before=a["timestamp"][:10]), "device": wo.get_device(a["device_id"])}
    shortage = [x["part_no"] for x in parts if x["shortage"] and x["required"]]
    msg = (f"{priority}, 착수 {p['planned_start']}, 기술자 {tech['tech_id'] if tech else '없음'}, "
           f"라인 능력 손실 {impact.get('capacity_loss_pct')}%" + (f", 재고 부족 {shortage}" if shortage else ""))
    return {"plan": p, "trace": _trace(state, "plan", msg)}


def _template_body(a, p, knowledge, failure_modes=()) -> str:
    lines = [f"■ 설비: {a['device_id']} ({KIND_KO[a['kind']]}, {p['device'].get('line')} {p['device'].get('zone')})",
             f"■ 위험도: {a['state_name']} (확률 {a['probabilities'][a['state_name']]:.0%}), 이상점수 {a['anomaly_score']:.2f}",
             f"■ 감지 시각: {a['timestamp']}",
             "■ 이상 증상:"]
    lines += [f"  - {s['label']} {s['value']:.1f} (정상 중앙값 {s['normal_median']:.1f}, 95백분위 {s['normal_p95']:.1f})"
              for s in a["top_signals"]] or ["  - 복합 신호 기반 모델 판정"]
    lines.append("■ 추정 원인·권장 조치 (근거 문서):")
    seen = set()
    for k in knowledge:
        if not k["refused"] and k["topic"] not in ("대응기준", "안전", "정비절차"):
            new = [ln for ln in k["answer"].splitlines() if ln not in seen][:4]
            seen.update(new)
            if new:
                lines.append(f"  [{k['topic']}]")
                lines += ["    " + ln for ln in new]
    if failure_modes:
        lines.append("■ 구동 전동기 고장모드 후보 (지식그래프 " + failure_modes[0]["source"] + "):")
        lines.append("  " + ", ".join(f"{f['failure_mode']}({f['sensor']})" for f in failure_modes))
    lines.append("■ 필요 부품 (★출고 예약 / ☆점검 후 필요 시):")
    for x in p["parts"]:
        s = f"  {'★' if x['required'] else '☆'} {x['part_no']} {x['name']} × {x['qty']} (재고 {x['stock']})"
        if x["shortage"] and x["required"]:
            s += f" → 부족 {x['shortage']}개, 긴급 구매 필요(리드타임 {x['lead_time_days']}일)"
        lines.append(s)
    t = p["technician"]
    lines.append(f"■ 배정: {t['tech_id']} {t['name']} ({t['skills']}, {t['shift']})" if t else "■ 배정: 가용 기술자 없음 — 수동 배정 필요")
    imp = p["impact"]
    w = imp["if_stop_now"] if p["priority"] == "P1" else imp["lowest_load_window"]
    lines.append(f"■ 일정: {p['planned_start']} 착수, 예상 {p['est_hours']}시간")
    lines.append(f"■ 생산 영향: {imp['line']} 가동 {imp['fleet_in_service']}대 중 1대 정지(능력 −{imp['capacity_loss_pct']}%), "
                 f"영향 반송 약 {w['moves_affected']}건, 긴급 LOT {w['hot_lots_in_window']}건")
    if p["history"]:
        h = p["history"][0]
        lines.append(f"■ 최근 정비 이력: {h['date']} {h['symptom']} → {h['action']}")
    lines.append("■ 안전: 정비 전 LOTO(잠금·표지) 절차 준수 [MNT-SAF-008]")
    proc = next((k for k in knowledge if k["topic"] == "정비절차"), None)
    if proc:
        lines.append("■ 공공 지침: 정비·보수 계획서에 에너지 차단·격리 계획 포함 " + " ".join(proc["citations"][:2]))
    return "\n".join(lines)


def draft(state: IncidentState) -> dict:
    a, p, k = state["assessment"], state["plan"], state["knowledge"]
    title = f"[{p['priority']}] {a['device_id']} {a['state_name']} — {'·'.join(p['tags'])} 이상 정비"
    body = _template_body(a, p, k, state.get("failure_modes") or [])
    polished = llm.complete(
        system=("당신은 제조 설비 정비 작업지시서를 작성하는 도우미입니다. 주어진 초안의 사실(수치·부품번호·시간·인용)을 "
                "바꾸지 말고, 현장 기술자가 읽기 쉽게 한국어로 정리하세요. 인용 표기 [..] 는 유지하세요."),
        user=body, max_tokens=3000)
    evidence = {"assessment": a, "knowledge": k, "history": p["history"]}
    wo_id = wo.create_draft(a["device_id"], a["state_name"], p["priority"], title, polished or body,
                            p["parts"], p["technician"], p["planned_start"], p["est_hours"], p["impact"], evidence)
    d = wo.get_work_order(wo_id)
    return {"wo_id": wo_id, "draft": {k2: d[k2] for k2 in ("wo_id", "title", "body", "priority", "planned_start", "tech_id")},
            "trace": _trace(state, "draft", f"작업지시 초안 {wo_id} 작성 — 관리자 승인 대기")}


def approval(state: IncidentState) -> dict:
    decision = interrupt({"type": "approval_required", "wo_id": state["wo_id"], "draft": state["draft"],
                          "required_approvers": "생산관리자,정비파트장" if state["plan"]["priority"] == "P1" else "정비파트장"})
    try:
        res = wo.decide(state["wo_id"], bool(decision.get("approve")), decision.get("approver", "unknown"),
                        decision.get("note", ""), decision.get("edits"))
    except (PermissionError, ValueError) as e:
        # 승인 요건 미충족 → 다시 승인 대기
        return {"decision": {"error": str(e)}, "trace": _trace(state, "approval", f"승인 거부됨: {e}")}
    return {"decision": res, "outcome": res["status"],
            "trace": _trace(state, "approval", f"{res['status']} by {decision.get('approver')}: {', '.join(res['actions'])}")}


def after_approval(state: IncidentState) -> str:
    return "approval" if "error" in (state.get("decision") or {}) else END


@lru_cache(maxsize=1)
def build_graph():
    g = StateGraph(IncidentState)
    for name, fn in [("monitor", monitor), ("watch", watch), ("normal", normal), ("diagnose", diagnose),
                     ("plan", plan), ("draft", draft), ("approval", approval)]:
        g.add_node(name, fn)
    g.add_edge(START, "monitor")
    g.add_conditional_edges("monitor", route, {"diagnose": "diagnose", "watch": "watch", "normal": "normal"})
    g.add_edge("watch", END)
    g.add_edge("normal", END)
    g.add_edge("diagnose", "plan")
    g.add_edge("plan", "draft")
    g.add_edge("draft", "approval")
    g.add_conditional_edges("approval", after_approval, {"approval": "approval", END: END})
    return g.compile(checkpointer=InMemorySaver())


def _result(thread_id: str, out: dict) -> dict:
    pending = out.get("__interrupt__")
    res = {k: v for k, v in out.items() if k != "__interrupt__"}
    res["thread_id"] = thread_id
    res["pending_approval"] = pending[0].value if pending else None
    return res


def start_incident(session_id: str, upto: int | None = None) -> dict:
    thread_id = f"inc-{uuid.uuid4().hex[:8]}"
    out = build_graph().invoke({"session_id": session_id, "upto": upto, "trace": []},
                               {"configurable": {"thread_id": thread_id}})
    return _result(thread_id, out)


def resume_incident(thread_id: str, approve: bool, approver: str, note: str = "", edits: dict | None = None) -> dict:
    out = build_graph().invoke(Command(resume={"approve": approve, "approver": approver, "note": note, "edits": edits}),
                               {"configurable": {"thread_id": thread_id}})
    return _result(thread_id, out)


def get_incident(thread_id: str) -> dict[str, Any]:
    snap = build_graph().get_state({"configurable": {"thread_id": thread_id}})
    res = dict(snap.values)
    res["thread_id"] = thread_id
    res["pending_approval"] = snap.interrupts[0].value if snap.interrupts else None
    return res
