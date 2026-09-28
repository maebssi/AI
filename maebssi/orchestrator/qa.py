"""관리자 질의 시나리오 — 자연어 질문을 적절한 전문 에이전트/도구로 라우팅.

- 설비 ID 가 포함된 상태 질문 → 설비 상태 에이전트 + 정비 이력·작업지시
- 작업지시 질문             → 작업지시 목록
- 부품·재고 질문            → 부품 재고 조회
- 생산·품질 데이터 질문     → 생산·품질 분석 에이전트(자연어→SQL, AI4I 가공 데이터)
- 그 외                      → 설비 지식 에이전트(RAG, 근거 없으면 거절)
"""
import re

from maebssi.analytics import nl2sql
from maebssi.equipment.agent import get_agent as get_equipment_agent, list_replay_sessions, replay_window
from maebssi.knowledge.rag import get_agent as get_knowledge_agent
from maebssi.workorder import tools as wo
from maebssi.workorder.db import session

DEVICE_RE = re.compile(r"(agv|oht)\s*-?\s*(\d{1,2})\s*(?:호기|번)?", re.I)
PART_RE = re.compile(r"\b([A-Z]{2,4}-[A-Z0-9-]+)\b")
STATUS_WORDS = ("상태", "위험", "괜찮", "이상", "어때", "점검 필요", "현재")
WO_WORDS = ("작업지시", "워크오더", "work order", "승인 대기", "발행")
PART_WORDS = ("재고", "부품")
# 생산·품질 데이터(가공 설비) 조회 신호: 데이터 열 동의어 또는 집계 표현
ANALYTICS_RE = re.compile(r"토크|회전\s*속도|공구\s*마모|공정\s*온도|공기\s*온도|고장률|불량률|(등급|타입)\s*별|"
                          r"(과부하|방열|전력|무작위|공구 마모)\s*고장|가공|생산 데이터|품질 데이터|몇\s*건|건수")


def route(question: str) -> str:
    q = question.lower()
    if any(w in q for w in WO_WORDS):
        return "work_orders"
    if any(w in question for w in PART_WORDS) and ("재고" in question or PART_RE.search(question)):
        return "parts"
    if DEVICE_RE.search(question) and any(w in question for w in STATUS_WORDS):
        return "equipment_status"
    if ANALYTICS_RE.search(question) and not re.search(r"원인|방법|절차|어떻게|왜|주기", question):
        return "analytics"
    return "knowledge"


def device_status(device_id: str) -> dict:
    sessions = list_replay_sessions()
    s = sessions[sessions.device_id == device_id].sort_values("start")
    out = {"device": wo.get_device(device_id), "history": wo.maintenance_history(device_id, 3),
           "work_orders": wo.list_work_orders(device_id=device_id)}
    if s.empty:
        out["assessment"] = None
        out["note"] = "실시간 센서 스트림이 연결되지 않은 설비입니다(시연 스트림은 17·18호기만 제공)."
        return out
    frames, thumbs, _ = replay_window(s.session_id.iloc[-1])
    a = get_equipment_agent().assess(frames, thumbs)
    out["assessment"] = a.to_dict()
    out["summary"] = a.summary()
    return out


def ask(question: str) -> dict:
    intent = route(question)
    if intent == "equipment_status":
        kind, num = DEVICE_RE.search(question).groups()
        dev = f"{kind.lower()}{int(num):02d}"
        r = device_status(dev)
        answer = r.get("summary") or r.get("note")
        return {"intent": intent, "answer": answer, "data": r}
    if intent == "work_orders":
        status = "draft" if "승인 대기" in question or "초안" in question else None
        m = DEVICE_RE.search(question)
        dev = f"{m.group(1).lower()}{int(m.group(2)):02d}" if m else None
        rows = wo.list_work_orders(status=status, device_id=dev)
        answer = "\n".join(f"- {r['wo_id']} {r['title']} [{r['status']}]" for r in rows) or "해당 작업지시가 없습니다."
        return {"intent": intent, "answer": answer, "data": rows}
    if intent == "parts":
        wo.ensure_db()
        with session() as con:
            m = PART_RE.search(question)
            if m:
                rows = [dict(r) for r in con.execute("SELECT * FROM parts WHERE part_no=?", (m.group(1),))]
            else:
                rows = [dict(r) for r in con.execute("SELECT * FROM parts ORDER BY stock - safety_stock")]
        answer = "\n".join(f"- {r['part_no']} {r['name']}: 재고 {r['stock']} (안전재고 {r['safety_stock']})"
                           + (" ⚠ 안전재고 미달" if r["stock"] < r["safety_stock"] else "") for r in rows) or "해당 부품이 없습니다."
        return {"intent": intent, "answer": answer, "data": rows}
    if intent == "analytics":
        r = nl2sql.run(question)
        return {"intent": intent, "answer": f"{r['answer']}\n(SQL: {r['sql']})", "data": r}
    r = get_knowledge_agent().answer(question)
    return {"intent": "knowledge", "answer": r["answer"], "data": r}
