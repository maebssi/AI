"""FastAPI 추론 서버.

실행: uvicorn server.app:app --port 8000
문서: http://localhost:8000/docs
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from maebssi.analytics import nl2sql, quality
from maebssi.equipment.agent import get_agent as get_equipment_agent, list_replay_sessions, replay_window
from maebssi.knowledge.failure_graph import failure_modes_for_symptoms, get_graph
from maebssi.knowledge.rag import get_agent as get_knowledge_agent
from maebssi.orchestrator import graph, qa
from maebssi.workorder import tools as wo
from maebssi.workorder.db import ensure_db

app = FastAPI(title="맵씨 제조 Agentic AI — 스마트공장 운영 지원", version="0.1.0")


@app.on_event("startup")
def _warmup():
    ensure_db()
    get_equipment_agent()
    get_knowledge_agent()


class IncidentReq(BaseModel):
    session_id: str
    upto: int | None = None


class DecisionReq(BaseModel):
    approve: bool
    approver: str
    note: str = ""
    edits: dict | None = None


class AskReq(BaseModel):
    question: str


class MachiningReq(BaseModel):
    air_temp_k: float
    process_temp_k: float
    rpm: float
    torque_nm: float
    tool_wear_min: float
    product_type: str = "L"


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health():
    return {"ok": True}


# ── 설비 상태 에이전트 ──
@app.get("/equipment/sessions")
def sessions():
    df = list_replay_sessions()
    df["start"] = df["start"].astype(str)
    return df.to_dict("records")


@app.get("/equipment/assess/{session_id}")
def assess(session_id: str, upto: int | None = None):
    try:
        frames, thumbs, labels = replay_window(session_id, upto)
    except KeyError as e:
        raise HTTPException(404, str(e))
    r = get_equipment_agent().assess(frames, thumbs).to_dict()
    r["ground_truth_state"] = int(labels[-1])  # 시연용: 실제 라벨 비교
    return r


# ── 설비 지식 에이전트 ──
@app.post("/knowledge/ask")
def knowledge(req: AskReq):
    return get_knowledge_agent().answer(req.question)


@app.get("/knowledge/failure-modes")
def failure_modes(symptoms: str = "과열,과전류", asset: str = "electric motor", sensor: str | None = None):
    """증상 태그(과열·과전류·분진) 또는 센서명으로 고장모드 후보 조회 (AssetOpsBench 지식그래프)."""
    if sensor:
        g = get_graph()
        return [] if g is None else [{"failure_mode": f, "weight": w} for f, w in g.failures_for_sensor(asset, sensor)]
    return failure_modes_for_symptoms([t.strip() for t in symptoms.split(",") if t.strip()], asset)


# ── 생산·품질 분석 에이전트 ──
@app.post("/analytics/query")
def analytics_query(req: AskReq):
    try:
        return nl2sql.run(req.question)
    except nl2sql.SQLValidationError as e:
        raise HTTPException(400, str(e))


@app.post("/quality/predict")
def quality_predict(req: MachiningReq):
    return quality.predict(req.model_dump())


# ── 오케스트레이터 ──
@app.post("/incidents")
def start_incident(req: IncidentReq):
    try:
        return graph.start_incident(req.session_id, req.upto)
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.get("/incidents/{thread_id}")
def get_incident(thread_id: str):
    return graph.get_incident(thread_id)


@app.post("/incidents/{thread_id}/decision")
def decide(thread_id: str, req: DecisionReq):
    return graph.resume_incident(thread_id, req.approve, req.approver, req.note, req.edits)


@app.post("/ask")
def ask(req: AskReq):
    return qa.ask(req.question)


# ── 작업지시·감사 로그 ──
@app.get("/work-orders")
def work_orders(status: str | None = None, device_id: str | None = None):
    return wo.list_work_orders(status, device_id)


@app.get("/work-orders/{wo_id}")
def work_order(wo_id: str):
    d = wo.get_work_order(wo_id)
    if not d:
        raise HTTPException(404, wo_id)
    return d


@app.get("/audit-log")
def audit(limit: int = 50):
    return wo.audit_log(limit)
