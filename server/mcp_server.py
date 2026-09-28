"""MCP(Model Context Protocol) 도구 서버 — 전문 에이전트 기능을 표준 도구로 노출.

Claude Desktop/Claude Code 등 MCP 클라이언트나 LLM 오케스트레이터가 같은 도구를 호출할 수 있다.
작업지시 '발행'은 노출하지 않는다(초안까지만, 승인은 사람이 FastAPI/대시보드에서 수행).

설치: pip install mcp
실행: python -m server.mcp_server   (stdio)
"""
from mcp.server.fastmcp import FastMCP

from maebssi.analytics import nl2sql, quality
from maebssi.equipment.agent import get_agent as get_equipment_agent, list_replay_sessions, replay_window
from maebssi.knowledge.failure_graph import failure_modes_for_symptoms
from maebssi.knowledge.rag import get_agent as get_knowledge_agent
from maebssi.orchestrator import graph
from maebssi.workorder import tools as wo

mcp = FastMCP("maebssi-factory")


@mcp.tool()
def list_equipment_streams() -> list[dict]:
    """시연용 설비 센서 스트림(세션) 목록."""
    df = list_replay_sessions()
    df["start"] = df["start"].astype(str)
    return df.to_dict("records")


@mcp.tool()
def assess_equipment(session_id: str, upto: int | None = None) -> dict:
    """설비 세션의 upto 번째 프레임 시점 위험도(정상/관심/경고/위험), 확률, 이상점수, 판정 근거 신호."""
    frames, thumbs, _ = replay_window(session_id, upto)
    return get_equipment_agent().assess(frames, thumbs).to_dict()


@mcp.tool()
def search_maintenance_knowledge(question: str) -> dict:
    """정비 지식 문서 검색·답변(출처 인용 포함, 근거 없으면 거절)."""
    r = get_knowledge_agent().answer(question)
    return {k: r[k] for k in ("answer", "refused", "citations", "hits")}


@mcp.tool()
def failure_modes(symptoms: list[str], asset: str = "electric motor") -> list[dict]:
    """증상 태그(과열/과전류/분진) → 고장모드 후보 (AssetOpsBench FailureSensorIQ 지식그래프)."""
    return failure_modes_for_symptoms(symptoms, asset)


@mcp.tool()
def query_production_data(question: str) -> dict:
    """생산·품질(가공 설비) 데이터에 대한 한국어 질문 → SQL 조회 결과 (읽기 전용)."""
    return nl2sql.run(question)


@mcp.tool()
def predict_machining_failure(air_temp_k: float, process_temp_k: float, rpm: float, torque_nm: float,
                              tool_wear_min: float, product_type: str = "L") -> dict:
    """가공 조건 → 고장 확률, 원인 변수, 권장 조치."""
    return quality.predict(dict(air_temp_k=air_temp_k, process_temp_k=process_temp_k, rpm=rpm,
                                torque_nm=torque_nm, tool_wear_min=tool_wear_min, product_type=product_type))


@mcp.tool()
def maintenance_history(device_id: str, limit: int = 5) -> list[dict]:
    """설비 정비 이력(CMMS 모의 데이터)."""
    return wo.maintenance_history(device_id, limit)


@mcp.tool()
def check_parts(part_numbers: dict[str, int]) -> list[dict]:
    """부품번호→필요수량 에 대한 재고·부족 수량·리드타임."""
    return wo.check_parts(part_numbers)


@mcp.tool()
def production_impact(device_id: str, at: str, hours: float = 2.0) -> dict:
    """설비 정지 시 라인 반송 능력 손실과 24시간 내 최저 부하 정비 시간대."""
    return wo.production_impact(device_id, at, hours)


@mcp.tool()
def start_incident_response(session_id: str, upto: int | None = None) -> dict:
    """이상 대응 워크플로 실행(감지→진단→계획→작업지시 초안). 발행은 관리자 승인 후에만 가능."""
    r = graph.start_incident(session_id, upto)
    return {k: r.get(k) for k in ("thread_id", "assessment", "draft", "outcome", "trace", "pending_approval")}


@mcp.tool()
def list_work_orders(status: str | None = None, device_id: str | None = None) -> list[dict]:
    """작업지시 목록 (status: draft/issued/rejected)."""
    return wo.list_work_orders(status, device_id)


if __name__ == "__main__":
    mcp.run()
