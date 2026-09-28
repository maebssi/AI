"""작업지시 에이전트 도구 — 정비 이력·부품 재고·기술자 일정·생산 영향 분석·작업지시 초안/승인.

오케스트레이터와 FastAPI(및 향후 MCP 서버)가 같은 함수를 도구로 호출한다.
"""
import json
from datetime import datetime, timedelta

from maebssi.workorder.db import audit, ensure_db, session

PRIORITY = {3: "P1", 2: "P2", 1: "P3"}
SLA_HOURS = {"P1": 1, "P2": 24, "P3": 72}
# 증상 태그 → 권장 부품 (지식 문서 MNT-TRB-003/004/005 의 '관련 부품' 절과 일치)
# required=True 는 승인 시 출고 예약, False 는 점검 결과에 따라 필요한 후보(재고만 확인)
TAG_PARTS = {
    "과열": [("FAN-80-24V", 1, True), ("TG-100", 1, True), ("BRG-6204ZZ", 2, False)],
    "과전류": [("WHL-PU-150", 2, True), ("GBX-30", 1, False), ("MTR-DRV-400W", 1, False), ("DRV-400", 1, False)],
    "분진": [("BRK-PAD-A", 2, True), ("PM-FLT-01", 1, True)],
}
TAG_EST_HOURS = {"과열": 1.5, "과전류": 3.0, "분진": 1.0}


def _ts(s) -> datetime:
    return s if isinstance(s, datetime) else datetime.fromisoformat(str(s))


def get_device(device_id: str) -> dict:
    ensure_db()
    with session() as con:
        r = con.execute("SELECT * FROM devices WHERE device_id=?", (device_id,)).fetchone()
        return dict(r) if r else {}


def maintenance_history(device_id: str, limit: int = 5, before: str | None = None) -> list[dict]:
    ensure_db()
    with session() as con:
        rows = con.execute("SELECT date,symptom,cause,action,parts_used,downtime_min FROM maintenance_history "
                           "WHERE device_id=? AND date<=? ORDER BY date DESC LIMIT ?",
                           (device_id, before or "9999", limit)).fetchall()
        return [dict(r) for r in rows]


def recommend_parts(kind: str, tags: list[str]) -> list[dict]:
    need, required = {}, set()
    for t in tags:
        for p, q, req in TAG_PARTS.get(t, []):
            need[p] = max(need.get(p, 0), q)
            if req:
                required.add(p)
    parts = check_parts(need, kind)
    for x in parts:
        x["required"] = x["part_no"] in required
    return sorted(parts, key=lambda x: not x["required"])


def check_parts(need: dict[str, int], kind: str | None = None) -> list[dict]:
    ensure_db()
    out = []
    with session() as con:
        for part_no, qty in need.items():
            r = con.execute("SELECT * FROM parts WHERE part_no=?", (part_no,)).fetchone()
            if not r or (kind and r["applies_to"] not in ("all", kind)):
                continue
            out.append({"part_no": part_no, "name": r["name"], "qty": qty, "stock": r["stock"],
                        "shortage": max(0, qty - r["stock"]), "lead_time_days": r["lead_time_days"],
                        "below_safety_after_use": r["stock"] - qty < r["safety_stock"]})
    return out


def production_impact(device_id: str, at: str | datetime, hours: float = 2.0) -> dict:
    """해당 설비를 at 부터 hours 동안 멈출 때 라인 반송 능력 손실과, 24시간 내 부하가 가장 낮은 정비 시간대."""
    ensure_db()
    at = _ts(at).replace(minute=0, second=0, microsecond=0)
    dev = get_device(device_id)
    with session() as con:
        fleet = con.execute("SELECT COUNT(*) FROM devices WHERE line=? AND status='운행'", (dev["line"],)).fetchone()[0]
        rows = con.execute("SELECT hour_ts,planned_moves,hot_lots FROM production_schedule WHERE line=? "
                           "AND hour_ts>=? AND hour_ts<? ORDER BY hour_ts",
                           (dev["line"], at.isoformat(sep=" "), (at + timedelta(hours=24)).isoformat(sep=" "))).fetchall()
    rows = [dict(r) for r in rows]
    if not rows:
        return {"line": dev["line"], "fleet": fleet, "note": "생산 계획 데이터 없음"}
    n = max(1, int(round(hours)))
    share = 1 / max(fleet, 1)
    now_moves = sum(r["planned_moves"] for r in rows[:n])
    now_hot = sum(r["hot_lots"] for r in rows[:n])
    best_i = min(range(len(rows) - n + 1), key=lambda i: sum(r["planned_moves"] + 50 * r["hot_lots"] for r in rows[i:i + n]))
    best = rows[best_i:best_i + n]
    return {
        "line": dev["line"], "fleet_in_service": fleet,
        "capacity_loss_pct": round(100 * share, 1),
        "if_stop_now": {"start": rows[0]["hour_ts"], "hours": n, "moves_affected": int(now_moves * share),
                        "hot_lots_in_window": now_hot},
        "lowest_load_window": {"start": best[0]["hour_ts"], "hours": n,
                               "moves_affected": int(sum(r["planned_moves"] for r in best) * share),
                               "hot_lots_in_window": sum(r["hot_lots"] for r in best)},
    }


def find_technician(kind: str, tags: list[str], start: str | datetime, hours: float) -> dict | None:
    """설비 종류·필요 기술(전기/기계)·근무조·기존 배정 충돌을 고려해 기술자 1명 선택."""
    ensure_db()
    start = _ts(start)
    end = start + timedelta(hours=hours)
    shift = "주간" if 8 <= start.hour < 20 else "야간"
    need = {"전기"} if "과전류" in tags else set()
    if {"과열", "분진"} & set(tags):
        need.add("기계")
    with session() as con:
        techs = [dict(r) for r in con.execute("SELECT * FROM technicians WHERE on_duty=1")]
        busy = {r["tech_id"] for r in con.execute(
            "SELECT tech_id FROM tech_bookings WHERE start_ts<? AND end_ts>?",
            (end.isoformat(sep=" "), start.isoformat(sep=" ")))}

    def score(t):
        skills = set(t["skills"].split(","))
        return (kind.upper() in skills) * 4 + len(need & skills) * 2 + (t["shift"] == shift) * 3 - len(skills) * 0.1

    cands = [t for t in techs if t["tech_id"] not in busy and kind.upper() in t["skills"]]
    if not cands:
        return None
    best = max(cands, key=score)
    best["matched_skills"] = sorted(need & set(best["skills"].split(",")))
    best["shift_match"] = best["shift"] == shift
    return best


def create_draft(device_id: str, risk_state: str, priority: str, title: str, body: str,
                 parts: list, tech: dict | None, planned_start: str, est_hours: float,
                 impact: dict, evidence: dict, actor: str = "agent:orchestrator") -> str:
    ensure_db()
    with session() as con:
        n = con.execute("SELECT COUNT(*) FROM work_orders").fetchone()[0] + 1
        wo_id = f"WO-{datetime.now():%y%m%d}-{n:04d}"
        con.execute("INSERT INTO work_orders VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (wo_id, device_id, datetime.now().isoformat(timespec="seconds"), priority, risk_state,
                     title, body, json.dumps(parts, ensure_ascii=False), tech["tech_id"] if tech else None,
                     planned_start, est_hours, json.dumps(impact, ensure_ascii=False),
                     json.dumps(evidence, ensure_ascii=False, default=str), "draft", None, None, None))
        audit(con, actor, "create_draft", {"wo_id": wo_id, "device_id": device_id, "priority": priority})
    return wo_id


def get_work_order(wo_id: str) -> dict:
    ensure_db()
    with session() as con:
        r = con.execute("SELECT * FROM work_orders WHERE wo_id=?", (wo_id,)).fetchone()
    if not r:
        return {}
    d = dict(r)
    for k in ("parts", "impact", "evidence"):
        d[k] = json.loads(d[k]) if d[k] else None
    return d


def list_work_orders(status: str | None = None, device_id: str | None = None) -> list[dict]:
    ensure_db()
    q, args = "SELECT wo_id,device_id,created_at,priority,risk_state,title,status,approver,tech_id,planned_start FROM work_orders WHERE 1=1", []
    if status:
        q += " AND status=?"
        args.append(status)
    if device_id:
        q += " AND device_id=?"
        args.append(device_id)
    with session() as con:
        return [dict(r) for r in con.execute(q + " ORDER BY created_at DESC", args)]


def decide(wo_id: str, approve: bool, approver: str, note: str = "", edits: dict | None = None) -> dict:
    """관리자 승인/반려. 승인 시에만 발행(issued)·기술자 일정 예약·부품 출고·위험 설비 격리를 실행한다."""
    wo = get_work_order(wo_id)
    if not wo:
        raise KeyError(wo_id)
    if wo["status"] != "draft":
        raise ValueError(f"{wo_id} 는 이미 {wo['status']} 상태입니다")
    if wo["priority"] == "P1" and approve and "," not in approver:
        # P1 은 생산 관리자 + 정비 파트장 공동 승인 (MNT-WO-009)
        raise PermissionError("P1 작업지시는 '생산관리자,정비파트장' 형태로 2인 승인이 필요합니다")
    now = datetime.now().isoformat(timespec="seconds")
    with session() as con:
        if edits:
            for k in ("tech_id", "planned_start", "est_hours", "title", "body"):
                if k in edits:
                    con.execute(f"UPDATE work_orders SET {k}=? WHERE wo_id=?", (edits[k], wo_id))
                    wo[k] = edits[k]
        status = "issued" if approve else "rejected"
        con.execute("UPDATE work_orders SET status=?,approver=?,decided_at=?,decision_note=? WHERE wo_id=?",
                    (status, approver, now, note, wo_id))
        actions = []
        if approve:
            if wo["tech_id"]:
                start = _ts(wo["planned_start"])
                con.execute("INSERT INTO tech_bookings (tech_id,start_ts,end_ts,wo_id) VALUES (?,?,?,?)",
                            (wo["tech_id"], start.isoformat(sep=" "),
                             (start + timedelta(hours=float(wo["est_hours"]))).isoformat(sep=" "), wo_id))
                actions.append(f"기술자 {wo['tech_id']} 일정 예약")
            for p in wo["parts"] or []:
                if not p.get("required", True):
                    if p["stock"] < p["qty"]:
                        actions.append(f"후보 부품 {p['part_no']} 재고 부족 — 점검 후 필요 시 구매")
                    continue
                take = min(p["qty"], p["stock"])
                if take:
                    con.execute("UPDATE parts SET stock=stock-? WHERE part_no=?", (take, p["part_no"]))
                    actions.append(f"{p['part_no']} {take}개 출고 예약")
                if p["shortage"]:
                    actions.append(f"{p['part_no']} {p['shortage']}개 긴급 구매 요청")
            if wo["priority"] == "P1":
                con.execute("UPDATE devices SET status='격리' WHERE device_id=?", (wo["device_id"],))
                actions.append(f"{wo['device_id']} 운행 제외(격리)")
        audit(con, f"human:{approver}", "approve" if approve else "reject",
              {"wo_id": wo_id, "note": note, "edits": edits, "actions": actions})
    return {"wo_id": wo_id, "status": status, "actions": actions}


def audit_log(limit: int = 50) -> list[dict]:
    ensure_db()
    with session() as con:
        return [dict(r) for r in con.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,))]


def log_agent_step(actor: str, action: str, detail: dict):
    ensure_db()
    with session() as con:
        audit(con, actor, action, detail)
