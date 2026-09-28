"""작업지시·부품 재고 모의 DB (SQLite).

실제 CMMS/MES 연동 전 개발·시연용. 모든 값은 난수 시드로 생성한 가상 데이터이다.
실행: python -m maebssi.workorder.db --reset
"""
import argparse
import json
import random
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta

from maebssi.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS devices (
  device_id TEXT PRIMARY KEY, kind TEXT, line TEXT, zone TEXT, manufacturer TEXT,
  install_year INTEGER, status TEXT DEFAULT '운행');
CREATE TABLE IF NOT EXISTS parts (
  part_no TEXT PRIMARY KEY, name TEXT, applies_to TEXT, stock INTEGER, safety_stock INTEGER,
  lead_time_days INTEGER, unit_price INTEGER);
CREATE TABLE IF NOT EXISTS technicians (
  tech_id TEXT PRIMARY KEY, name TEXT, skills TEXT, shift TEXT, on_duty INTEGER DEFAULT 1);
CREATE TABLE IF NOT EXISTS tech_bookings (
  id INTEGER PRIMARY KEY AUTOINCREMENT, tech_id TEXT, start_ts TEXT, end_ts TEXT, wo_id TEXT);
CREATE TABLE IF NOT EXISTS maintenance_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT, device_id TEXT, date TEXT, symptom TEXT, cause TEXT,
  action TEXT, parts_used TEXT, downtime_min INTEGER);
CREATE TABLE IF NOT EXISTS production_schedule (
  line TEXT, hour_ts TEXT, planned_moves INTEGER, hot_lots INTEGER, PRIMARY KEY (line, hour_ts));
CREATE TABLE IF NOT EXISTS work_orders (
  wo_id TEXT PRIMARY KEY, device_id TEXT, created_at TEXT, priority TEXT, risk_state TEXT,
  title TEXT, body TEXT, parts TEXT, tech_id TEXT, planned_start TEXT, est_hours REAL,
  impact TEXT, evidence TEXT, status TEXT, approver TEXT, decided_at TEXT, decision_note TEXT);
CREATE TABLE IF NOT EXISTS audit_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, actor TEXT, action TEXT, detail TEXT);
"""

LINES = {"oht": ("FAB1-OHT-L1", "FAB1-OHT-L2"), "agv": ("PKG-AGV-L1", "PKG-AGV-L2")}

PARTS = [
    # part_no, name, applies_to, stock, safety, lead_days, price
    ("FAN-80-24V", "제어함 냉각팬 80mm 24V", "all", 6, 4, 3, 38000),
    ("BRG-6204ZZ", "모터 베어링 6204ZZ", "all", 12, 6, 5, 9000),
    ("BRK-PAD-A", "브레이크 패드 A형", "agv", 2, 4, 7, 52000),
    ("TG-100", "방열 그리스 TG-100", "all", 9, 3, 2, 15000),
    ("MTR-DRV-400W", "주행 모터 400W", "all", 1, 2, 21, 1250000),
    ("GBX-30", "감속기 1/30", "all", 2, 1, 14, 480000),
    ("WHL-PU-150", "구동 휠 우레탄 150mm", "all", 8, 6, 10, 130000),
    ("DRV-400", "모터 드라이버 400W", "all", 0, 1, 28, 890000),
    ("PM-FLT-01", "분진 센서 필터", "agv", 20, 10, 2, 6000),
    ("GRL-OHT-50", "OHT 가이드 롤러", "oht", 14, 8, 7, 42000),
]

TECHS = [
    ("T01", "정비1조 A", "전기,OHT", "주간"), ("T02", "정비1조 B", "기계,OHT", "주간"),
    ("T03", "정비2조 A", "전기,AGV", "주간"), ("T04", "정비2조 B", "기계,AGV", "주간"),
    ("T05", "야간조 A", "전기,기계,OHT", "야간"), ("T06", "야간조 B", "전기,기계,AGV", "야간"),
    ("T07", "설비엔지니어", "전기,기계,OHT,AGV", "주간"),
]

HISTORY_TEMPLATES = [
    ("열화상 최고온도 상승", "냉각팬 고장", "냉각팬 교체", "FAN-80-24V", 45),
    ("주행 전류 증가", "구동 휠 마모", "구동 휠 교체", "WHL-PU-150", 90),
    ("주행 이상음·온도 상승", "모터 베어링 마모", "베어링 교체 및 그리스 보충", "BRG-6204ZZ,TG-100", 120),
    ("분진 농도 상승", "브레이크 패드 마모", "브레이크 패드 교체", "BRK-PAD-A", 60),
    ("간헐적 과전류 알람", "커넥터 접촉 불량", "커넥터 재체결", "", 30),
    ("정기 점검", "-", "월간 점검(절연저항·청소)", "", 40),
]


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


@contextmanager
def session():
    con = connect()
    try:
        yield con
        con.commit()
    finally:
        con.close()


def seed(con: sqlite3.Connection, seed_value: int = 7):
    rnd = random.Random(seed_value)
    con.executescript(SCHEMA)
    for kind, lines in LINES.items():
        for n in range(1, 19):
            dev = f"{kind}{n:02d}"
            con.execute("INSERT INTO devices VALUES (?,?,?,?,?,?,?)",
                        (dev, kind, lines[0] if n <= 9 else lines[1], f"Z{(n - 1) % 3 + 1}",
                         "A" if kind == "oht" else ("B" if n % 2 else "C"),
                         2024 - (13 if kind == "oht" else 7), "운행"))
            day = datetime(2024, 3, 1)
            while True:  # 데이터 수집 시작(2024-08-26) 이전의 이력만 생성
                day += timedelta(days=rnd.randint(20, 55))
                if day >= datetime(2024, 8, 20):
                    break
                t = rnd.choice(HISTORY_TEMPLATES)
                con.execute("INSERT INTO maintenance_history (device_id,date,symptom,cause,action,parts_used,downtime_min)"
                            " VALUES (?,?,?,?,?,?,?)", (dev, day.date().isoformat(), *t))
    con.executemany("INSERT INTO parts VALUES (?,?,?,?,?,?,?)", PARTS)
    con.executemany("INSERT INTO technicians (tech_id,name,skills,shift) VALUES (?,?,?,?)", TECHS)
    # 생산(반송) 계획: 시간대별 계획 반송 건수. 주간 피크, 새벽 저부하, 주말 감소
    start = datetime(2024, 8, 20)
    for kind, lines in LINES.items():
        base = 900 if kind == "oht" else 260
        for line in lines:
            for h in range(24 * 75):
                ts = start + timedelta(hours=h)
                f = 1.0 if 8 <= ts.hour < 20 else (0.45 if 2 <= ts.hour < 6 else 0.75)
                f *= 0.7 if ts.weekday() >= 5 else 1.0
                con.execute("INSERT INTO production_schedule VALUES (?,?,?,?)",
                            (line, ts.strftime("%Y-%m-%d %H:00:00"), int(base * f * rnd.uniform(0.9, 1.1)),
                             rnd.randint(0, 6) if f >= 1 else rnd.randint(0, 2)))


def reset():
    if DB_PATH.exists():
        DB_PATH.unlink()
    with session() as con:
        seed(con)


def ensure_db():
    if not DB_PATH.exists():
        reset()


def audit(con, actor: str, action: str, detail: dict):
    con.execute("INSERT INTO audit_log (ts,actor,action,detail) VALUES (?,?,?,?)",
                (datetime.now().isoformat(timespec="seconds"), actor, action,
                 json.dumps(detail, ensure_ascii=False, default=str)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()
    reset() if args.reset else ensure_db()
    with session() as con:
        for t in ("devices", "parts", "technicians", "maintenance_history", "production_schedule"):
            print(t, con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
