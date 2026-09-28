"""생산·품질 분석 DB (SQLite, 읽기 전용 조회).

기본으로 UCI AI4I 2020(가공 설비 1만 건)을 `machining` 테이블로 적재한다.
KAMP 등 다른 CSV 도 `--add` 로 등록할 수 있다(열 설명은 schema_catalog 에 자동 기록, 필요시 수정).

  python -m maebssi.analytics.db                   # AI4I 적재
  python -m maebssi.analytics.db --add kamp.csv injection_molding
"""
import argparse
import json
import re
import sqlite3
from pathlib import Path

import pandas as pd

from maebssi.config import AI4I_CSV, ARTIFACT_DIR

ANALYTICS_DB = ARTIFACT_DIR / "analytics.db"

AI4I_COLUMNS = {
    # 원본 열 → (DB 열, 한국어 설명, 동의어)
    "UDI": ("udi", "행 번호", []),
    "Product ID": ("product_id", "제품 ID (등급 문자 + 일련번호)", ["제품번호", "제품 ID"]),
    "Type": ("product_type", "제품 품질 등급 L(저)/M(중)/H(고)", ["타입", "등급", "제품 유형", "품질 등급", "제품등급"]),
    "Air temperature [K]": ("air_temp_k", "공기 온도(K)", ["공기 온도", "공기온도", "주변 온도"]),
    "Process temperature [K]": ("process_temp_k", "공정 온도(K)", ["공정 온도", "공정온도", "가공 온도"]),
    "Rotational speed [rpm]": ("rpm", "회전 속도(rpm)", ["회전 속도", "회전속도", "rpm", "RPM", "속도"]),
    "Torque [Nm]": ("torque_nm", "토크(Nm)", ["토크"]),
    "Tool wear [min]": ("tool_wear_min", "공구 마모 시간(분)", ["공구 마모", "공구마모", "마모"]),
    "Machine failure": ("machine_failure", "설비 고장 여부(0/1)", ["고장", "불량"]),
    "TWF": ("twf", "공구 마모 고장(Tool Wear Failure)", ["공구 마모 고장", "TWF"]),
    "HDF": ("hdf", "방열 고장(Heat Dissipation Failure)", ["방열 고장", "열 방출 고장", "HDF"]),
    "PWF": ("pwf", "전력 고장(Power Failure)", ["전력 고장", "PWF"]),
    "OSF": ("osf", "과부하 고장(Overstrain Failure)", ["과부하 고장", "과부하", "OSF"]),
    "RNF": ("rnf", "무작위 고장(Random Failure)", ["무작위 고장", "랜덤 고장", "RNF"]),
}


def connect(readonly: bool = True) -> sqlite3.Connection:
    if readonly:
        ensure()
        con = sqlite3.connect(f"file:{ANALYTICS_DB.as_posix()}?mode=ro", uri=True, check_same_thread=False)
    else:
        ANALYTICS_DB.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(ANALYTICS_DB, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def _catalog(con, table, rows):
    con.execute("CREATE TABLE IF NOT EXISTS schema_catalog (table_name TEXT, column_name TEXT, description TEXT, synonyms TEXT)")
    con.execute("DELETE FROM schema_catalog WHERE table_name=?", (table,))
    con.executemany("INSERT INTO schema_catalog VALUES (?,?,?,?)",
                    [(table, c, d, json.dumps(s, ensure_ascii=False)) for c, d, s in rows])


def load_ai4i(con):
    df = pd.read_csv(AI4I_CSV, encoding="utf-8-sig")
    df = df.rename(columns={k: v[0] for k, v in AI4I_COLUMNS.items()})
    df.to_sql("machining", con, if_exists="replace", index=False)
    _catalog(con, "machining", [(v[0], v[1], v[2]) for v in AI4I_COLUMNS.values()])


def add_csv(con, path: str, table: str):
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", table):
        raise ValueError("테이블명은 영문 소문자·숫자·_ 만 허용")
    df = pd.read_csv(path, encoding_errors="replace")
    orig = list(df.columns)
    cols = []
    for i, c in enumerate(orig):
        name = re.sub(r"[^0-9a-zA-Z]+", "_", str(c)).strip("_").lower() or f"col{i}"
        cols.append(name if name not in cols else f"{name}_{i}")
    df.columns = cols
    df.to_sql(table, con, if_exists="replace", index=False)
    _catalog(con, table, [(n, str(o), [str(o)]) for n, o in zip(cols, orig)])
    return len(df)


def ensure():
    if not ANALYTICS_DB.exists():
        con = connect(readonly=False)
        load_ai4i(con)
        con.commit()
        con.close()


def schema_text(con) -> str:
    """LLM 프롬프트·설명용 스키마 요약."""
    out = []
    for (t,) in con.execute("SELECT DISTINCT table_name FROM schema_catalog"):
        n = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        out.append(f"테이블 {t} ({n}행)")
        for c, d in con.execute("SELECT column_name, description FROM schema_catalog WHERE table_name=?", (t,)):
            out.append(f"  - {c}: {d}")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", nargs=2, metavar=("CSV", "TABLE"))
    args = ap.parse_args()
    con = connect(readonly=False)
    if args.add:
        print("rows", add_csv(con, *args.add))
    else:
        load_ai4i(con)
    con.commit()
    print(schema_text(con))
