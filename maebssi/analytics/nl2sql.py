"""자연어 → SQL 변환 (생산·품질 데이터 조회).

1) LLM 사용 가능 시: 스키마 설명을 주고 SELECT 문 생성 → 검증 → 실행
2) 불가·실패 시: 규칙 기반 의미 파서 (집계·그룹·필터·상위 N 패턴)
안전: 읽기 전용 연결, 단일 SELECT 문만 허용.
"""
import re

from maebssi import llm
from maebssi.analytics.db import AI4I_COLUMNS, connect, schema_text

TABLE = "machining"
NUMERIC = ["air_temp_k", "process_temp_k", "rpm", "torque_nm", "tool_wear_min"]
FAIL_FLAGS = ["twf", "hdf", "pwf", "osf", "rnf"]
PCT_COLS = {"machine_failure", *FAIL_FLAGS}

# 동의어 → 열 (긴 표현부터 매칭해 '공구 마모 고장'이 '공구 마모'보다 먼저 잡히게 함)
SYNONYMS = sorted(((s, v[0]) for v in AI4I_COLUMNS.values() for s in v[2] + [v[0]]
                   if v[0] not in ("udi", "product_id", "product_type", "machine_failure")),
                  key=lambda x: -len(x[0]))
CMP = {"이상": ">=", "초과": ">", "이하": "<=", "미만": "<", "넘는": ">", "보다 큰": ">", "보다 작은": "<"}
AGG_WORDS = [("평균", "AVG"), ("최댓값", "MAX"), ("최대", "MAX"), ("최솟값", "MIN"), ("최소", "MIN"),
             ("합계", "SUM"), ("총합", "SUM")]
COUNT_RE = re.compile(r"몇\s*건|건수|개수|몇\s*개|얼마나 많")
RATE_RE = re.compile(r"고장률|불량률|고장 비율|비율")
FAIL_FILTER_RE = re.compile(r"고장(이|가)?\s*(발생한|난|났던|인|발생|있는)|고장난")
NO_FAIL_RE = re.compile(r"고장(이|가)?\s*없는|정상인")
TOPK_RE = re.compile(r"(가장\s*(높은|큰|많은|낮은|작은|적은)|상위|하위)\s*(\d+)?\s*(건|개)?")


class SQLValidationError(ValueError):
    pass


def validate(sql: str) -> str:
    s = sql.strip().rstrip(";").strip()
    if ";" in s or not re.match(r"(?is)^\s*(select|with)\b", s):
        raise SQLValidationError("단일 SELECT 문만 허용됩니다")
    if re.search(r"(?i)\b(insert|update|delete|drop|alter|attach|pragma|create|replace)\b", s):
        raise SQLValidationError("변경 구문은 허용되지 않습니다")
    return s


def _find_columns(q: str):
    """질문에서 열 언급을 찾아 (위치, 끝, 열) 목록으로 반환 — 겹치는 짧은 표현은 무시."""
    taken = [False] * len(q)
    found = []
    for syn, col in SYNONYMS:
        for m in re.finditer(re.escape(syn), q):
            if any(taken[m.start():m.end()]):
                continue
            for i in range(m.start(), m.end()):
                taken[i] = True
            found.append((m.start(), m.end(), col))
    return sorted(found)


def rule_based(q: str) -> str:
    cols = _find_columns(q)
    where, used = [], set()

    m = re.search(r"\b([LMH])\s*(등급|타입|제품|형|유형)", q)
    if m:
        where.append(f"product_type = '{m.group(1)}'")

    # 수치 조건: '<열>(이/가/는) <숫자>(단위) <비교어>'
    for start, end, col in cols:
        if col not in NUMERIC:
            continue
        m = re.match(r"\s*(이|가|은|는|이\s*|가\s*)?\s*(\d+(?:\.\d+)?)\s*(분|rpm|K|Nm|도)?\s*(이상|초과|이하|미만|넘는|보다 큰|보다 작은)",
                     q[end:])
        if m:
            where.append(f"{col} {CMP[m.group(4)]} {m.group(2)}")
            used.add((start, col))

    flag_cols = [c for s, e, c in cols if c in FAIL_FLAGS]
    fail_col = flag_cols[0] if flag_cols else "machine_failure"
    by_type_of_failure = bool(re.search(r"고장\s*유형별", q))
    group = "product_type" if re.search(r"(등급|타입|유형|제품)\s*별", q) and not by_type_of_failure else None

    measures = [c for s, e, c in cols if c in NUMERIC and (s, c) not in used]
    agg = next((a for w, a in AGG_WORDS if w in q), None)
    is_rate = bool(RATE_RE.search(q))
    is_count = bool(COUNT_RE.search(q)) or (bool(re.search(r"\d+\s*건", q)) and not TOPK_RE.search(q))

    # 고장 조건 필터 (고장률/고장 유형 집계가 목적이 아닐 때)
    fail_filter = FAIL_FILTER_RE.search(q) or (flag_cols and (measures or is_count) and not group)
    if NO_FAIL_RE.search(q):
        where.append("machine_failure = 0")
    elif fail_filter and not is_rate and not by_type_of_failure:
        where.append(f"{fail_col} = 1")
    elif not is_rate and not by_type_of_failure and is_count and re.search(r"고장", q) and not group:
        where.append(f"{fail_col} = 1")

    # 상위 N
    tops = [t for t in TOPK_RE.finditer(q) if t.group(3) or t.group(1) in ("상위", "하위")]
    if tops and not agg:
        n = int(next((t.group(3) for t in tops if t.group(3)), 5))
        desc = not re.search(r"낮은|작은|적은|하위", q[tops[0].start():tops[-1].end()])
        order = measures[0] if measures else "machine_failure"
        return _compose("*", where, None, f"ORDER BY {order} {'DESC' if desc else 'ASC'} LIMIT {n}")

    if by_type_of_failure:
        select = ", ".join(f"SUM({f})" for f in FAIL_FLAGS)
    elif is_rate:
        select = f"AVG({fail_col})"
    elif agg and measures:
        select = ", ".join(f"{agg}({c})" for c in measures)
    elif is_count and group and (flag_cols or re.search(r"고장", q)):
        select = f"SUM({fail_col})"
        where = [w for w in where if w != f"{fail_col} = 1"]
    elif is_count:
        select = "COUNT(*)"
    elif measures:
        select = ", ".join(f"AVG({c})" for c in measures)
    else:
        select = "COUNT(*)"
    if group:
        select = f"{group}, {select}"
    return _compose(select, where, group, "")


def _compose(select, where, group, tail):
    sql = f"SELECT {select} FROM {TABLE}"
    if where:
        sql += " WHERE " + " AND ".join(where)
    if group:
        sql += f" GROUP BY {group}"
    return (sql + (" " + tail if tail else "")).strip()


def llm_sql(q: str, con) -> str | None:
    text = llm.complete(
        system=("SQLite 전문가로서 한국어 질문을 하나의 SELECT 문으로 바꾸세요. SQL만 출력하고 설명·코드블록 표시는 쓰지 마세요.\n"
                + schema_text(con)),
        user=q, max_tokens=1000)
    if not text:
        return None
    return re.sub(r"^```\w*|```$", "", text.strip(), flags=re.M).strip()


def run(q: str, prefer_llm: bool = True, max_rows: int = 50) -> dict:
    con = connect(readonly=True)
    try:
        sql, source = None, "rule"
        if prefer_llm and llm.available():
            try:
                sql, source = validate(llm_sql(q, con) or ""), "llm"
                con.execute(f"EXPLAIN {sql}")
            except Exception:
                sql, source = None, "rule"
        if sql is None:
            sql = validate(rule_based(q))
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = [list(r) for r in cur.fetchmany(max_rows)]
    finally:
        con.close()
    return {"question": q, "sql": sql, "source": source, "columns": cols, "rows": rows,
            "answer": summarize(cols, rows)}


def summarize(cols, rows) -> str:
    def fmt(c, v):
        if v is None:
            return "-"
        inner = re.search(r"\((\w+)\)", c)
        if c.startswith("AVG(") and inner and inner.group(1) in PCT_COLS:
            return f"{v * 100:.2f}%"
        return f"{v:,.2f}" if isinstance(v, float) else f"{v:,}" if isinstance(v, int) else str(v)
    if not rows:
        return "조건에 맞는 데이터가 없습니다."
    if len(rows) == 1 and len(cols) <= 5:
        return ", ".join(f"{c} = {fmt(c, v)}" for c, v in zip(cols, rows[0]))
    lines = [" | ".join(cols)] + [" | ".join(fmt(c, v) for c, v in zip(cols, r)) for r in rows[:10]]
    return "\n".join(lines) + (f"\n… 외 {len(rows) - 10}행" if len(rows) > 10 else "")
