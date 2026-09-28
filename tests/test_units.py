"""데이터·모델 없이 돌아가는 단위 테스트 (python -m pytest tests)."""
import numpy as np
import pandas as pd

from maebssi.equipment.agent import symptom_tags
from maebssi.equipment.features import add_features, feature_columns
from maebssi.knowledge.rag import BM25, load_chunks, query_coverage, tokenize
from maebssi.orchestrator.qa import route


def _frames(n=40):
    ts = pd.date_range("2024-09-01 08:00:00", periods=n, freq="s")
    rnd = np.random.default_rng(0)
    d = {c: rnd.normal(10, 1, n) for c in ["NTC", "PM1.0", "PM2.5", "PM10", "CT1", "CT2", "CT3", "CT4",
                                           "ex_temperature", "ex_humidity", "ex_illuminance",
                                           "ir_mean", "ir_std", "ir_max", "ir_p50", "ir_p90", "ir_p99",
                                           "ir_hot_frac", "ir_center_mean", "ir_grad"]}
    return pd.DataFrame({**d, "ts": ts, "session_id": "s1", "kind": "oht", "device_id": "oht01",
                         "cumulative_operating_day": 1.0, "equipment_history": 13.0})


def test_features_are_causal():
    df = _frames()
    full = add_features(df)
    part = add_features(df.iloc[:20])
    cols = feature_columns(full)
    # 앞 20프레임 특징은 뒤 프레임 유무와 무관해야 한다 (미래 정보 누설 없음)
    np.testing.assert_allclose(full[cols].iloc[:20].to_numpy(), part[cols].to_numpy(), rtol=1e-6)


def test_symptom_tags():
    top = [{"signal": "CT2", "robust_z": 9}, {"signal": "ir_max", "robust_z": 5}]
    assert symptom_tags(top) == ["과열", "과전류"]


def test_chunks_have_doc_numbers():
    chunks = load_chunks()
    assert chunks and all(c.doc_no.startswith("MNT-") for c in chunks if c.source == "sample")
    assert all(c.doc_no.startswith("KOSHA ") for c in chunks if c.source == "kosha")


def test_bm25_prefers_matching_chunk():
    chunks = load_chunks()
    bm = BM25([tokenize(c.text) for c in chunks])
    best = chunks[int(np.argmax(bm.scores(tokenize("LOTO 잠금 표지 절차"))))]
    assert best.doc == "08_safety_loto"


def test_query_coverage_out_of_domain():
    vocab = BM25([tokenize(c.text) for c in load_chunks()]).idf
    assert query_coverage("오늘 점심 메뉴 추천해줘", vocab) < 0.5
    assert query_coverage("모터 절연저항 기준값은?", vocab) >= 0.5


def test_routing():
    assert route("oht17 지금 상태 어때?") == "equipment_status"
    assert route("승인 대기 작업지시") == "work_orders"
    assert route("DRV-400 재고") == "parts"
    assert route("과전류 원인은?") == "knowledge"


# ── 추가 데이터 기능 ──
import re

import pytest

from maebssi.analytics.nl2sql import SQLValidationError, rule_based, validate
from maebssi.config import AI4I_CSV, ASSETOPS_HF_DIR, KOSHA_DIR


def test_sql_validation_blocks_writes():
    for bad in ["DROP TABLE machining", "SELECT 1; DELETE FROM machining", "UPDATE machining SET rpm=0"]:
        with pytest.raises(SQLValidationError):
            validate(bad)
    assert validate("SELECT COUNT(*) FROM machining;") == "SELECT COUNT(*) FROM machining"


def test_rule_based_sql_patterns():
    assert rule_based("등급별 고장률") == "SELECT product_type, AVG(machine_failure) FROM machining GROUP BY product_type"
    assert rule_based("토크가 60 이상인 건수는?") == "SELECT COUNT(*) FROM machining WHERE torque_nm >= 60"
    assert "ORDER BY rpm ASC LIMIT 3" in rule_based("회전속도가 가장 낮은 3건")


@pytest.mark.skipif(not any(KOSHA_DIR.glob("KOSHA_*.pdf")) if KOSHA_DIR.exists() else True, reason="KOSHA PDF 없음")
def test_kosha_pdf_sections():
    from maebssi.knowledge.pdf_loader import load_pdf
    doc_no, title, chunks = load_pdf(next(KOSHA_DIR.glob("KOSHA_Z-30-2022.pdf")))
    assert doc_no == "KOSHA Z-30-2022" and "정비보수" in title.replace(" ", "")
    assert any(c["section"].startswith("4.1") for c in chunks)
    # 쪽 머리말(단독 줄의 'KOSHA GUIDE', 지침번호, 쪽번호)은 제거되어야 함 — 본문 속 다른 지침 인용은 유지
    for c in chunks:
        lines = c["text"].splitlines()
        assert "KOSHA GUIDE" not in lines and not any(re.fullmatch(r"-\s*\d+\s*-", l) for l in lines)


@pytest.mark.skipif(not (ASSETOPS_HF_DIR / "data").exists(), reason="AssetOpsBench 없음")
def test_failure_graph_motor_current():
    from maebssi.knowledge.failure_graph import failure_modes_for_symptoms
    fms = failure_modes_for_symptoms(["과전류"])
    assert fms and all("source" in f for f in fms)
