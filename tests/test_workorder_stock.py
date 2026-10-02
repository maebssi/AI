"""Approval must reserve current stock, even for older drafts."""
import pytest

from maebssi.workorder import db, tools


@pytest.fixture
def plant(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "plant.db")
    with db.session() as con:
        con.executescript(db.SCHEMA)
        con.execute("INSERT INTO parts VALUES (?,?,?,?,?,?,?)",
                    ("FAN-80-24V", "Fan", "all", 3, 0, 1, 1))
    return db


def draft():
    return tools.create_draft(
        "oht01", "WARNING", "P2", "Inspect fan", "Replace fan",
        tools.check_parts({"FAN-80-24V": 2}), None,
        "2026-10-02 10:00:00", 1, {}, {})


def test_approval_uses_current_stock(plant):
    ids = [draft() for _ in range(3)]
    results = [tools.decide(i, True, "manager") for i in ids]
    with plant.session() as con:
        stock = con.execute("SELECT stock FROM parts").fetchone()[0]
    assert stock == 0
    assert results[0]["actions"] == ["FAN-80-24V 2개 출고 예약"]
    assert results[1]["actions"] == ["FAN-80-24V 1개 출고 예약", "FAN-80-24V 1개 긴급 구매 요청"]
    assert results[2]["actions"] == ["FAN-80-24V 2개 긴급 구매 요청"]


def test_rejected_draft_does_not_reserve_stock(plant):
    wo_id = draft()
    assert tools.decide(wo_id, False, "manager")["actions"] == []
    with plant.session() as con:
        assert con.execute("SELECT stock FROM parts").fetchone()[0] == 3


def test_repeated_approval_does_not_reserve_twice(plant):
    wo_id = draft()
    tools.decide(wo_id, True, "manager")
    with pytest.raises(ValueError):
        tools.decide(wo_id, True, "manager")
    with plant.session() as con:
        assert con.execute("SELECT stock FROM parts").fetchone()[0] == 1
