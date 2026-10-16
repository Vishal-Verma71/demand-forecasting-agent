import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

from src import db, inventory
from src.data import load_or_generate, m5_files_available


def test_seed_inventory():
    db.init_db()
    df = load_or_generate()
    inventory.seed_inventory_from_data(df)
    rows = inventory.get_inventory("CA_1", "FOODS")
    assert rows and rows[0]["on_hand"] > 0


def test_stock_status():
    db.init_db()
    df = load_or_generate()
    inventory.seed_inventory_from_data(df)
    st = inventory.stock_status("CA_1", "FOODS")
    assert st["risk_level"] in ("OK", "WARNING", "CRITICAL")
    assert st["avg_daily_demand"] >= 0


def test_reorder():
    db.init_db()
    df = load_or_generate()
    inventory.seed_inventory_from_data(df)
    r = inventory.reorder_recommendation("CA_1", "FOODS")
    assert r["recommended_qty"] >= 0
    assert r["supplier"]


def test_spike():
    db.init_db()
    df = load_or_generate()
    s = inventory.detect_demand_spike("CA_1", "FOODS")
    assert "spike" in s and isinstance(s["spike"], bool)


def test_date_freshness():
    df = load_or_generate()
    # The committed table is M5-derived, so with or without the raw download the data spans
    # the competition's historical calendar.
    assert df["Date"].min() == pd.Timestamp("2011-01-29")
    assert df["Date"].max() == pd.Timestamp("2016-04-24")


def test_supplier_stability():
    db.init_db()
    df = load_or_generate()
    inventory.seed_inventory_from_data(df)
    for s in ["CA_1", "TX_1"]:
        rows = inventory.get_inventory(s)
        assert rows
        suppliers = {r["supplier"] for r in rows}
        assert all(sup and sup != "None" for sup in suppliers)


def test_reorder_horizon_consistency():
    db.init_db()
    df = load_or_generate()
    inventory.seed_inventory_from_data(df)
    for horizon in (7, 14, 28):
        rec = inventory.reorder_recommendation("CA_1", "FOODS", horizon)
        assert rec["recommended_qty"] >= 0
        po = inventory.draft_po("CA_1", "FOODS", horizon)
        assert po["planning_horizon_days"] == horizon
        assert po["recommended_qty"] == rec["recommended_qty"]


def test_trend_offline_fallback_reports_no_numbers(monkeypatch):
    from src import trends

    def offline(*_args):
        raise ConnectionError("offline")

    monkeypatch.setattr(trends, "_fetch_live_trend", offline)
    trends._cache.clear()
    r = trends.query_trend("FOODS")
    assert r["direction"] == "unavailable" and r["delta_pct"] is None


def testDiagnosticCountsAreBalancedAcrossSelectedStores():
    from src.data import data_diagnostics
    d = data_diagnostics()
    if m5_files_available():  # raw item-level counts need the (uncommitted) M5 download
        assert set(d["m5_selected_item_rows_by_store"]) == {"CA_1", "CA_2", "TX_1", "TX_2", "WI_1"}
    assert len(set(d["cached_rows_by_store"].values())) == 1
    assert d["cached_rows_by_store"]["CA_1"] == 5739


def test_forced_rebuild_without_raw_m5_keeps_real_data(tmp_path):
    import shutil
    from src.data import DATA_PATH
    cache = tmp_path / "sales.csv"
    shutil.copy(DATA_PATH, cache)
    df = load_or_generate(path=cache, force=True, raw_dir=tmp_path / "no_m5_here")
    assert df["Date"].min() == pd.Timestamp("2011-01-29")   # still M5, not the synthetic fallback
    assert df["Date"].max() == pd.Timestamp("2016-04-24")


def testRiskScanCompletesSuccessfully():
    db.init_db()
    df = load_or_generate()
    inventory.seed_inventory_from_data(df)
    result = inventory.run_risk_scan(28)
    assert result["status"] == "COMPLETED"
    assert result["items_evaluated"] == 15
    assert len(result["items"]) == result["flagged"]
    assert all(item["risk_level"] in ("CRITICAL", "WARNING") for item in result["items"])
    assert sum(result["severity_counts"].values()) == 15


def _force_shortage(store="CA_1", dept="FOODS"):
    with db.conn() as c:
        c.execute("UPDATE inventory SET on_hand=10, on_order=0 WHERE store=? AND dept=?", (store, dept))


def testDuplicateDraftPoRequestIsIdempotent():
    inventory.seed_inventory_from_data(load_or_generate())
    _force_shortage()
    first = inventory.draft_po("CA_1", "FOODS", 28)
    second = inventory.draft_po("CA_1", "FOODS", 28)
    assert first["po_ids"] and first["po_ids"] == second["po_ids"]
    assert first["status"].startswith("DRAFT created")
    assert "idempotent" in second["status"]
    assert len(db.get_pos()) == len(first["po_ids"])


def test_no_po_when_stock_is_sufficient():
    inventory.seed_inventory_from_data(load_or_generate())
    with db.conn() as c:
        c.execute("UPDATE inventory SET on_hand=1e9 WHERE store='CA_1' AND dept='FOODS'")
    po = inventory.draft_po("CA_1", "FOODS", 28)
    assert po["status"] == "NO_ORDER_NEEDED" and po["po_ids"] == [] and db.get_pos() == []


def test_po_approval_flow():
    inventory.seed_inventory_from_data(load_or_generate())
    _force_shortage()
    po_id = inventory.draft_po("CA_1", "FOODS", 28)["po_ids"][0]
    assert db.set_po_status(po_id, "APPROVED")["status"] == "APPROVED"
    with pytest.raises(ValueError):
        db.set_po_status(po_id, "REJECTED")  # only drafts can be decided


def test_supplier_allocation_splits_urgent_and_value():
    lines = inventory.allocate_suppliers("HOBBIES", qty=1000, urgent_qty=300, deadline_days=12)
    assert {line["supplier"]: line["qty"] for line in lines} == {"ToyWorld": 300, "CraftHub": 700}
    late = inventory.allocate_suppliers("HOBBIES", qty=1000, urgent_qty=0, deadline_days=8)
    assert [line["supplier"] for line in late] == ["ToyWorld"]  # CraftHub (10d) would arrive too late


def test_alerts_are_not_duplicated_by_repeated_scans():
    inventory.seed_inventory_from_data(load_or_generate())
    inventory.run_risk_scan(28)
    first = len(db.get_alerts(1000))
    inventory.run_risk_scan(28)
    assert len(db.get_alerts(1000)) == first


def test_approved_po_becomes_inbound_stock():
    inventory.seed_inventory_from_data(load_or_generate())
    _force_shortage()
    po = inventory.draft_po("CA_1", "FOODS", 28)
    before = inventory.stock_status("CA_1", "FOODS")["on_order"]
    for po_id in po["po_ids"]:
        db.set_po_status(po_id, "APPROVED")
    after = inventory.stock_status("CA_1", "FOODS")
    assert after["on_order"] == round(before + po["recommended_qty"], 2)
    assert after["risk_level"] == "OK"
    assert inventory.reorder_recommendation("CA_1", "FOODS")["recommended_qty"] == 0
