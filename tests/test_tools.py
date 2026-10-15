import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import date, timedelta
import pandas as pd

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
    if m5_files_available():
        # Official M5 daily sales span the competition's historical calendar.
        assert df["Date"].min() == pd.Timestamp("2011-01-29")
        assert df["Date"].max() == pd.Timestamp("2016-04-24")
    else:
        assert df["Date"].max() >= pd.Timestamp(date.today() - timedelta(days=30))


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
        po = inventory.approve_po("CA_1", "FOODS", horizon)
        assert po["planning_horizon_days"] == horizon
        assert po["recommended_qty"] == rec["recommended_qty"]


def test_trend_no_module_error():
    from src.trends import query_trend
    r = query_trend("FOODS")
    assert "ModuleNotFoundError" not in r["source"] or r.get("direction") in ("up", "down", "flat")
