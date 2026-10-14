import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import db, inventory
from src.data import load_or_generate


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
