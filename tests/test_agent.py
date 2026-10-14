import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import agent, db, inventory
from src.data import load_or_generate


def test_deterministic_forecast():
    db.init_db()
    df = load_or_generate()
    inventory.seed_inventory_from_data(df)
    r = agent._deterministic("forecast demand for CA_1 FOODS")
    assert "steps" in r and r["steps"]
    assert "projected demand" in r["answer"] or "risk" in r["answer"]


def test_risk_scan():
    db.init_db()
    df = load_or_generate()
    inventory.seed_inventory_from_data(df)
    r = agent._deterministic("run a risk scan")
    assert "Flagged" in r["answer"] or "scanned" in r["answer"].lower()
