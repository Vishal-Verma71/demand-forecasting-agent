import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import db, inventory
from src.data import load_or_generate


def _controlled(on_hand: int, safety: int = 10, lead: int = 5, days_to_review: int = 3):
    load_or_generate()
    inventory.seed_inventory_from_data()
    with db.conn() as c:
        c.execute(
            "UPDATE inventory SET on_hand=?, on_order=0, safety_stock=?, lead_time_days=?, days_to_review=? "
            "WHERE store='CA_1' AND dept='FOODS'",
            (on_hand, safety, lead, days_to_review),
        )
    st = inventory.stock_status("CA_1", "FOODS")
    # Demand that must be covered until the next scheduled delivery (review + lead time).
    return st, st["protection_demand"]


def test_risk_boundary_critical():
    st, _ = _controlled(on_hand=1, safety=10, lead=5)
    assert st["risk_level"] == "CRITICAL"
    assert st["on_hand"] == 1
    assert st["protection_days"] == 5 + 3
    assert "next scheduled delivery" in st["classification_reason"]


def test_risk_boundary_ok_at_reorder_point():
    _, protection_demand = _controlled(on_hand=1, safety=10, lead=5)
    st, protection_demand = _controlled(on_hand=int(protection_demand + 11), safety=10, lead=5)
    assert st["risk_level"] == "OK"
    assert abs(st["reorder_point"] - (protection_demand + 10)) < 0.02


def test_risk_boundary_warning():
    _, protection_demand = _controlled(on_hand=1, safety=10, lead=5)
    st, _ = _controlled(on_hand=int(protection_demand + 5), safety=10, lead=5)
    assert st["risk_level"] == "WARNING"


def test_inbound_orders_count_toward_position():
    _, protection_demand = _controlled(on_hand=1, safety=10, lead=5)
    with db.conn() as c:
        c.execute("UPDATE inventory SET on_order=? WHERE store='CA_1' AND dept='FOODS'", (protection_demand + 100,))
    st = inventory.stock_status("CA_1", "FOODS")
    assert st["risk_level"] == "OK"
    assert abs(st["inventory_position"] - (1 + protection_demand + 100)) < 0.02


def test_unknown_inventory():
    db.init_db()
    load_or_generate()
    inventory.seed_inventory_from_data()
    with db.conn() as c:
        c.execute("UPDATE inventory SET on_hand=-1 WHERE store='CA_1' AND dept='FOODS'")
    st = inventory.stock_status("CA_1", "FOODS")
    assert st["risk_level"] == "UNKNOWN"
    assert st["inputs_valid"] is False


def test_complete_global_scan():
    db.init_db()
    load_or_generate()
    inventory.seed_inventory_from_data()
    result = inventory.run_risk_scan(28)
    assert result["status"] == "COMPLETED"
    assert result["items_evaluated"] == 15
    assert len(result["all_items"]) == 15
    assert sum(result["severity_counts"].values()) == result["items_evaluated"]
    assert len(result["items"]) == result["flagged"]


def test_risk_scan_api_reconciles_counts():
    from src import api

    db.init_db()
    load_or_generate()
    inventory.seed_inventory_from_data()
    result = api.risk_scan(28)
    assert result["items_evaluated"] == 15
    assert "all_items" in result
    assert sum(result["severity_counts"].values()) == result["items_evaluated"]


def test_horizon_does_not_change_lead_time():
    db.init_db()
    load_or_generate()
    inventory.seed_inventory_from_data()
    r7 = inventory.stock_status("CA_1", "FOODS", 7)
    r28 = inventory.stock_status("CA_1", "FOODS", 28)
    assert r7["lead_time_days"] == r28["lead_time_days"]
    assert r7["lead_time_demand"] == r28["lead_time_demand"]
    assert r7["forecast_horizon"] == 7
    assert r28["forecast_horizon"] == 28
    assert r7["forecast_horizon_demand"] != r28["forecast_horizon_demand"]


def test_reorder_tops_position_up_to_reorder_point():
    st, protection_demand = _controlled(on_hand=100, safety=10, lead=5)
    rec = inventory.reorder_recommendation("CA_1", "FOODS", 28)
    expected = protection_demand + 10 - 100
    assert st["risk_level"] == "CRITICAL"
    assert abs(rec["recommended_qty"] - expected) <= len(rec["allocations"])  # per-line rounding
    assert rec["allocations"][0]["supplier"] in ("FreshCo", "GlobalFoods")


def test_ok_series_gets_no_order():
    _controlled(on_hand=10**9, safety=10, lead=5)
    rec = inventory.reorder_recommendation("CA_1", "FOODS", 28)
    assert rec["risk_level"] == "OK" and rec["recommended_qty"] == 0 and rec["allocations"] == []


def test_reorder_defaults_horizon_28():
    db.init_db()
    load_or_generate()
    inventory.seed_inventory_from_data()
    rec = inventory.reorder_recommendation("CA_1", "FOODS")
    assert rec["recommended_qty"] >= 0
