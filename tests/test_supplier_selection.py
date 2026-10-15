import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import STORES, DEPARTMENTS
from src.data import SUPPLIERS, select_supplier, load_or_generate
from src import db, inventory


def test_supplier_selection_prefers_shortest_lead_time():
    for dept in DEPARTMENTS:
        supplier, lead_time = select_supplier("CA_1", dept)
        assert (supplier, lead_time) == min(SUPPLIERS[dept], key=lambda choice: (choice[1], choice[0]))
        assert lead_time == min(choice[1] for choice in SUPPLIERS[dept])


def test_supplier_leaders_min_lead_times():
    expected = {
        "FOODS": ("FreshCo", 3),
        "HOBBIES": ("ToyWorld", 7),
        "HOUSEHOLD": ("CleanPro", 4),
    }
    for dept, expected_supplier in expected.items():
        assert select_supplier("ANY_STORE", dept) == expected_supplier


def test_selected_supplier_is_persisted_to_inventory():
    db.init_db()
    load_or_generate(force=True)
    inventory.seed_inventory_from_data()
    for store in STORES:
        for dept in DEPARTMENTS:
            rows = inventory.get_inventory(store, dept)
            assert rows and rows[0]["supplier"] == select_supplier(store, dept)[0]
            assert rows[0]["lead_time_days"] == select_supplier(store, dept)[1]
