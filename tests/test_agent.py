import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import agent, db, inventory, tools
from src.data import load_or_generate


def test_deterministic_forecast():
    inventory.seed_inventory_from_data(load_or_generate())
    r = agent._deterministic("forecast demand for CA_1 FOODS")
    assert [s["tool"] for s in r["steps"]] == ["forecast_demand", "detect_stockout_risk"]
    assert "28-day forecast" in r["answer"] and "80% range" in r["answer"]


def test_risk_scan():
    inventory.seed_inventory_from_data(load_or_generate())
    r = agent._deterministic("run a risk scan")
    assert r["steps"][0]["tool"] == "run_risk_scan"
    assert "Scanned 15 series" in r["answer"]


_STUB = {"status": "COMPLETED", "items": [], "all_items": [], "severity_counts": {}, "items_evaluated": 15,
         "summary": {"total_demand": 1, "total_lower": 1, "total_upper": 1, "avg_daily": 1,
                     "first_day": "x", "last_day": "y"},
         "risk_level": "OK", "classification_reason": "fine", "inventory_position": 1, "on_hand": 1,
         "on_order": 0, "reorder_point": 1, "days": 14, "start": "x", "snap_promo_days": [], "events": [],
         "category": "FOODS", "query": "q", "direction": "up", "delta_pct": 1.0, "source": "s",
         "rows": [{"on_hand": 1, "on_order": 0, "safety_stock": 1, "supplier": "s", "lead_time_days": 1}],
         "recent_7d_avg": 1, "change_pct": 1, "baseline_mean": 1, "forecast_next_7d_avg": 1,
         "upcoming_change_pct": 1, "threshold": 1, "spike": False, "upcoming_spike": False,
         "po": {"po_ids": [], "rationale": "r"}}


def _route(query, monkeypatch):
    """Run the offline planner with tools stubbed; return the (tool, args) calls it made."""
    called = []
    monkeypatch.setattr(tools, "execute_tool", lambda name, args: called.append((name, args)) or dict(_STUB))
    agent._deterministic(query)
    return called


@pytest.mark.parametrize("query,expected", [
    ("Look at sales for TX_1 HOBBIES", "forecast_demand"),          # 'ok' inside 'look' is not a severity
    ("Show the report for WI_1", None),                               # 'po' inside 'report' is not an order
    ("What is the impact of the discount on FOODS?", None),          # 'count' inside 'discount'
    ("What are the external trends for HOBBIES?", "query_trends"),   # 'for <dept>' must not error
    ("Give me a forecast for next week", "forecast_demand"),
    ("Any upcoming promotions for CA_2?", "get_promo_calendar"),
    ("Plan replenishment", "run_risk_scan"),
    ("Reorder WI_1 HOUSEHOLD", "detect_stockout_risk"),
])
def test_router_intents(query, expected, monkeypatch):
    called = _route(query, monkeypatch)
    assert (called[0][0] if called else None) == expected


def test_horizon_parsing(monkeypatch):
    assert agent.parse_horizon("forecast for the next 14 days") == 14
    assert agent.parse_horizon("next week") == 7
    assert agent.parse_horizon("two weeks") == 14
    assert agent.parse_horizon("90 days") == 56
    called = _route("How many critical items in the next 14 days?", monkeypatch)
    assert called[0] == ("run_risk_scan", {"horizon": 14})


def test_unknown_store_is_rejected():
    with pytest.raises(ValueError, match="CA_3"):
        agent._deterministic("forecast for CA_3 FOODS")


def test_replenishment_workflow_end_to_end():
    inventory.seed_inventory_from_data(load_or_generate())
    with db.conn() as c:
        c.execute("UPDATE inventory SET on_hand=10, on_order=0 WHERE store='TX_1' AND dept='HOBBIES'")
    r = agent._deterministic("plan replenishment for TX_1")
    used = [s["tool"] for s in r["steps"]]
    assert used[0] == "run_risk_scan"
    for tool in ("forecast_demand", "get_promo_calendar", "detect_demand_spike", "query_trends",
                 "draft_purchase_order"):
        assert tool in used
    assert "TX_1/HOBBIES [CRITICAL]" in r["answer"] and "Draft PO #" in r["answer"]
    assert db.get_pos() and all(p["status"] == "DRAFT" for p in db.get_pos())


def test_llm_tool_loop_with_mock_client():
    """The LLM path must serialise tool results (dates) and rebuild assistant turns."""
    inventory.seed_inventory_from_data(load_or_generate())
    sent = []

    class Completions:
        turn = 0

        def create(self, **kw):
            sent.append(list(kw["messages"]))
            self.turn += 1
            if self.turn == 1:
                call = NS(id="c1", function=NS(name="forecast_demand",
                                               arguments='{"store": "CA_1", "dept": "FOODS", "horizon": 7}'))
                return NS(choices=[NS(message=NS(content=None, tool_calls=[call]))])
            return NS(choices=[NS(message=NS(content="Forecast done.", tool_calls=None))])

    client = NS(chat=NS(completions=Completions()))
    r = agent.run_agent("forecast CA_1 FOODS", client_factory=lambda: ("mock", client, "m"))
    assert r["backend"] == "mock" and r["answer"] == "Forecast done."
    assert r["steps"][0]["tool"] == "forecast_demand" and "error" not in r["steps"][0]["observation"]
    assistant_turn, tool_msg = sent[1][-2], sent[1][-1]
    assert assistant_turn["tool_calls"][0]["function"]["name"] == "forecast_demand"
    assert tool_msg["role"] == "tool" and "2016-04-25" in tool_msg["content"]
