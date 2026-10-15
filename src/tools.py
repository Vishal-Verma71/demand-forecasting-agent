"""Tool definitions and a dispatcher used by the agent."""
from __future__ import annotations

import re

from . import inventory, trends
from .config import STORES, DEPARTMENTS
from .model import forecast_demand, interval_method, MAX_FORECAST_HORIZON

_STORE_TOKEN = re.compile(r"\b([A-Za-z]{2}_\d{1,2})\b")


def _norm_store_dept(query: str):
    q = query.upper()
    store = next((s for s in STORES if re.search(rf"\b{s}\b", q)), None)
    dept = next((d for d in DEPARTMENTS if re.search(rf"\b{d}\b", q)), None)
    return store, dept


def unknown_store_tokens(query: str) -> list[str]:
    """Store-like tokens (e.g. CA_3) that are not configured stores."""
    return [t.upper() for t in _STORE_TOKEN.findall(query) if t.upper() not in STORES]


def _store_dept_schema(extra: dict | None = None, required=("store", "dept")) -> dict:
    props = {"store": {"type": "string", "enum": STORES},
             "dept": {"type": "string", "enum": DEPARTMENTS}}
    props.update(extra or {})
    return {"type": "object", "properties": props, "required": list(required)}


def _fn(name: str, description: str, parameters: dict) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


_HORIZON = {"horizon": {"type": "integer", "minimum": 1, "maximum": MAX_FORECAST_HORIZON, "default": 28}}

OPENAI_TOOLS = [
    _fn("forecast_demand",
        "Forecast daily demand for a store/department for the next N days with the LightGBM model "
        "(uses known SNAP promotion days and retail events). Returns totals and an 80% interval.",
        _store_dept_schema(_HORIZON)),
    _fn("detect_stockout_risk",
        "Assess stockout risk: inventory position (on hand + inbound) vs forecast demand until the next "
        "scheduled delivery plus safety stock. Returns CRITICAL/WARNING/OK and a projected stockout date.",
        _store_dept_schema()),
    _fn("detect_demand_spike",
        "Detect a sudden demand spike in the last 7 days, or one forecast for the next 7 days, vs a 90-day baseline.",
        _store_dept_schema()),
    _fn("get_promo_calendar",
        "List upcoming SNAP promotion days and retail events (M5 calendar) for a store.",
        {"type": "object", "properties": {"store": {"type": "string", "enum": STORES},
                                          "days": {"type": "integer", "default": 28}},
         "required": ["store"]}),
    _fn("query_trends",
        "Query the external Google Trends demand signal for a department category (FOODS/HOBBIES/HOUSEHOLD).",
        {"type": "object", "properties": {"category": {"type": "string", "enum": DEPARTMENTS}},
         "required": ["category"]}),
    _fn("recommend_reorder",
        "Recommend an expedite order quantity with a supplier allocation (fastest vs best-value supplier). "
        "Does not create a purchase order.",
        _store_dept_schema(_HORIZON)),
    _fn("draft_purchase_order",
        "Create DRAFT purchase orders for the recommended reorder. Drafts need human approval in the UI.",
        _store_dept_schema(_HORIZON)),
    _fn("inventory_snapshot",
        "Return current inventory (on hand, inbound, safety stock, supplier) for optional store/department filters.",
        _store_dept_schema(required=())),
    _fn("run_risk_scan",
        "Scan all 15 store/department series for stockout risk and raise alerts for CRITICAL/WARNING ones.",
        {"type": "object", "properties": dict(_HORIZON)}),
]


def _forecast_payload(store: str, dept: str, horizon: int) -> dict:
    fc = forecast_demand(store, dept, horizon)
    records = [{"date": str(r.Date.date()), "yhat": round(float(r.yhat), 1),
                "lower": round(float(r.yhat_lower), 1), "upper": round(float(r.yhat_upper), 1)}
               for r in fc.itertuples()]
    summary = dict(store=store, dept=dept, horizon=horizon,
                   total_demand=round(float(fc["yhat"].sum()), 1),
                   total_lower=round(float(fc["yhat_lower"].sum()), 1),
                   total_upper=round(float(fc["yhat_upper"].sum()), 1),
                   avg_daily=round(float(fc["yhat"].mean()), 1),
                   first_day=records[0]["date"], last_day=records[-1]["date"],
                   interval=interval_method(store, dept))
    return {"summary": summary, "forecast": records}


def execute_tool(name: str, args: dict) -> dict:
    args = args or {}
    try:
        if name == "forecast_demand":
            return {"tool": name, **_forecast_payload(args["store"], args["dept"], int(args.get("horizon", 28)))}
        if name == "detect_stockout_risk":
            return {"tool": name, **inventory.stock_status(args["store"], args["dept"])}
        if name == "detect_demand_spike":
            return {"tool": name, **inventory.detect_demand_spike(args["store"], args["dept"])}
        if name == "get_promo_calendar":
            return {"tool": name, **inventory.promo_calendar(args["store"], int(args.get("days", 28)))}
        if name == "query_trends":
            return {"tool": name, **trends.query_trend(args["category"])}
        if name == "recommend_reorder":
            rec = inventory.reorder_recommendation(args["store"], args["dept"], int(args.get("horizon", 28)))
            return {"tool": name, "recommendation": rec}
        if name == "draft_purchase_order":
            return {"tool": name, "po": inventory.draft_po(args["store"], args["dept"], int(args.get("horizon", 28)))}
        if name == "inventory_snapshot":
            rows = inventory.get_inventory(args.get("store"), args.get("dept"))
            if not rows:
                inventory.seed_inventory_from_data()
                rows = inventory.get_inventory(args.get("store"), args.get("dept"))
            return {"tool": name, "rows": rows}
        if name == "run_risk_scan":
            result = inventory.run_risk_scan(int(args.get("horizon", 28)))
            return {"tool": name, "status": result.get("status"), "flagged": result.get("flagged", 0),
                    "items": result.get("items", []), "all_items": result.get("all_items", []),
                    "severity_counts": result.get("severity_counts", {}),
                    "items_evaluated": result.get("items_evaluated", 0)}
        return {"tool": name, "error": "unknown tool"}
    except Exception as e:
        return {"tool": name, "error": str(e)}
