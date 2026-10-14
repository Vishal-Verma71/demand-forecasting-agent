"""Tool definitions and a dispatcher used by the agent."""
from __future__ import annotations

import json
import re

from . import inventory, trends
from .config import STORES, DEPARTMENTS
from .model import forecast_demand, predict_lightgbm


def _norm_store_dept(query: str):
    q = query.upper()
    store = next((s for s in STORES if s in q), None)
    dept = next((d for d in DEPARTMENTS if d in q), None)
    return store, dept


OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "forecast_demand",
            "description": "Forecast demand for a store/department for the next N days using Prophet (primary) and LightGBM (comparison).",
            "parameters": {
                "type": "object",
                "properties": {
                    "store": {"type": "string", "enum": STORES},
                    "dept": {"type": "string", "enum": DEPARTMENTS},
                    "horizon": {"type": "integer", "default": 28},
                },
                "required": ["store", "dept"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_stockout_risk",
            "description": "Assess inventory risk (days of supply vs threshold) for a store/department.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store": {"type": "string", "enum": STORES},
                    "dept": {"type": "string", "enum": DEPARTMENTS},
                },
                "required": ["store", "dept"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_trends",
            "description": "Query external (Google Trends) demand signal for a category (FOODS/HOBBIES/HOUSEHOLD).",
            "parameters": {
                "type": "object",
                "properties": {"category": {"type": "string"}},
                "required": ["category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_reorder",
            "description": "Recommend a reorder quantity, supplier and create a draft PO.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store": {"type": "string", "enum": STORES},
                    "dept": {"type": "string", "enum": DEPARTMENTS},
                    "horizon": {"type": "integer", "default": 28},
                },
                "required": ["store", "dept"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inventory_snapshot",
            "description": "Return current inventory KPIs across (optionally filtered) stores/departments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store": {"type": "string"},
                    "dept": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_demand_spike",
            "description": "Detect sudden demand spikes for a store/department vs. trailing baseline.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store": {"type": "string", "enum": STORES},
                    "dept": {"type": "string", "enum": DEPARTMENTS},
                },
                "required": ["store", "dept"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_risk_scan",
            "description": "Scan all store/department combinations and flag stockout risks, raising alerts.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def execute_tool(name: str, args: dict) -> dict:
    args = args or {}
    try:
        if name == "forecast_demand":
            store = args["store"]; dept = args["dept"]; horizon = int(args.get("horizon", 28))
            fc = forecast_demand(store, dept, horizon)
            lgb_df = None
            try:
                lgb_df = predict_lightgbm(store, dept, horizon)
            except Exception:
                pass
            summary = dict(store=store, dept=dept, horizon=horizon,
                           total_demand=round(float(fc["yhat"].sum()), 1),
                           avg_daily=round(float(fc["yhat"].mean()), 1),
                           first_day=str(fc["Date"].iloc[0].date()),
                           last_day=str(fc["Date"].iloc[-1].date()))
            return {"tool": name, "forecast": fc.to_dict("records"),
                    "lightgbm": lgb_df.to_dict("records") if lgb_df is not None else None,
                    "summary": summary}
        if name == "detect_stockout_risk":
            return {"tool": name, **inventory.stock_status(args["store"], args["dept"])}
        if name == "query_trends":
            return {"tool": name, **trends.query_trend(args["category"])}
        if name == "recommend_reorder":
            store = args["store"]; dept = args["dept"]; horizon = int(args.get("horizon", 28))
            rec = inventory.reorder_recommendation(store, dept, horizon)
            po = inventory.approve_po(store, dept)
            return {"tool": name, "recommendation": rec, "po": po}
        if name == "inventory_snapshot":
            rows = inventory.get_inventory(args.get("store"), args.get("dept"))
            if not rows:
                inventory.seed_inventory_from_data()
                rows = inventory.get_inventory(args.get("store"), args.get("dept"))
            return {"tool": name, "rows": rows}
        if name == "detect_demand_spike":
            return {"tool": name, **inventory.detect_demand_spike(args["store"], args["dept"])}
        if name == "run_risk_scan":
            alerts = inventory.run_risk_scan()
            return {"tool": name, "flagged": len(alerts), "items": alerts}
        return {"tool": name, "error": "unknown tool"}
    except Exception as e:
        return {"tool": name, "error": str(e)}
