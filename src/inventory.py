"""Inventory risk engine + reorder recommendations."""
from __future__ import annotations

import sqlite3
from functools import lru_cache
from datetime import timedelta

import numpy as np
import pandas as pd

from .config import SAFETY_STOCK_FACTOR, STOCKOUT_THRESHOLD_DAYS
from .data import load_or_generate
from .db import upsert_inventory, get_inventory, add_alert, add_po, get_pos, get_alerts
from .model import forecast_demand as forecast_model


def seed_inventory_from_data(df=None) -> None:
    if df is None:
        df = load_or_generate()
    # last row per series = current on-hand
    latest = df.sort_values("Date").groupby(["Store", "Dept"]).tail(1)
    rows = []
    for _, r in latest.iterrows():
        rows.append(dict(store=r["Store"], dept=r["Dept"], sku=f"{r['Store']}-{r['Dept']}",
                         on_hand=float(r["OnHand"]), safety_stock=float(r["OnHand"]) * 0.3,
                         lead_time_days=int(r["LeadTime"]), supplier=r["Supplier"]))
    upsert_inventory(rows)
    stock_status.cache_clear()


@lru_cache(maxsize=512)
def stock_status(store: str, dept: str, horizon: int = 28) -> dict:
    inv_rows = get_inventory(store, dept)
    inv = inv_rows[0] if inv_rows else None
    on_hand = float(inv["on_hand"]) if inv else 0.0
    lead = int(inv["lead_time_days"]) if inv else 7
    supplier = inv["supplier"] if inv else "Unknown"
    safety = float(inv["safety_stock"]) if inv else on_hand * 0.3

    # forecast demand over horizon
    try:
        fc = forecast_model(store, dept, horizon)
        total_demand = float(fc["yhat"].sum())
        avg_daily = total_demand / horizon
    except Exception:
        total_demand = 0.0
        avg_daily = 0.0

    days_of_supply = (on_hand / avg_daily) if avg_daily > 0 else float("inf")
    risk_level = "OK"
    if days_of_supply < STOCKOUT_THRESHOLD_DAYS:
        risk_level = "CRITICAL"
    elif days_of_supply < STOCKOUT_THRESHOLD_DAYS * 2:
        risk_level = "WARNING"
    if on_hand < safety:
        risk_level = "CRITICAL"

    return dict(store=store, dept=dept, on_hand=round(on_hand, 2), safety_stock=round(safety, 2),
                avg_daily_demand=round(avg_daily, 2), days_of_supply=round(min(days_of_supply, 999), 2),
                forecast_horizon_demand=round(total_demand, 2), lead_time_days=lead,
                supplier=supplier, risk_level=risk_level)


def reorder_recommendation(store: str, dept: str, horizon: int = 28) -> dict:
    st = stock_status(store, dept, horizon)
    # EOQ-ish: order enough for horizon + safety + lead-time demand
    lead_demand = st["avg_daily_demand"] * st["lead_time_days"]
    target = st["forecast_horizon_demand"] + (st["safety_stock"] * SAFETY_STOCK_FACTOR) + lead_demand
    qty = max(0.0, target - st["on_hand"])
    # round
    qty = round(qty, 0)
    return dict(store=store, dept=dept, recommended_qty=int(qty),
                supplier=st["supplier"], lead_time_days=st["lead_time_days"],
                rationale=f"Cover {horizon}d demand ({st['forecast_horizon_demand']}) + safety {st['safety_stock']} + lead-time demand {round(lead_demand)}; on hand {st['on_hand']}.",
                risk_level=st["risk_level"])


def detect_demand_spike(store: str, dept: str, window: int = 90) -> dict:
    from .data import load_or_generate
    from .features import build_series
    df = load_or_generate()
    s = build_series(df, store, dept)
    recent = s["Sales"].tail(7).mean()
    base = s["Sales"].tail(window + 7).head(window)
    mu, sd = base.mean(), base.std()
    threshold = mu + 2 * sd
    spike = bool(recent > threshold)
    pct = (recent - mu) / max(mu, 1) * 100
    return dict(store=store, dept=dept, recent_7d_avg=round(float(recent), 2),
                baseline_mean=round(float(mu), 2), baseline_std=round(float(sd), 2),
                threshold=round(float(threshold), 2), spike=spike,
                change_pct=round(float(pct), 1))


def run_risk_scan() -> list[dict]:
    alerts = []
    for store in ["CA_1", "CA_2", "TX_1", "TX_2", "WI_1"]:
        for dept in ["FOODS", "HOBBIES", "HOUSEHOLD"]:
            st = stock_status(store, dept)
            if st["risk_level"] in ("CRITICAL", "WARNING"):
                add_alert(store, dept, st["risk_level"],
                          f"{store}/{dept}: {st['days_of_supply']} days of supply (on hand {st['on_hand']}, daily demand {st['avg_daily_demand']})")
                alerts.append(st)
    return alerts


def approve_po(store, dept, horizon: int = 28) -> dict:
    rec = reorder_recommendation(store, dept, horizon)
    po_id = add_po(store, dept, rec["supplier"], rec["recommended_qty"], rec["lead_time_days"])
    return {"po_id": po_id, **rec, "status": "DRAFT created", "planning_horizon_days": horizon}


def clear_stock_cache():
    try:
        stock_status.cache_clear()
    except AttributeError:
        pass
