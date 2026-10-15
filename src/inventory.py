"""Inventory risk engine, reorder recommendations and supplier allocation.

All demand figures used for risk and ordering come from the LightGBM forecast, so the
ML model is the tool that drives the agent's decisions.  Units are store/department
aggregate units throughout.
"""
from __future__ import annotations

import math
import sqlite3

import numpy as np
import pandas as pd

from .config import STORES, DEPARTMENTS, FORECAST_HORIZON_DEFAULT, SERVICE_LEVEL_Z
from .data import (load_or_generate, future_calendar, SUPPLIERS, SUPPLIER_TERMS,
                   REVIEW_PERIOD_DAYS)
from .db import upsert_inventory, get_inventory, add_alert, add_po, get_po_by_key
from .model import forecast_demand as forecast_model, MAX_FORECAST_HORIZON

# Window of recent actual sales used for the demand-variability estimate.
DEMAND_HISTORY_WINDOW_DAYS = 28


def _history(store: str, dept: str) -> pd.DataFrame:
    df = load_or_generate()
    return df[(df["Store"] == store) & (df["Dept"] == dept)].sort_values("Date")


def _demand_sigma(sales: pd.Series) -> float:
    """Daily demand variability with the weekly pattern removed.

    Raw daily std mostly measures the weekend peak, which the forecast already
    predicts.  The week-over-week difference divided by sqrt(2) estimates the
    unpredictable part instead.
    """
    diff = sales.diff(7).dropna().tail(DEMAND_HISTORY_WINDOW_DAYS * 2)
    return float(diff.std(ddof=1) / math.sqrt(2)) if len(diff) > 1 else 0.0


def safety_stock_units(sigma_daily: float, lead_time_days: int, z: float = SERVICE_LEVEL_Z) -> float:
    """Classic safety stock: z * sigma_daily * sqrt(lead time)."""
    return z * sigma_daily * math.sqrt(max(lead_time_days, 0))


def seed_inventory_from_data(df=None) -> None:
    if df is None:
        df = load_or_generate()
    rows = []
    for (store, dept), sub in df.sort_values("Date").groupby(["Store", "Dept"]):
        last = sub.iloc[-1]
        lead = int(last["LeadTime"])
        days_to_review = int(last.get("DaysToReview", REVIEW_PERIOD_DAYS))
        rows.append(dict(store=store, dept=dept, sku=f"{store}-{dept}",
                         on_hand=float(last["OnHand"]),
                         on_order=float(last.get("OnOrder", 0.0)),
                         days_to_review=days_to_review,
                         safety_stock=round(safety_stock_units(_demand_sigma(sub["Sales"]),
                                                               lead + days_to_review), 1),
                         lead_time_days=lead, supplier=last["Supplier"]))
    upsert_inventory(rows)


def stock_status(store: str, dept: str, horizon: int = 28) -> dict:
    """Return an auditable stockout-risk assessment for one store/department.

    The store orders on a weekly review cycle, so stock on hand plus inbound must last
    until the order placed at the *next* review arrives (the protection interval):

    inventory_position = on_hand + on_order (units already ordered from suppliers)
    protection_days    = days_to_next_review + lead_time
    protection_demand  = LightGBM forecast units over the protection interval
    reorder_point      = protection_demand + safety_stock

      CRITICAL: inventory_position < protection_demand  (stockout before the next scheduled delivery)
      WARNING:  inventory_position < reorder_point      (safety buffer will be consumed)
      OK:       inventory_position >= reorder_point
      UNKNOWN:  inputs missing or invalid
    """
    inv_rows = get_inventory(store, dept)
    inv = inv_rows[0] if inv_rows else None
    on_hand = float(inv["on_hand"]) if inv is not None else None
    on_order = float(inv.get("on_order") or 0.0) if inv is not None else 0.0
    lead = int(inv["lead_time_days"]) if inv is not None else 7
    supplier = inv["supplier"] if inv is not None else "Unknown"
    safety = float(inv["safety_stock"]) if inv is not None else 0.0
    days_to_review = int(inv.get("days_to_review") or REVIEW_PERIOD_DAYS) if inv is not None else REVIEW_PERIOD_DAYS
    protection_days = lead + days_to_review

    sub = _history(store, dept)
    sales = sub["Sales"].tail(DEMAND_HISTORY_WINDOW_DAYS)
    historical_avg_daily = float(sales.mean()) if len(sales) else 0.0

    input_errors = []
    forecast_error = None
    daily_forecast = np.array([])
    fc_len = min(MAX_FORECAST_HORIZON, max(horizon, protection_days, 1))
    try:
        daily_forecast = forecast_model(store, dept, fc_len)["yhat"].to_numpy(dtype=float)
    except Exception as exc:
        forecast_error = str(exc)

    if len(daily_forecast) >= max(protection_days, 1):
        lead_time_demand = float(daily_forecast[:lead].sum())
        protection_demand = float(daily_forecast[:protection_days].sum())
        daily_demand = protection_demand / protection_days
        forecast_horizon_demand = float(daily_forecast[:max(horizon, 0)].sum())
        demand_source = "LightGBM forecast"
    else:
        daily_demand = historical_avg_daily
        lead_time_demand = daily_demand * lead
        protection_demand = daily_demand * protection_days
        forecast_horizon_demand = 0.0
        demand_source = "trailing 28-day actual sales (forecast unavailable)"
    avg_forecast_daily = forecast_horizon_demand / horizon if horizon > 0 else 0.0

    if horizon <= 0:
        input_errors.append("forecast horizon must be positive")
    if lead < 0:
        input_errors.append("lead time is negative")
    if safety < 0:
        input_errors.append("safety stock is negative")
    if on_hand is None:
        input_errors.append("on-hand inventory is unavailable")
    elif on_hand < 0:
        input_errors.append("on-hand inventory is negative")
    if daily_demand <= 0:
        input_errors.append("daily demand is zero or unavailable")
    inputs_valid = not input_errors

    position = (on_hand or 0.0) + on_order
    reorder_point = protection_demand + safety
    days_of_supply = position / daily_demand if daily_demand > 0 else float("inf")

    # Day on which cumulative forecast demand first exceeds the inventory position.
    stockout_date = None
    if inputs_valid and len(daily_forecast):
        cum = np.cumsum(daily_forecast)
        hit = np.nonzero(cum > position)[0]
        if len(hit):
            first_day = sub["Date"].max() + pd.Timedelta(days=int(hit[0]) + 1)
            stockout_date = str(first_day.date())

    if not inputs_valid:
        risk_level, reason = "UNKNOWN", "insufficient data: " + "; ".join(input_errors)
    elif position < protection_demand:
        risk_level = "CRITICAL"
        reason = (f"stock plus inbound runs out before the next scheduled delivery "
                  f"(review in {days_to_review}d + {lead}d lead time)")
    elif position < reorder_point:
        risk_level = "WARNING"
        reason = "stock plus inbound covers demand until the next delivery but not the safety buffer"
    else:
        risk_level = "OK"
        reason = "stock plus inbound covers forecast demand until the next delivery plus safety stock"
    if forecast_error is not None:
        reason = f"{reason}; forecast failure: {forecast_error}"

    return dict(
        store=store, dept=dept,
        on_hand=round(on_hand or 0.0, 2),
        on_order=round(on_order, 2),
        inventory_position=round(position, 2),
        safety_stock=round(safety, 2),
        avg_daily_demand=round(daily_demand, 4),
        demand_source=demand_source,
        avg_forecast_daily_demand=round(avg_forecast_daily, 2),
        historical_daily_average_demand=round(historical_avg_daily, 2),
        historical_demand_window_days=DEMAND_HISTORY_WINDOW_DAYS,
        forecast_horizon=horizon,
        forecast_horizon_demand=round(forecast_horizon_demand, 2),
        days_of_supply=round(min(days_of_supply, 999), 2),
        projected_stockout_date=stockout_date,
        lead_time_days=lead,
        days_to_next_review=days_to_review,
        protection_days=protection_days,
        lead_time_demand=round(lead_time_demand, 2),
        protection_demand=round(protection_demand, 2),
        reorder_point=round(reorder_point, 2),
        supplier=supplier, risk_level=risk_level,
        classification_reason=reason,
        inputs_valid=inputs_valid,
        input_errors=input_errors,
        forecast_error=forecast_error,
        inventory_description="simulated store inventory ledger (M5 has no stock data)",
    )


def allocate_suppliers(dept: str, qty: float, urgent_qty: float, deadline_days: int | None = None) -> list[dict]:
    """Split an order between the department's suppliers.

    The urgent part (what is short before even the fastest supplier can deliver) goes
    to the fastest supplier.  The rest goes to the best-value supplier, scored as unit
    cost divided by on-time reliability, provided it can deliver within
    ``deadline_days``; otherwise it also goes to the fastest supplier.
    Returns one line per supplier used.
    """
    choices = SUPPLIERS.get(dept, [("UnknownSupplier", 7)])
    fastest = min(choices, key=lambda c: (c[1], c[0]))

    def value_score(choice):
        terms = SUPPLIER_TERMS.get(choice[0], {"unit_cost": 1.0, "reliability": 0.9})
        return terms["unit_cost"] / terms["reliability"]

    in_time = [c for c in choices if deadline_days is None or c[1] <= deadline_days] or [fastest]
    best_value = min(in_time, key=value_score)
    urgent = min(max(urgent_qty, 0.0), qty)
    lines = {}
    if urgent > 0:
        lines[fastest] = [urgent, "urgent: shortfall before any delivery can land (fastest lead time)"]
    rest = qty - urgent
    if rest > 0:
        if best_value in lines:
            lines[best_value][0] += rest
        else:
            lines[best_value] = [rest, "cycle stock: best cost / reliability score"]
    out = []
    for (name, lead), (units, why) in lines.items():
        terms = SUPPLIER_TERMS.get(name, {"unit_cost": 1.0, "reliability": 0.9})
        # Round up so the order fully closes the gap to the reorder point.
        out.append(dict(supplier=name, qty=int(math.ceil(units - 1e-9)), lead_time_days=lead,
                        unit_cost=terms["unit_cost"], reliability=terms["reliability"], reason=why))
    return [line for line in out if line["qty"] > 0]


def reorder_recommendation(store: str, dept: str, horizon: int = FORECAST_HORIZON_DEFAULT) -> dict:
    """Expedite recommendation for series the weekly static order will not protect.

    Only CRITICAL/WARNING series get an order.  The quantity tops the inventory position
    back up to the reorder point (forecast demand until the next scheduled delivery plus
    safety stock), so the following routine review takes over again.
    """
    st = stock_status(store, dept, horizon)
    coverage_days = st["protection_days"]
    base = dict(store=store, dept=dept, supplier=st["supplier"], lead_time_days=st["lead_time_days"],
                risk_level=st["risk_level"], coverage_days=coverage_days,
                inventory_position=st["inventory_position"], reorder_point=st["reorder_point"])

    if st["risk_level"] not in ("CRITICAL", "WARNING"):
        why = ("inventory position is above the reorder point; no order needed until the next review"
               if st["risk_level"] == "OK" else st["classification_reason"])
        return {**base, "recommended_qty": 0, "allocations": [], "estimated_cost_index": 0.0,
                "promo_days_in_cover": 0, "events_in_cover": [], "rationale": why}

    fc = forecast_model(store, dept, min(coverage_days, MAX_FORECAST_HORIZON))
    coverage_demand = float(fc["yhat"].sum())
    qty = max(0.0, coverage_demand + st["safety_stock"] - st["inventory_position"])
    fastest_lead = min(lead for _, lead in SUPPLIERS.get(dept, [("UnknownSupplier", 7)]))
    before_fastest = float(fc["yhat"].iloc[:fastest_lead].sum())
    urgent = max(0.0, before_fastest + st["safety_stock"] - st["inventory_position"])
    allocations = allocate_suppliers(dept, qty, urgent, deadline_days=coverage_days)

    cal = future_calendar(store, fc["Date"].iloc[0], len(fc))
    promo_days = int(cal["Promo"].sum())
    events = [e for e in cal["Event"].tolist() if e]
    rationale = (f"Cover {coverage_days}d (next review in {st['days_to_next_review']}d + {st['lead_time_days']}d lead) of forecast demand "
                 f"{round(coverage_demand)} + safety {round(st['safety_stock'])} − position {round(st['inventory_position'])} "
                 f"(on hand {round(st['on_hand'])} + inbound {round(st['on_order'])}).")
    if promo_days or events:
        rationale += f" Window includes {promo_days} SNAP promo day(s)" + (f" and events: {', '.join(events)}" if events else "") + \
                     " — already reflected in the forecast."
    return {**base,
            "supplier": allocations[0]["supplier"] if allocations else st["supplier"],
            "recommended_qty": int(sum(a["qty"] for a in allocations)),
            "allocations": allocations,
            "estimated_cost_index": round(sum(a["qty"] * a["unit_cost"] for a in allocations), 1),
            "coverage_demand": round(coverage_demand, 1),
            "promo_days_in_cover": promo_days, "events_in_cover": events,
            "rationale": rationale}


def detect_demand_spike(store: str, dept: str, window: int = 90) -> dict:
    """Flag a spike when the last 7 days, or the next 7 forecast days, exceed mean + 2 std."""
    s = _history(store, dept)["Sales"].astype(float)
    recent = s.tail(7).mean()
    base = s.tail(window + 7).head(window)
    mu, sd = base.mean(), base.std()
    threshold = mu + 2 * sd
    try:
        upcoming = float(forecast_model(store, dept, 7)["yhat"].mean())
    except Exception:
        upcoming = float("nan")
    return dict(store=store, dept=dept, recent_7d_avg=round(float(recent), 2),
                forecast_next_7d_avg=round(upcoming, 2),
                baseline_mean=round(float(mu), 2), baseline_std=round(float(sd), 2),
                threshold=round(float(threshold), 2),
                spike=bool(recent > threshold),
                upcoming_spike=bool(upcoming > threshold),
                change_pct=round(float((recent - mu) / max(mu, 1) * 100), 1),
                upcoming_change_pct=round(float((upcoming - mu) / max(mu, 1) * 100), 1))


def promo_calendar(store: str, days: int = 28) -> dict:
    """Upcoming SNAP promotion days and retail events from the M5 calendar."""
    df = load_or_generate()
    start = df["Date"].max() + pd.Timedelta(days=1)
    cal = future_calendar(store, start, min(days, MAX_FORECAST_HORIZON))
    return dict(store=store, start=str(start.date()), days=len(cal),
                snap_promo_days=[str(d.date()) for d in cal.loc[cal["Promo"] == 1, "Date"]],
                events=[{"date": str(r.Date.date()), "event": r.Event} for r in cal.itertuples() if r.Event],
                source="M5 calendar.csv (SNAP + events, known in advance)")


def run_risk_scan(horizon: int = 28):
    alerts = []
    evaluated_results = []
    errors = []
    severity_counts = {"CRITICAL": 0, "WARNING": 0, "OK": 0, "UNKNOWN": 0}
    for store in STORES:
        for dept in DEPARTMENTS:
            try:
                st = stock_status(store, dept, horizon)
            except Exception as exc:
                st = {"store": store, "dept": dept, "on_hand": 0, "on_order": 0, "inventory_position": 0,
                      "protection_demand": 0,
                      "safety_stock": 0, "avg_daily_demand": 0, "lead_time_days": 0, "lead_time_demand": 0,
                      "reorder_point": 0, "days_of_supply": 0, "supplier": "Unknown", "risk_level": "UNKNOWN",
                      "classification_reason": f"scan failure: {exc}", "forecast_error": str(exc)}
                errors.append({"store": store, "dept": dept, "error": str(exc)})
            severity_counts[st["risk_level"]] += 1
            evaluated_results.append(st)
            if st["risk_level"] in ("CRITICAL", "WARNING"):
                add_alert(store, dept, st["risk_level"],
                          f"{store}/{dept}: {st['classification_reason']} (position {st['inventory_position']}, "
                          f"reorder point {st['reorder_point']}, demand to next delivery {st['protection_demand']})")
                alerts.append(st)
    alerts.sort(key=lambda i: (i["risk_level"] != "CRITICAL", i["days_of_supply"]))
    return {
        "status": "FAILED" if errors else "COMPLETED",
        "items": alerts,
        "all_items": evaluated_results,
        "flagged": len(alerts),
        "items_evaluated": len(evaluated_results),
        "severity_counts": severity_counts,
        "errors": errors,
        "inventory_assumption": {
            "on_hand": "simulated weekly order-up-to inventory ledger replayed over actual M5 sales (M5 has no stock data)",
            "on_order": "orders placed by that simulated policy and not yet received",
            "safety_stock": f"z={SERVICE_LEVEL_Z} x de-seasonalised daily demand std x sqrt(lead time)",
            "lead_time": "supplier business assumption, not a Kaggle field",
            "demand": "LightGBM forecast over the protection interval (days to next review + lead time)",
        },
        "risk_policy": {
            "UNKNOWN": "inputs required for the calculation are unavailable or invalid",
            "CRITICAL": "stock plus inbound runs out before the next scheduled delivery",
            "WARNING": "stock plus inbound covers demand to the next delivery but not the safety buffer",
            "OK": "stock plus inbound covers demand to the next delivery plus safety stock",
        },
    }


def draft_po(store, dept, horizon: int = FORECAST_HORIZON_DEFAULT) -> dict:
    """Create DRAFT purchase orders (one per supplier line) for a human to approve.

    Idempotent: repeating the same recommendation returns the existing drafts.
    """
    rec = reorder_recommendation(store, dept, horizon)
    if rec["recommended_qty"] <= 0:
        return {"po_id": None, "po_ids": [], **rec, "status": "NO_ORDER_NEEDED",
                "planning_horizon_days": horizon}
    ids, created = [], False
    for line in rec["allocations"]:
        key = f"{store}:{dept}:{line['supplier']}:{line['qty']}:{line['lead_time_days']}"
        existing = get_po_by_key(key)
        if existing is None:
            try:
                ids.append(add_po(store, dept, line["supplier"], line["qty"], line["lead_time_days"], key,
                                  unit_cost=line["unit_cost"], reason=line["reason"]))
                created = True
                continue
            except sqlite3.IntegrityError:
                existing = get_po_by_key(key)
        ids.append(existing["id"])
    status = "DRAFT created — awaiting approval" if created else "DRAFT already existed (idempotent, no duplicate)"
    return {"po_id": ids[0], "po_ids": ids, **rec, "status": status, "planning_horizon_days": horizon}
