"""FastAPI backend exposing the existing Python modules as JSON for the Next.js UI."""
from __future__ import annotations

import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import db, inventory, report
from src.agent import run_agent
from src.config import STORES, DEPARTMENTS
from src.data import load_or_generate, data_diagnostics
from src.features import build_series
from src.llm import backend_name, is_available
from src.model import (models_ready, train_all, forecast_demand, get_metrics, clear_model_cache,
                       interval_method, MAX_FORECAST_HORIZON, MODEL_NAME)
from src.eval import load_evaluation, run_backtest
from src.trends import query_trend


@asynccontextmanager
async def lifespan(_app):
    get_df()
    threading.Thread(target=_prewarm_model_cache, daemon=True).start()
    yield


app = FastAPI(title="Demand Forecasting Agent API", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",  # any local dev port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_df_cache = None


def _check(store: str | None = None, dept: str | None = None, horizon: int | None = None) -> None:
    """Reject bad inputs with 400 instead of letting them surface as 500s."""
    if store is not None and store not in STORES:
        raise HTTPException(400, f"Unknown store '{store}'. Expected one of {STORES}")
    if dept is not None and dept not in DEPARTMENTS:
        raise HTTPException(400, f"Unknown department '{dept}'. Expected one of {DEPARTMENTS}")
    if horizon is not None and not 1 <= horizon <= MAX_FORECAST_HORIZON:
        raise HTTPException(400, f"horizon must be between 1 and {MAX_FORECAST_HORIZON}")


def get_df():
    global _df_cache
    if _df_cache is None:
        db.init_db()
        _df_cache = load_or_generate()
        # Re-seed only when the stored snapshot is missing or came from different data, so
        # inbound stock from approved POs survives a restart.
        latest = _df_cache.sort_values("Date").groupby(["Store", "Dept"]).tail(1)
        expected = {(r.Store, r.Dept): round(float(r.OnHand), 1) for r in latest.itertuples()}
        stored = {(r["store"], r["dept"]): round(float(r["on_hand"]), 1) for r in inventory.get_inventory()}
        if stored != expected:
            inventory.seed_inventory_from_data(_df_cache)
    return _df_cache


def _prewarm_model_cache():
    try:
        # Build the full interface-horizon forecast once per series so arbitrary slider values
        # reuse cached recursive predictions instead of re-running the expensive forecast path.
        for s in STORES:
            for d in DEPARTMENTS:
                forecast_demand(s, d, MAX_FORECAST_HORIZON)
    except Exception:
        pass


def _jsonable(df):
    out = df.copy()
    for c in out.columns:
        if str(out[c].dtype).startswith("datetime"):
            out[c] = out[c].astype(str)
    return out.to_dict("records")


@app.get("/health")
def health():
    return {"status": "ok", "stores": STORES, "departments": DEPARTMENTS,
            "models_ready": models_ready(), "model": MODEL_NAME, "llm_backend": backend_name(),
            "llm_available": is_available()}


@app.get("/kpi")
def kpi(horizon: int = 28):
    _check(horizon=horizon)
    df = get_df()
    inv = inventory.get_inventory()
    at_risk = []
    for s in STORES:
        for d in DEPARTMENTS:
            stt = inventory.stock_status(s, d, horizon)
            if stt["risk_level"] in ("CRITICAL", "WARNING"):
                at_risk.append(stt)
    pos = db.get_pos(10000)
    return {
        "rows": int(len(df)),
        "stores": int(df["Store"].nunique()),
        "departments": int(df["Dept"].nunique()),
        "on_hand_units": float(sum(r["on_hand"] for r in inv)) if inv else 0.0,
        "inbound_units": float(sum(r.get("on_order") or 0 for r in inv)) if inv else 0.0,
        "at_risk_combinations": len(at_risk),
        "critical_combinations": len([a for a in at_risk if a["risk_level"] == "CRITICAL"]),
        "draft_pos": len([p for p in pos if p.get("status") == "DRAFT"]),
        "horizon_default": 28,
    }


@app.get("/dashboard/risk")
def dashboard_risk(horizon: int = 28):
    _check(horizon=horizon)
    items = []
    for s in STORES:
        for d in DEPARTMENTS:
            stt = inventory.stock_status(s, d, horizon)
            items.append({"store": s, "dept": d, "days_of_supply": stt["days_of_supply"],
                          "risk_level": stt["risk_level"], "on_hand": stt["on_hand"],
                          "on_order": stt["on_order"], "reorder_point": stt["reorder_point"],
                          "projected_stockout_date": stt["projected_stockout_date"],
                          "protection_days": stt["protection_days"],
                          "avg_daily_demand": round(stt["avg_daily_demand"], 1)})
    return {"items": items, "at_risk": [i for i in items if i["risk_level"] in ("CRITICAL", "WARNING")]}


@app.get("/alerts")
def alerts(limit: int = 50):
    return {"alerts": db.get_alerts(limit)}


@app.get("/diagnostics")
def diagnostics():
    return data_diagnostics()


@app.post("/risk-scan")
def risk_scan(horizon: int = 28):
    _check(horizon=horizon)
    a = inventory.run_risk_scan(horizon)
    return {
        "status": a["status"],
        "flagged": a["flagged"],
        "items": a["items"],
        "all_items": a.get("all_items", []),
        "items_evaluated": a["items_evaluated"],
        "severity_counts": a["severity_counts"],
        "errors": a["errors"],
        "risk_policy": a["risk_policy"],
        "inventory_assumption": a["inventory_assumption"],
    }


@app.get("/inventory")
def inventory_view(store: str | None = None, dept: str | None = None):
    _check(store, dept)
    rows = inventory.get_inventory(store, dept)
    if not rows:
        inventory.seed_inventory_from_data(get_df())
        rows = inventory.get_inventory(store, dept)
    return {"rows": rows}


@app.get("/series")
def series(store: str, dept: str, limit: int = 90):
    _check(store, dept)
    df = get_df()
    s = build_series(df, store, dept).tail(limit)[["Date", "Sales"]]
    return {"records": _jsonable(s)}


@app.get("/forecast")
def forecast(store: str, dept: str, horizon: int = 28):
    _check(store, dept, horizon)
    fc = forecast_demand(store, dept, horizon)
    summary = {"store": store, "dept": dept, "horizon": horizon,
               "total_demand": round(float(fc["yhat"].sum()), 1),
               "avg_daily": round(float(fc["yhat"].mean()), 1),
               "first_day": str(fc["Date"].iloc[0].date()),
               "last_day": str(fc["Date"].iloc[-1].date()),
               "interval": interval_method(store, dept)}
    # Show the honest multi-step backtest score for this series, not the one-step holdout.
    metrics = {}
    evaluation = load_evaluation()
    if evaluation:
        series_metric = next((m for m in evaluation.get("per_series_primary", [])
                              if m["series"] == f"{store}/{dept}"), None)
        if series_metric:
            metrics = {"source": f"{evaluation['forecast_horizon']}-day recursive backtest, this series",
                       "MAE": series_metric["model"]["MAE"], "MAPE": series_metric["model"]["MAPE_nonzero"],
                       "WAPE": series_metric["model"]["WAPE"],
                       "seasonal_naive_MAE": series_metric["seasonal_naive"]["MAE"]}
    if not metrics:
        metrics = {"source": "one-step holdout", **get_metrics()}
    return {"summary": summary, "records": _jsonable(fc), "metrics": metrics}


@app.get("/stock-risk")
def stock_risk(store: str, dept: str):
    _check(store, dept)
    return {"status": inventory.stock_status(store, dept)}


@app.get("/spike")
def spike(store: str, dept: str):
    _check(store, dept)
    return {"status": inventory.detect_demand_spike(store, dept)}


@app.get("/trend")
def trend(category: str):
    _check(dept=category.upper())
    return {"trend": query_trend(category)}


@app.get("/reorder")
def reorder(store: str, dept: str, horizon: int = 28):
    _check(store, dept, horizon)
    return {"recommendation": inventory.reorder_recommendation(store, dept, horizon)}


@app.post("/reorder")
def reorder_create(payload: dict = Body(...)):
    store = payload.get("store")
    dept = payload.get("dept")
    try:
        horizon = int(payload.get("horizon", 28) or 28)
    except (TypeError, ValueError):
        raise HTTPException(400, "horizon must be an integer")
    if not store or not dept:
        raise HTTPException(400, "store and dept are required")
    _check(store, dept, horizon)
    return {"result": inventory.draft_po(store, dept, horizon)}


@app.get("/pos")
def pos(limit: int = 100):
    return {"pos": db.get_pos(limit)}


@app.get("/promotions")
def promotions(store: str, days: int = 28):
    _check(store, horizon=days)
    return inventory.promo_calendar(store, days)


def _decide_po(po_id: int, status: str):
    try:
        return {"po": db.set_po_status(po_id, status)}
    except KeyError as exc:
        raise HTTPException(404, exc.args[0])
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.post("/pos/{po_id}/approve")
def approve_po(po_id: int):
    return _decide_po(po_id, "APPROVED")


@app.post("/pos/{po_id}/reject")
def reject_po(po_id: int):
    return _decide_po(po_id, "REJECTED")


@app.post("/agent")
def agent_endpoint(payload: dict = Body(...)):
    q = str(payload.get("query", payload.get("q", "")) or "").strip()
    if not q:
        return JSONResponse(status_code=400, content={"answer": "Please enter a question.", "steps": [],
                                                      "backend": "offline-router"})
    try:
        return run_agent(q, payload.get("store"), payload.get("dept"))
    except (ValueError, KeyError) as exc:
        return JSONResponse(status_code=400, content={"answer": str(exc), "steps": [], "backend": "offline-router"})


@app.get("/report")
def report_md():
    return {"markdown": report.build_report()}


@app.get("/evaluation")
def evaluation():
    result = load_evaluation()
    if result is None:
        return {"status": "NOT_RUN", "metrics": None}
    return {"status": "OK", "metrics": result}


@app.post("/evaluation/run")
def evaluation_run():
    result = run_backtest(horizon=28, windows=3)
    clear_model_cache()  # prediction intervals are derived from the new backtest residuals
    return {"status": "OK", "metrics": result}


@app.post("/train")
def train():
    df = get_df()
    lb = train_all(df)
    clear_model_cache()
    return {"metrics": lb["metrics"], "models_ready": models_ready()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
