"""FastAPI backend exposing the existing Python modules as JSON for the Next.js UI."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import FastAPI, Query, Body
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import db, inventory, report
from src.agent import run_agent
from src.config import STORES, DEPARTMENTS, MODEL_DIR
from src.data import load_or_generate
from src.features import build_series
from src.llm import backend_name, is_available
from src.model import models_ready, train_all, forecast_demand, get_metrics
from src.trends import query_trend

app = FastAPI(title="Demand Forecasting Agent API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_df_cache = None


def get_df():
    global _df_cache
    if _df_cache is None:
        db.init_db()
        _df_cache = load_or_generate()
        inventory.seed_inventory_from_data(_df_cache)
    return _df_cache


@app.on_event("startup")
def startup():
    get_df()


def _jsonable(df):
    out = df.copy()
    for c in out.columns:
        if str(out[c].dtype).startswith("datetime"):
            out[c] = out[c].astype(str)
    return out.to_dict("records")


@app.get("/health")
def health():
    return {"status": "ok", "stores": STORES, "departments": DEPARTMENTS,
            "models_ready": models_ready(), "llm_backend": backend_name(),
            "llm_available": is_available()}


@app.get("/kpi")
def kpi():
    df = get_df()
    inv = inventory.get_inventory()
    at_risk = []
    for s in STORES:
        for d in DEPARTMENTS:
            stt = inventory.stock_status(s, d)
            if stt["risk_level"] in ("CRITICAL", "WARNING"):
                at_risk.append(stt)
    pos = db.get_pos(10000)
    return {
        "rows": int(len(df)),
        "stores": int(df["Store"].nunique()),
        "departments": int(df["Dept"].nunique()),
        "on_hand_units": float(sum(r["on_hand"] for r in inv)) if inv else 0.0,
        "at_risk_skus": len(at_risk),
        "draft_pos": len(pos),
        "horizon_default": 28,
    }


@app.get("/dashboard/risk")
def dashboard_risk():
    items = []
    for s in STORES:
        for d in DEPARTMENTS:
            stt = inventory.stock_status(s, d)
            items.append({"store": s, "dept": d, "days_of_supply": stt["days_of_supply"],
                          "risk_level": stt["risk_level"], "on_hand": stt["on_hand"],
                          "avg_daily_demand": stt["avg_daily_demand"]})
    return {"items": items, "at_risk": [i for i in items if i["risk_level"] in ("CRITICAL", "WARNING")]}


@app.get("/alerts")
def alerts(limit: int = 50):
    return {"alerts": db.get_alerts(limit)}


@app.post("/risk-scan")
def risk_scan():
    a = inventory.run_risk_scan()
    return {"flagged": len(a), "items": a}


@app.get("/inventory")
def inventory_view(store: str | None = None, dept: str | None = None):
    rows = inventory.get_inventory(store, dept)
    if not rows:
        inventory.seed_inventory_from_data(get_df())
        rows = inventory.get_inventory(store, dept)
    return {"rows": rows}


@app.get("/series")
def series(store: str, dept: str, limit: int = 90):
    df = get_df()
    s = build_series(df, store, dept).tail(limit)[["Date", "Sales"]]
    return {"records": _jsonable(s)}


@app.get("/forecast")
def forecast(store: str, dept: str, horizon: int = 28):
    fc = forecast_demand(store, dept, horizon)
    summary = {"store": store, "dept": dept, "horizon": horizon,
               "total_demand": round(float(fc["yhat"].sum()), 1),
               "avg_daily": round(float(fc["yhat"].mean()), 1),
               "first_day": str(fc["Date"].iloc[0].date()),
               "last_day": str(fc["Date"].iloc[-1].date())}
    metrics = get_metrics()
    return {"summary": summary, "records": _jsonable(fc), "metrics": metrics}


@app.get("/stock-risk")
def stock_risk(store: str, dept: str):
    return {"status": inventory.stock_status(store, dept)}


@app.get("/spike")
def spike(store: str, dept: str):
    return {"status": inventory.detect_demand_spike(store, dept)}


@app.get("/trend")
def trend(category: str):
    return {"trend": query_trend(category)}


@app.get("/reorder")
def reorder(store: str, dept: str, horizon: int = 28):
    return {"recommendation": inventory.reorder_recommendation(store, dept, horizon)}


@app.post("/reorder")
def reorder_create(payload: dict = Body(...)):
    store = payload.get("store")
    dept = payload.get("dept")
    rec = inventory.approve_po(store, dept)
    return {"result": rec}


@app.get("/pos")
def pos(limit: int = 100):
    return {"pos": db.get_pos(limit)}


@app.post("/agent")
def agent_endpoint(payload: dict = Body(...)):
    q = payload.get("query", "")
    return run_agent(q)


@app.get("/report")
def report_md():
    return {"markdown": report.build_report()}


@app.post("/train")
def train():
    df = get_df()
    lb = train_all(df)
    return {"metrics": lb["metrics"], "models_ready": models_ready()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
