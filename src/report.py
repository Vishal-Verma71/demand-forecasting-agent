"""Auto-generated Markdown report of forecasts, risks, reorder actions."""
from __future__ import annotations

from datetime import datetime

from . import inventory
from .config import STORES, DEPARTMENTS
from .db import get_alerts, get_pos
from .eval import load_evaluation
from .data import data_diagnostics


def build_report() -> str:
    lines = []
    lines.append(f"# Demand Forecasting Agent — Supply Chain Report")
    lines.append(f"")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("## 1. Scope")
    lines.append(f"Stores: {', '.join(STORES)} | Departments: {', '.join(DEPARTMENTS)}")
    lines.append("")
    lines.append("## 2. Inventory Risk Snapshot")
    lines.append("| Store | Dept | On hand | Inbound | Reorder point | Days of supply | Projected stockout | Risk |")
    lines.append("|---|---|---:|---:|---:|---:|---|---|")
    for s in STORES:
        for d in DEPARTMENTS:
            st = inventory.stock_status(s, d)
            lines.append(f"| {s} | {d} | {round(st['on_hand'])} | {round(st['on_order'])} | {round(st['reorder_point'])} | "
                         f"{st['days_of_supply']} | {st['projected_stockout_date'] or '-'} | {st['risk_level']} |")
    lines.append("")
    lines.append("## 3. Reorder Recommendations")
    lines.append("| Store | Dept | Risk | Qty | Supplier split |")
    lines.append("|---|---|---|---:|---|")
    for s in STORES:
        for d in DEPARTMENTS:
            r = inventory.reorder_recommendation(s, d)
            if r["recommended_qty"] <= 0:
                continue
            split = "; ".join(f"{a['qty']} × {a['supplier']} ({a['lead_time_days']}d)" for a in r["allocations"])
            lines.append(f"| {s} | {d} | {r['risk_level']} | {r['recommended_qty']} | {split} |")
    lines.append("")
    lines.append("## 4. Recent Alerts")
    for a in get_alerts(20):
        lines.append(f"- [{a['level']}] {a['store']}/{a['dept']}: {a['message']}")
    lines.append("")
    lines.append("## 5. Purchase Orders (drafted by the agent, approved by a planner)")
    for p in get_pos(20):
        lines.append(f"- PO#{p['id']} {p['store']}/{p['dept']} — {p['qty']} units from {p['supplier']} (lead {p['lead_time_days']}d) [{p['status']}]")
    lines.append("")
    lines.append("## 6. Method")
    lines.append("LightGBM global model over all 15 series with lag, rolling, calendar, SNAP-promotion and event features,")
    lines.append("forecast recursively. Risk = inventory position vs forecast demand until the next scheduled delivery plus")
    lines.append("z·σ·√L safety stock. The agent calls 9 tools through an LLM (Groq/OpenAI/Ollama) or an offline planner;")
    lines.append("external demand signal from Google Trends (pytrends), reported as unavailable when it cannot be reached.")
    evaluation = load_evaluation()
    if evaluation:
        lines.append("")
        lines.append("## 7. Chronological Forecast Evaluation")
        lines.append(f"Dataset: {evaluation['dataset']}")
        lines.append(f"Holdout period: {evaluation['primary_test_period']['from']} to {evaluation['primary_test_period']['to']}")
        lines.append(f"Horizon: {evaluation['forecast_horizon']} days; windows: {evaluation['windows']}; test observations: {evaluation['primary_test_observations']}")
        primary = evaluation["primary_overall_macro"]
        lines.append("| Model | MAE | RMSE | WAPE | MAPE non-zero |")
        lines.append("|---|---:|---:|---:|---:|")
        for model_name in ("naive", "seasonal_naive", "model"):
            metric = primary[model_name]
            lines.append(f"| {model_name} | {metric['MAE']} | {metric['RMSE']} | {metric['WAPE']} | {metric['MAPE_nonzero']} |")
        lines.append(f"Model beats naive: **{primary['model']['MAE'] < primary['naive']['MAE']}**")
        lines.append(f"Model beats seasonal naive: **{primary['model']['MAE'] < primary['seasonal_naive']['MAE']}**")
        lines.append(f"Predictions file: {evaluation['files']['predictions']}")
    diagnostics = data_diagnostics()
    lines.append("")
    lines.append("## 8. Data Source Diagnostics")
    lines.append(f"Cached rows: {diagnostics['cached_data_rows']}")
    lines.append("M5 selected store/department combinations:")
    for item in sorted(diagnostics["m5_selected_item_rows_by_store"].items()):
        lines.append(f"- {item[0]}: {item[1]} source items")
    lines.append("Cached rows by store/department:")
    for key, value in diagnostics["cached_rows_by_store_department"].items():
        lines.append(f"- {key}: {value} rows")
    lines.append("")
    lines.append("M5 has no stock or supplier data: on-hand and inbound come from a simulated weekly min/max ledger replayed")
    lines.append("over actual M5 sales, and supplier lead times, costs and reliability are documented assumptions.")
    return "\n".join(lines)
