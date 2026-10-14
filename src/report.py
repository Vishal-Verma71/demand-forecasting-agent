"""Auto-generated Markdown report of forecasts, risks, reorder actions."""
from __future__ import annotations

from datetime import datetime

from . import inventory
from .config import STORES, DEPARTMENTS
from .db import get_alerts, get_pos


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
    lines.append("| Store | Dept | On hand | Daily demand | Days of supply | Risk |")
    lines.append("|---|---|---|---|---|---|")
    for s in STORES:
        for d in DEPARTMENTS:
            st = inventory.stock_status(s, d)
            lines.append(f"| {s} | {d} | {st['on_hand']} | {st['avg_daily_demand']} | {st['days_of_supply']} | {st['risk_level']} |")
    lines.append("")
    lines.append("## 3. Reorder Recommendations")
    lines.append("| Store | Dept | Qty | Supplier | Lead (d) | Risk |")
    lines.append("|---|---|---|---|---|---|")
    for s in STORES:
        for d in DEPARTMENTS:
            r = inventory.reorder_recommendation(s, d)
            lines.append(f"| {s} | {d} | {r['recommended_qty']} | {r['supplier']} | {r['lead_time_days']} | {r['risk_level']} |")
    lines.append("")
    lines.append("## 4. Recent Alerts")
    for a in get_alerts(20):
        lines.append(f"- [{a['level']}] {a['store']}/{a['dept']}: {a['message']}")
    lines.append("")
    lines.append("## 5. Draft Purchase Orders")
    for p in get_pos(20):
        lines.append(f"- PO#{p['id']} {p['store']}/{p['dept']} — {p['qty']} units from {p['supplier']} (lead {p['lead_time_days']}d) [{p['status']}]")
    lines.append("")
    lines.append("## 6. Method")
    lines.append("HistGradientBoostingRegressor (scikit-learn global model). Agent uses ReAct loop with 7 tools,")
    lines.append("supports Groq/OpenAI/Ollama or a fully offline deterministic router. External trend via pytrends (fallback if unavailable).")
    return "\n".join(lines)
