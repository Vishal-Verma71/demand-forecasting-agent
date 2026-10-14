"""Streamlit app: Demand Forecasting Agent for logistics supply chain."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import db, inventory, report
from src.agent import run_agent
from src.config import STORES, DEPARTMENTS, MODEL_DIR
from src.data import load_or_generate
from src.features import build_series
from src.llm import backend_name, is_available
from src.model import (forecast_demand, models_ready, train_all, train_lightgbm,
                      predict_lightgbm)
from src.trends import query_trend

st.set_page_config(page_title="Demand Forecasting Agent", page_icon="📦", layout="wide")

# ---------- styling ----------
st.markdown("""
<style>
:root { --card:#101826; --accent:#6366f1; }
.main > div { padding-top: 1rem; }
.kpi-card { background: linear-gradient(135deg,#1e293b,#0f172a); border:1px solid #263042;
  border-radius:16px; padding:18px 20px; box-shadow:0 8px 24px rgba(0,0,0,.25); }
.kpi-label { color:#94a3b8; font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; }
.kpi-value { color:#fff; font-size:1.9rem; font-weight:700; margin-top:4px; }
.kpi-sub { color:#64748b; font-size:.78rem; margin-top:2px; }
.header { background: linear-gradient(90deg,#6366f1,#22d3ee); -webkit-background-clip:text;
  -webkit-text-fill-color:transparent; font-size:2.2rem; font-weight:800; }
.hero { background: radial-gradient(1200px 300px at 10% -20%, #312e81 0%, #0f172a 60%);
  padding:22px 26px; border-radius:18px; margin-bottom:18px; border:1px solid #1f2a3d; }
.badge { display:inline-block; padding:3px 10px; border-radius:999px; font-size:.75rem;
  background:#1f2937; color:#cbd5e1; margin-right:6px; border:1px solid #334155; }
.tool-step { background:#0b1220; border-left:3px solid #6366f1; padding:10px 12px;
  margin:6px 0; border-radius:8px; font-family: ui-monospace, monospace; font-size:.82rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_data():
    db.init_db()
    df = load_or_generate()
    inventory.seed_inventory_from_data(df)
    return df


df = get_data()

# ---------- sidebar ----------
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    store = st.selectbox("Store", STORES, index=0)
    dept = st.selectbox("Department", DEPARTMENTS, index=0)
    horizon = st.slider("Forecast horizon (days)", 7, 56, 28)
    st.divider()

    if models_ready():
        st.success("Models ready ✅")
    else:
        st.warning("Models not trained yet")
        if st.button("🚀 Train model (one-time)", type="primary"):
            with st.spinner("Training LightGBM..."):
                try:
                    lb = train_all(df)
                    st.success(f"Trained. MAPE {lb['metrics']['MAPE']}%")
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"Training failed: {e}")
    st.divider()
    st.markdown(f"**LLM backend:** `{backend_name()}`" + (" ✅" if is_available() else " (offline router)"))
    try:
        mm = json.loads((MODEL_DIR / "gb_metrics.json").read_text())
        st.caption(f"Model: GB · MAPE {mm.get('MAPE')}% · MAE {mm.get('MAE')}")
    except Exception:
        pass
    st.caption("Set GROQ_API_KEY / OPENAI_API_KEY in .env for richer agent chat. Works fully offline otherwise.")

# ---------- header ----------
st.markdown('<div class="hero"><h1 class="header">Demand Forecasting Agent</h1>'
            '<p style="color:#cbd5e1;margin:4px 0 10px">Logistics supply chain intelligence · Prophet + LightGBM · ReAct agent · Stockout risk & auto-reorder</p>'
            f'<span class="badge">Stores: {len(STORES)}</span>'
            f'<span class="badge">Departments: {len(DEPARTMENTS)}</span>'
            f'<span class="badge">Rows: {len(df):,}</span>'
            f'<span class="badge">Horizon: {horizon}d</span></div>', unsafe_allow_html=True)

# KPI row
total_rows = len(df)
stores_n = df["Store"].nunique()
now_inv = inventory.get_inventory()
at_risk = sum(1 for r in (inventory.run_risk_scan() if False else []) )
risk_items = []
at_risk_n = 0
for _s in STORES:
    for _d in DEPARTMENTS:
        try:
            _st = inventory.stock_status(_s, _d)
            if _st["risk_level"] in ("CRITICAL", "WARNING"):
                at_risk_n += 1
                risk_items.append(_st)
        except Exception:
            pass
total_on_hand = sum(r["on_hand"] for r in now_inv) if now_inv else 0

c1, c2, c3, c4 = st.columns(4)
def kpi(col, label, value, sub):
    col.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>',
                unsafe_allow_html=True)
kpi(c1, "Data rows", f"{total_rows:,}", f"{stores_n} stores × {df['Dept'].nunique()} depts")
kpi(c2, "On-hand units", f"{total_on_hand:,.0f}", "across all SKUs")
kpi(c3, "At-risk SKUs", f"{at_risk_n}", f"horizon {horizon}d")
po_count = len(db.get_pos(1000))
kpi(c4, "Draft POs", f"{po_count}", "auto-generated reorder drafts")

tabs = st.tabs(["📊 Dashboard", "🤖 Agent", "📈 Forecast", "🛒 Reorders", "📄 Report"])

# ================= Dashboard =================
with tabs[0]:
    colA, colB = st.columns([2, 1])
    with colA:
        st.subheader("Inventory risk map")
        rows = []
        for s in STORES:
            for d in DEPARTMENTS:
                stt = inventory.stock_status(s, d)
                rows.append({"Store": s, "Dept": d, "Days of supply": stt["days_of_supply"],
                             "Risk": stt["risk_level"], "On hand": stt["on_hand"]})
        rdf = pd.DataFrame(rows)
        fig = px.bar(rdf, x="Store", y="Days of supply", color="Risk", barmode="group",
                     facet_col="Dept", color_discrete_map={"OK": "#22c55e",
                                                         "WARNING": "#f59e0b", "CRITICAL": "#ef4444"})
        fig.update_layout(height=420, margin=dict(t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        st.subheader("Risk feed")
        if risk_items:
            for r in risk_items:
                color = "🔴" if r["risk_level"] == "CRITICAL" else "🟡"
                st.markdown(f"{color} **{r['store']}/{r['dept']}** — {r['days_of_supply']} days supply")
        else:
            st.success("No at-risk SKUs 🎉")
        if st.button("🔎 Run full risk scan (raise alerts)"):
            a = inventory.run_risk_scan()
            st.info(f"Flagged {len(a)} combos and stored alerts.")
        st.divider()
        st.subheader("External trend")
        cat = st.selectbox("Category", DEPARTMENTS, key="trcat")
        tr = query_trend(cat)
        st.metric("Trend direction", tr["direction"], f"{tr['delta_pct']}%")
        st.caption(tr["source"])

    st.subheader("Recent demand (selected series)")
    s_sub = build_series(df, store, dept).tail(90)
    if not s_sub.empty:
        fig2 = px.line(s_sub, x="Date", y="Sales", title=f"{store} / {dept} — last 90 days")
        fig2.update_layout(height=320, margin=dict(t=40, b=10))
        st.plotly_chart(fig2, use_container_width=True)

# ================= Agent =================
with tabs[1]:
    st.subheader("Ask the agent")
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("steps"):
                with st.expander("Reasoning trace (tool calls)"):
                    for stp in msg["steps"]:
                        st.markdown(f'<div class="tool-step"><b>{stp["tool"]}</b> {json.dumps(stp["args"])}<br>'
                                    f'<span style="color:#94a3b8">{str(stp["observation"])[:600]}</span></div>',
                                    unsafe_allow_html=True)
    prompt = st.chat_input("e.g. Forecast 28 days for TX_1 HOUSEHOLD and check risk")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.spinner("Agent thinking..."):
            result = run_agent(prompt)
        answer = result["answer"]
        with st.chat_message("assistant"):
            st.markdown(answer)
            with st.expander("Reasoning trace (tool calls)"):
                for stp in result["steps"]:
                    st.markdown(f'<div class="tool-step"><b>{stp["tool"]}</b> {json.dumps(stp["args"])}<br>'
                                f'<span style="color:#94a3b8">{str(stp["observation"])[:700]}</span></div>',
                                unsafe_allow_html=True)
        st.session_state.chat.append({"role": "user", "content": prompt})
        st.session_state.chat.append({"role": "assistant", "content": answer, "steps": result["steps"]})

    st.divider()
    st.caption("Quick prompts:")
    qp = st.columns(4)
    for i, q in enumerate(["Forecast CA_1 FOODS 28 days",
                            "Stockout risk for TX_2 HOBBIES",
                            "Reorder WI_1 HOUSEHOLD",
                            "Run a risk scan across all"]):
        if qp[i].button(q):
            st.session_state["_qp"] = q
    if "_qp" in st.session_state:
        q = st.session_state.pop("_qp")
        with st.spinner("Agent thinking..."):
            result = run_agent(q)
        st.session_state.chat.append({"role": "user", "content": q})
        st.session_state.chat.append({"role": "assistant", "content": result["answer"], "steps": result["steps"]})
        st.rerun()

# ================= Forecast =================
with tabs[2]:
    st.subheader(f"Forecast · {store} / {dept} · {horizon} days")
    if not models_ready():
        st.info("Train models first (sidebar).")
    else:
        fc = forecast_demand(store, dept, horizon)
        try:
            lgb = predict_lightgbm(store, dept, horizon)
        except Exception:
            lgb = None
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fc["Date"], y=fc["yhat"], name="Prophet forecast", line=dict(color="#6366f1", width=3)))
        fig.add_trace(go.Scatter(x=fc["Date"], y=fc["yhat_upper"], name="Upper", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=fc["Date"], y=fc["yhat_lower"], name="Confidence band",
                                 fill="tonexty", fillcolor="rgba(99,102,241,.18)", line=dict(width=0)))
        if lgb is not None:
            fig.add_trace(go.Scatter(x=lgb["Date"], y=lgb["yhat"], name="LightGBM", line=dict(color="#22d3ee", dash="dash")))
        # history
        sub = build_series(df, store, dept).tail(60)
        fig.add_trace(go.Scatter(x=sub["Date"], y=sub["Sales"], name="History",
                                 line=dict(color="#94a3b8", width=1.5)))
        fig.update_layout(height=460, margin=dict(t=20, b=10), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total projected demand", f"{fc['yhat'].sum():,.0f}")
        c2.metric("Avg daily", f"{fc['yhat'].mean():.1f}")
        c3.metric("Peak day", f"{fc['yhat'].max():.0f}")
        # metrics
        try:
            mm = json.loads((MODEL_DIR / "gb_metrics.json").read_text())
            st.caption(f"Gradient-Boosting holdout metrics: MAE {mm.get('MAE')} · MAPE {mm.get('MAPE')}% · RMSE {mm.get('RMSE')} · Baseline MAE (lag7) {mm.get('Baseline_MAE_lag7')}")
        except Exception:
            pass
        st.subheader("Forecast table")
        out = fc.copy()
        out["Date"] = out["Date"].astype(str)
        st.dataframe(out, use_container_width=True, height=320)

# ================= Reorders =================
with tabs[3]:
    st.subheader("Reorder recommendations")
    recs = []
    for s in STORES:
        for d in DEPARTMENTS:
            r = inventory.reorder_recommendation(s, d, horizon)
            recs.append(r)
    rdf2 = pd.DataFrame(recs)
    st.dataframe(rdf2[["store", "dept", "recommended_qty", "supplier", "lead_time_days", "risk_level", "rationale"]],
                 use_container_width=True, height=420)
    col1, col2 = st.columns(2)
    with col1:
        sel_store = st.selectbox("Store", STORES, key="po_store")
    with col2:
        sel_dept = st.selectbox("Dept", DEPARTMENTS, key="po_dept")
    if st.button("✅ Create draft PO"):
        res = inventory.approve_po(sel_store, sel_dept)
        st.success(f"Draft PO #{res['po_id']} created: {res['recommended_qty']} units from {res['supplier']}")
        st.rerun()
    st.subheader("Draft POs")
    st.dataframe(pd.DataFrame(db.get_pos(100)), use_container_width=True)

# ================= Report =================
with tabs[4]:
    st.subheader("Auto-generated report")
    rep = report.build_report()
    st.download_button("⬇️ Download report (Markdown)", rep, file_name="supply_chain_report.md",
                       mime="text/markdown")
    st.markdown(rep)
