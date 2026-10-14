"""ReAct-style agent orchestrating the demand-forecasting tools."""
from __future__ import annotations

import json
import re

from . import llm, tools
from .config import STORES, DEPARTMENTS

SYSTEM_PROMPT = """You are a Demand Forecasting Agent for a logistics supply chain.
You have tools to forecast demand, detect stockout risk, query external trends,
recommend reorders, and inspect inventory. Always pick the right tool(s) for the user's
request, then give a concise business-friendly answer with numbers and actions.
When the user does not specify a store/department, default to CA_1 / FOODS and say so.
Keep answers under 200 words."""


def _parse_store_dept(query):
    store, dept = tools._norm_store_dept(query)
    return store or "CA_1", dept or "FOODS"


def _deterministic(query: str) -> dict:
    q = query.lower()
    steps = []
    store, dept = _parse_store_dept(query)
    if any(k in q for k in ["scan", "all risk", "risk scan", "portfolio"]):
        obs = tools.execute_tool("run_risk_scan", {})
        steps.append({"thought": "Scanning all store/department pairs for stockout risk.", "tool": "run_risk_scan",
                      "args": {}, "observation": obs})
        top = obs.get("items", [])[:5]
        ans = f"Scanned all series. Flagged {obs.get('flagged', 0)} at-risk combos. " + \
              "; ".join(f"{i['store']}/{i['dept']}: {i['days_of_supply']} days supply ({i['risk_level']})" for i in top)
        return {"steps": steps, "answer": ans or "No risks flagged.", "backend": "offline-router"}
    if any(k in q for k in ["trend", "external", "market"]):
        cat = dept if dept in ["FOODS", "HOBBIES", "HOUSEHOLD"] else "FOODS"
        obs = tools.execute_tool("query_trends", {"category": cat})
        steps.append({"thought": f"Querying external trend signal for {cat}.", "tool": "query_trends",
                      "args": {"category": cat}, "observation": obs})
        ans = f"Trend for {cat}: {obs['direction']} ({obs['delta_pct']}% vs baseline, latest {obs['latest']}). Source: {obs['source']}."
        return {"steps": steps, "answer": ans, "backend": "offline-router"}
    if any(k in q for k in ["reorder", "order", "po", "purchase"]):
        obs = tools.execute_tool("recommend_reorder", {"store": store, "dept": dept, "horizon": 28})
        steps.append({"thought": f"Computing reorder for {store}/{dept}.", "tool": "recommend_reorder",
                      "args": {"store": store, "dept": dept}, "observation": obs})
        rec = obs["recommendation"]
        ans = (f"Reorder {rec['recommended_qty']} units of {store}/{dept} from {rec['supplier']} "
               f"(lead time {rec['lead_time_days']}d). Risk: {rec['risk_level']}. Draft PO #{obs['po']['po_id']} created. "
               f"Rationale: {rec['rationale']}")
        return {"steps": steps, "answer": ans, "backend": "offline-router"}
    if any(k in q for k in ["inventory", "stock level", "on hand", "snapshot"]):
        obs = tools.execute_tool("inventory_snapshot", {"store": store, "dept": dept})
        rows = obs.get("rows", [])
        if not rows:
            return {"steps": steps, "answer": "No inventory rows found. Seed data first.", "backend": "offline-router"}
        r = rows[0]
        steps.append({"thought": f"Fetching inventory for {store}/{dept}.", "tool": "inventory_snapshot",
                      "args": {"store": store, "dept": dept}, "observation": obs})
        ans = (f"{store}/{dept}: on hand {r['on_hand']}, safety {r['safety_stock']}, "
               f"supplier {r['supplier']} (lead {r['lead_time_days']}d).")
        return {"steps": steps, "answer": ans, "backend": "offline-router"}
    if any(k in q for k in ["risk", "stockout", "shortage"]):
        obs = tools.execute_tool("detect_stockout_risk", {"store": store, "dept": dept})
        steps.append({"thought": f"Assessing stockout risk for {store}/{dept}.", "tool": "detect_stockout_risk",
                      "args": {"store": store, "dept": dept}, "observation": obs})
        ans = (f"{store}/{dept}: risk={obs['risk_level']}, days of supply={obs['days_of_supply']}, "
               f"on hand={obs['on_hand']}, daily demand={obs['avg_daily_demand']}, lead time={obs['lead_time_days']}d.")
        return {"steps": steps, "answer": ans, "backend": "offline-router"}
    # default: forecast
    obs = tools.execute_tool("forecast_demand", {"store": store, "dept": dept, "horizon": 28})
    steps.append({"thought": f"Forecasting 28-day demand for {store}/{dept}.", "tool": "forecast_demand",
                  "args": {"store": store, "dept": dept, "horizon": 28}, "observation": obs})
    s = obs.get("summary") or {}
    ans = (f"{store}/{dept}: 28-day projected demand = {s.get('total_demand')} units "
           f"(avg {s.get('avg_daily')}/day) from {s.get('first_day')} to {s.get('last_day')}.")
    if "risk" not in q:
        r = tools.execute_tool("detect_stockout_risk", {"store": store, "dept": dept})
        steps.append({"thought": "Checking inventory risk off the forecast.", "tool": "detect_stockout_risk",
                      "args": {"store": store, "dept": dept}, "observation": r})
        ans += f" Current risk: {r['risk_level']} ({r['days_of_supply']} days of supply)."
    return {"steps": steps, "answer": ans, "backend": "offline-router"}


def run_agent(query: str) -> dict:
    if not llm.is_available():
        return _deterministic(query)
    try:
        from openai import OpenAI
    except Exception:
        return _deterministic(query)
    # resolve client
    prov = llm.backend_name()
    if prov == "groq":
        from .config import GROQ_API_KEY
        client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
        model = llm.GROQ_MODEL if hasattr(llm, "GROQ_MODEL") else "llama-3.3-70b-versatile"
    elif prov == "openai":
        from .config import OPENAI_API_KEY, OPENAI_MODEL
        client = OpenAI(api_key=OPENAI_API_KEY)
        model = OPENAI_MODEL
    else:
        from .config import OLLAMA_BASE_URL, OLLAMA_MODEL
        client = OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL)
        model = OLLAMA_MODEL

    # fix model attr (llm module doesn't re-export)
    if prov == "groq":
        model = __import__("os").getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    elif prov == "openai":
        model = __import__("os").getenv("OPENAI_MODEL", "gpt-4o-mini")
    else:
        model = __import__("os").getenv("OLLAMA_MODEL", "llama3.2")

    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query}]
    steps = []
    try:
        for _ in range(6):
            resp = client.chat.completions.create(model=model, messages=messages,
                                                tools=tools.OPENAI_TOOLS, tool_choice="auto",
                                                temperature=0.2, max_tokens=700)
            msg = resp.choices[0].message
            if getattr(msg, "tool_calls", None):
                messages.append(msg.model_dump())
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except Exception:
                        args = {}
                    obs = tools.execute_tool(tc.function.name, args)
                    steps.append({"thought": f"Calling {tc.function.name}.",
                                  "tool": tc.function.name, "args": args, "observation": obs})
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                                    "content": json.dumps(obs)[:4000]})
            else:
                return {"steps": steps, "answer": msg.content or "", "backend": prov}
    except Exception as e:
        # fall back to deterministic
        r = _deterministic(query)
        r["backend"] = f"offline-router (llm failed: {e})"
        return r
    return {"steps": steps, "answer": "Reached max tool steps.", "backend": prov}
