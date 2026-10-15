"""Demand-planning agent orchestrating the forecasting and inventory tools.

Two planners share the same tools:
  * LLM planner (Groq / OpenAI / Ollama, OpenAI-compatible tool calling) when a backend
    is configured: the model decides which tools to call, in a loop of up to 8 steps.
  * Offline planner otherwise: intent detection plus fixed multi-step plans, so the
    demo works with no API key.  Every tool call is recorded in the reasoning trace.
"""
from __future__ import annotations

import json
import re

from . import llm, tools
from .config import (STORES, DEPARTMENTS, GROQ_API_KEY, GROQ_MODEL, OPENAI_API_KEY,
                     OPENAI_MODEL, OLLAMA_BASE_URL, OLLAMA_MODEL)
from .model import MAX_FORECAST_HORIZON

SYSTEM_PROMPT = f"""You are a demand-planning agent for a retail supply chain (Walmart M5 data,
stores {", ".join(STORES)}; departments {", ".join(DEPARTMENTS)}).
Workflow for replenishment questions:
 1. run_risk_scan or detect_stockout_risk to find series at risk,
 2. forecast_demand, get_promo_calendar, detect_demand_spike and query_trends to explain why,
 3. recommend_reorder for the quantity and supplier split,
 4. draft_purchase_order only when the user asks to order/replenish; drafts need human approval.
Use the dashboard selection when the user gives no store/department, and say so.
Answer in under 200 words with concrete numbers, dates and next actions."""

MAX_LLM_STEPS = 8
WORKFLOW_MAX_ITEMS = 5


def _has(q: str, *phrases: str) -> bool:
    """Whole-word / whole-phrase match, so 'ok' does not match 'look' or 'po' 'report'."""
    return any(re.search(rf"\b{re.escape(p)}\b", q) for p in phrases)


def parse_horizon(q: str, default: int = 28) -> int:
    q = q.lower()
    m = re.search(r"\b(\d{1,3})\s*(?:-\s*)?(?:days?|d)\b", q)
    if m:
        days = int(m.group(1))
    else:
        m = re.search(r"\b(\d{1,2}|one|two|three|four|five|six|seven|eight)\s*weeks?\b", q)
        words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}
        if m:
            days = 7 * (int(m.group(1)) if m.group(1).isdigit() else words[m.group(1)])
        elif _has(q, "next week", "a week", "this week"):
            days = 7
        elif _has(q, "next month", "a month", "this month"):
            days = 28
        else:
            days = default
    return max(1, min(MAX_FORECAST_HORIZON, days))


def _parse_store_dept(query, default_store=None, default_dept=None):
    unknown = tools.unknown_store_tokens(query)
    if unknown:
        raise ValueError(f"Unknown store {', '.join(unknown)}. Configured stores: {', '.join(STORES)}")
    store, dept = tools._norm_store_dept(query)
    return (store or default_store or "CA_1").upper(), (dept or default_dept or "FOODS").upper()


class _Trace:
    def __init__(self):
        self.steps = []

    def call(self, thought: str, tool: str, args: dict) -> dict:
        obs = tools.execute_tool(tool, args)
        self.steps.append({"thought": thought, "tool": tool, "args": args, "observation": obs})
        return obs


def _fmt_trend(t: dict) -> str:
    if t.get("delta_pct") is None:
        return f"external trend for {t.get('category')} unavailable"
    return f"Google Trends '{t['query']}' {t['direction']} ({t['delta_pct']:+.1f}% vs 12-month average)"


def replenishment_workflow(trace: _Trace, horizon: int = 28, store: str | None = None,
                           dept: str | None = None, draft: bool = True) -> str:
    """Detect -> explain -> recommend -> draft, for every at-risk series (worst first)."""
    scan = trace.call("Scan every store/department for stockout risk.", "run_risk_scan", {"horizon": horizon})
    if scan.get("error"):
        return f"Risk scan failed: {scan['error']}"
    items = [i for i in scan.get("items", [])
             if (store is None or i["store"] == store) and (dept is None or i["dept"] == dept)]
    counts = scan.get("severity_counts", {})
    header = (f"Scanned {scan.get('items_evaluated', 0)} series: {counts.get('CRITICAL', 0)} CRITICAL, "
              f"{counts.get('WARNING', 0)} WARNING, {counts.get('OK', 0)} OK.")
    if not items:
        return header + " Nothing in scope needs an expedite order; routine weekly orders cover demand."

    lines, trend_cache, promo_cache = [header], {}, {}
    for item in items[:WORKFLOW_MAX_ITEMS]:
        s, d = item["store"], item["dept"]
        fc = trace.call(f"{s}/{d} is {item['risk_level']}: forecast demand to size the gap.",
                        "forecast_demand", {"store": s, "dept": d, "horizon": horizon})
        if s not in promo_cache:
            promo_cache[s] = trace.call(f"Check promotions and events that could lift {s} demand.",
                                        "get_promo_calendar", {"store": s, "days": 14})
        spike = trace.call(f"Check {s}/{d} for a demand spike.", "detect_demand_spike", {"store": s, "dept": d})
        if d not in trend_cache:
            trend_cache[d] = trace.call(f"Query the external trend signal for {d}.", "query_trends", {"category": d})
        tool = "draft_purchase_order" if draft else "recommend_reorder"
        res = trace.call(f"Size the expedite order and split it across suppliers for {s}/{d}.",
                         tool, {"store": s, "dept": d, "horizon": horizon})
        rec = res.get("po") or res.get("recommendation") or {}
        alloc = ", ".join(f"{a['qty']} from {a['supplier']} ({a['lead_time_days']}d)" for a in rec.get("allocations", []))
        promo = promo_cache[s]
        context = []
        if promo.get("snap_promo_days"):
            context.append(f"{len(promo['snap_promo_days'])} SNAP days in next 2 weeks")
        if promo.get("events"):
            context.append("events: " + ", ".join(e["event"] for e in promo["events"][:3]))
        if spike.get("spike") or spike.get("upcoming_spike"):
            context.append("demand spike detected")
        context.append(_fmt_trend(trend_cache[d]))
        po_note = ""
        if draft and rec.get("po_ids"):
            po_note = f" Draft PO #{', #'.join(str(i) for i in rec['po_ids'])} awaiting approval."
        stockout = item.get("projected_stockout_date")
        lines.append(
            f"• {s}/{d} [{item['risk_level']}] position {round(item['inventory_position'])} vs "
            f"{round(item['protection_demand'])} forecast demand to next delivery"
            + (f", projected stockout {stockout}" if stockout else "") + ". "
            + (f"Order {rec.get('recommended_qty', 0)} units: {alloc}." if alloc else "No order needed.")
            + po_note + f" Context: {'; '.join(context)}."
            + f" ({fc.get('summary', {}).get('total_demand', '?')} units forecast over {horizon}d.)")
    if len(items) > WORKFLOW_MAX_ITEMS:
        lines.append(f"{len(items) - WORKFLOW_MAX_ITEMS} more at-risk series; open the Reorders tab for all of them.")
    return "\n".join(lines)


def _deterministic(query: str, store: str | None = None, dept: str | None = None) -> dict:
    q = query.lower()
    trace = _Trace()
    explicit_store, explicit_dept = tools._norm_store_dept(query)
    store, dept = _parse_store_dept(query, store, dept)
    horizon = parse_horizon(q)

    def done(answer: str) -> dict:
        return {"steps": trace.steps, "answer": answer, "backend": "offline-router"}

    # 1. Severity counts ("how many critical ...")
    if _has(q, "how many", "count", "number of") and _has(q, "critical", "warning", "ok", "unknown", "risk", "risks",
                                                          "at risk", "at-risk", "stockout", "stockouts", "series"):
        obs = trace.call("Run a fresh risk scan before counting.", "run_risk_scan", {"horizon": horizon})
        if obs.get("status") != "COMPLETED":
            return done("The risk scan did not complete, so counts cannot be reported. " + obs.get("error", ""))
        items = [i for i in obs.get("all_items", [])
                 if (explicit_store is None or i["store"] == explicit_store)
                 and (explicit_dept is None or i["dept"] == explicit_dept)]
        scope = "/".join(x for x in (explicit_store, explicit_dept) if x) or "all stores and departments"
        counts = {}
        for item in items:
            counts[item["risk_level"]] = counts.get(item["risk_level"], 0) + 1
        asked = next((lvl for lvl in ("CRITICAL", "WARNING", "OK", "UNKNOWN") if _has(q, lvl.lower())), None)
        if asked:
            names = [f"{i['store']}/{i['dept']}" for i in items if i["risk_level"] == asked]
            return done(f"{counts.get(asked, 0)} of {len(items)} series in {scope} are {asked}"
                        + (f": {', '.join(names)}." if names else "."))
        return done(f"Scope {scope}, {len(items)} series: CRITICAL {counts.get('CRITICAL', 0)}, "
                    f"WARNING {counts.get('WARNING', 0)}, OK {counts.get('OK', 0)}, UNKNOWN {counts.get('UNKNOWN', 0)}.")

    # 2. Full agentic workflow
    if _has(q, "replenish", "replenishment", "autopilot", "plan", "what should i order", "what do i need to order",
            "restock", "workflow"):
        answer = replenishment_workflow(trace, horizon, explicit_store, explicit_dept, draft=True)
        return done(answer)

    if _has(q, "scan", "all risks", "risk scan", "portfolio", "at risk", "at-risk") and not explicit_dept:
        answer = replenishment_workflow(trace, horizon, explicit_store, None, draft=False)
        return done(answer + "\nSay 'plan replenishment' to draft the purchase orders.")

    if _has(q, "promo", "promos", "promotion", "promotions", "snap", "event", "events", "calendar", "holiday", "holidays"):
        obs = trace.call(f"Look up the promotional calendar for {store}.", "get_promo_calendar",
                         {"store": store, "days": horizon})
        events = "; ".join(f"{e['date']} {e['event']}" for e in obs.get("events", [])) or "none"
        return done(f"{store}, next {obs.get('days')} days from {obs.get('start')}: "
                    f"{len(obs.get('snap_promo_days', []))} SNAP promotion days "
                    f"({', '.join(obs.get('snap_promo_days', [])[:10])}). Events: {events}. "
                    "These are model inputs, so the forecast already reflects them.")

    if _has(q, "trend", "trends", "external", "market", "google"):
        obs = trace.call(f"Query the external trend signal for {dept}.", "query_trends", {"category": dept})
        return done(_fmt_trend(obs) + f". Source: {obs.get('source')}.")

    if _has(q, "reorder", "re-order", "order", "orders", "po", "purchase", "buy"):
        risk = trace.call(f"Check stockout risk for {store}/{dept} first.", "detect_stockout_risk",
                          {"store": store, "dept": dept})
        if risk.get("error"):
            return done(f"Could not assess {store}/{dept}: {risk['error']}")
        trace.call("Check upcoming promotions that affect the order window.", "get_promo_calendar",
                   {"store": store, "days": 14})
        res = trace.call(f"Size the order and split it across suppliers for {store}/{dept}.",
                         "draft_purchase_order", {"store": store, "dept": dept, "horizon": horizon})
        po = res.get("po", {})
        if res.get("error"):
            return done(f"Could not draft a PO: {res['error']}")
        if not po.get("po_ids"):
            return done(f"{store}/{dept} is {risk['risk_level']}: {po.get('rationale').rstrip('.')}. No purchase order drafted.")
        alloc = "; ".join(f"{a['qty']} units from {a['supplier']} ({a['lead_time_days']}d lead, {a['reason']})"
                          for a in po["allocations"])
        return done(f"{store}/{dept} is {risk['risk_level']}. Recommended {po['recommended_qty']} units: {alloc}. "
                    f"Draft PO #{', #'.join(map(str, po['po_ids']))} ({po['status']}). {po['rationale']}")

    if _has(q, "inventory", "stock level", "stock levels", "on hand", "snapshot"):
        obs = trace.call(f"Fetch inventory for {store}/{dept}.", "inventory_snapshot", {"store": store, "dept": dept})
        rows = obs.get("rows", [])
        if not rows:
            return done("No inventory rows found.")
        r = rows[0]
        return done(f"{store}/{dept}: on hand {round(r['on_hand'])}, inbound {round(r.get('on_order') or 0)}, "
                    f"safety stock {round(r['safety_stock'])}, supplier {r['supplier']} (lead {r['lead_time_days']}d), "
                    f"next review in {r.get('days_to_review')}d.")

    if _has(q, "spike", "spikes", "surge", "anomaly", "jump"):
        obs = trace.call(f"Detect demand spikes for {store}/{dept}.", "detect_demand_spike", {"store": store, "dept": dept})
        return done(f"{store}/{dept}: last 7 days avg {obs['recent_7d_avg']} ({obs['change_pct']:+}% vs 90-day mean "
                    f"{obs['baseline_mean']}), next 7 days forecast avg {obs['forecast_next_7d_avg']} "
                    f"({obs['upcoming_change_pct']:+}%). Spike threshold {obs['threshold']}: "
                    f"recent spike={obs['spike']}, upcoming spike={obs['upcoming_spike']}.")

    if _has(q, "forecast", "predict", "projection", "demand", "sales"):
        obs = trace.call(f"Forecast {horizon}-day demand for {store}/{dept}.", "forecast_demand",
                         {"store": store, "dept": dept, "horizon": horizon})
        if obs.get("error"):
            return done(f"Forecast failed: {obs['error']}")
        s = obs["summary"]
        r = trace.call("Check stockout risk against the forecast.", "detect_stockout_risk", {"store": store, "dept": dept})
        return done(f"{store}/{dept}: {horizon}-day forecast {s['total_demand']} units (80% range "
                    f"{s['total_lower']}–{s['total_upper']}), avg {s['avg_daily']}/day, {s['first_day']} to "
                    f"{s['last_day']}. Risk: {r['risk_level']} — {r['classification_reason']}.")

    if _has(q, "risk", "stockout", "stock-out", "shortage", "run out"):
        obs = trace.call(f"Assess stockout risk for {store}/{dept}.", "detect_stockout_risk", {"store": store, "dept": dept})
        stockout = obs.get("projected_stockout_date")
        return done(f"{store}/{dept}: {obs['risk_level']} — {obs['classification_reason']}. Position "
                    f"{round(obs['inventory_position'])} (on hand {round(obs['on_hand'])} + inbound {round(obs['on_order'])}) "
                    f"vs reorder point {round(obs['reorder_point'])}"
                    + (f"; projected stockout {stockout}" if stockout else "") + ".")

    return done("I can run the replenishment workflow ('plan replenishment'), forecast demand, check stockout risk, "
                "count at-risk series, size reorders with supplier allocation, show promotions/events, demand "
                "spikes, external trends and inventory. Name a store (CA_1, CA_2, TX_1, TX_2, WI_1) and "
                "department (FOODS, HOBBIES, HOUSEHOLD), or I use the dashboard selection.")


def _client():
    from openai import OpenAI
    prov = llm.backend_name()
    if prov == "groq":
        return prov, OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1"), GROQ_MODEL
    if prov == "openai":
        return prov, OpenAI(api_key=OPENAI_API_KEY), OPENAI_MODEL
    return prov, OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL), OLLAMA_MODEL


def _observation_for_llm(obs: dict) -> str:
    slim = dict(obs)
    if "forecast" in slim:  # daily rows are noise for the LLM; the summary carries the totals
        slim["forecast"] = slim["forecast"][:7]
    if "all_items" in slim:
        slim.pop("all_items")
    return json.dumps(slim, default=str)[:6000]


def run_agent(query: str, store: str | None = None, dept: str | None = None, client_factory=None) -> dict:
    if not llm.is_available() and client_factory is None:
        return _deterministic(query, store, dept)
    try:
        prov, client, model = (client_factory or _client)()
    except Exception:
        return _deterministic(query, store, dept)

    context_store, context_dept = _parse_store_dept("", store, dept)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{query}\n[Dashboard selection: {context_store}/{context_dept}]"}]
    steps = []
    try:
        for _ in range(MAX_LLM_STEPS):
            resp = client.chat.completions.create(model=model, messages=messages, tools=tools.OPENAI_TOOLS,
                                                  tool_choice="auto", temperature=0.2, max_tokens=700)
            msg = resp.choices[0].message
            calls = getattr(msg, "tool_calls", None) or []
            if not calls:
                return {"steps": steps, "answer": msg.content or "", "backend": prov}
            # Rebuild the assistant turn explicitly; provider SDK objects carry extra fields
            # (refusal, audio, ...) that some OpenAI-compatible servers reject.
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [{"id": tc.id, "type": "function",
                                             "function": {"name": tc.function.name,
                                                          "arguments": tc.function.arguments or "{}"}}
                                            for tc in calls]})
            for tc in calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                obs = tools.execute_tool(tc.function.name, args)
                steps.append({"thought": (msg.content or f"Calling {tc.function.name}."),
                              "tool": tc.function.name, "args": args, "observation": obs})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": _observation_for_llm(obs)})
    except Exception as e:
        r = _deterministic(query, store, dept)
        r["backend"] = f"offline-router (LLM failed: {type(e).__name__})"
        return r
    return {"steps": steps, "answer": "Stopped after the maximum number of tool steps.", "backend": prov}
