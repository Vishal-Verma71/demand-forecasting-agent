# Architecture

## Components

| Layer | File | Responsibility |
|---|---|---|
| Data | `src/data.py` | Load and aggregate M5 sales, save the future M5 calendar, simulate the store inventory ledger, define supplier terms |
| Model | `src/model.py` | LightGBM global model, recursive forecasting, empirical prediction intervals, thread-safe forecast cache |
| Evaluation | `src/eval.py` | 3-window chronological backtest against naive and seasonal-naive baselines |
| Inventory | `src/inventory.py` | Protection-interval risk, safety stock, expedite sizing, supplier allocation, PO drafting, spike detection, promo calendar |
| Trends | `src/trends.py` | Google Trends via pytrends (1-hour cache, 10 s timeout, explicit "unavailable") |
| Tools | `src/tools.py` | 9 tool schemas (OpenAI function-calling format) and the dispatcher |
| Agent | `src/agent.py` | LLM tool loop (Groq/OpenAI/Ollama) and offline planner, including the `replenishment_workflow` |
| Persistence | `src/db.py` | SQLite: inventory snapshot, alerts (de-duplicated), purchase orders (DRAFT → APPROVED/REJECTED) |
| API | `src/api.py` | FastAPI: validated endpoints for the UI |
| UI | `frontend/src/app/page.tsx` | Next.js dashboard |

## Agent tools

| Tool | Purpose |
|---|---|
| `run_risk_scan` | Assess all 15 series, raise alerts, sort worst first |
| `detect_stockout_risk` | Risk level for one series, plus the date it runs out with no delivery |
| `forecast_demand` | LightGBM forecast with an 80% interval |
| `detect_demand_spike` | Recent and upcoming spike vs a 90-day baseline |
| `get_promo_calendar` | Upcoming SNAP days and events |
| `query_trends` | External Google Trends signal |
| `recommend_reorder` | Expedite quantity and supplier split (no side effects) |
| `draft_purchase_order` | Create idempotent DRAFT POs for human approval |
| `inventory_snapshot` | On hand, inbound, safety stock, supplier |

## Request flow

1. The UI calls `POST /agent {query, store, dept}`.
2. `run_agent` uses the LLM tool loop if a key is configured, otherwise `_deterministic`.
3. Tools call the forecaster and the inventory engine, which read the cached dataset and SQLite.
4. The response contains the answer and every step `{thought, tool, args, observation}`.
5. Draft POs appear in the Reorders tab. `POST /pos/{id}/approve` adds the quantity to inbound
   stock, so the next scan reflects it.

## Key decisions

* **LightGBM global model.** One model learns shared weekly and seasonal patterns across all
  series, and trains in seconds. It is pinned to `!=4.7.0` because 4.7.0 crashes natively with
  numpy 1.26 on Windows.
* **Recursive forecasting, shared by training and serving.** The backtest error is the
  error users will see.
* **Protection-interval risk.** This is the correct check for a periodic-review store. A plain
  lead-time check misses stock that runs out before the next weekly order arrives.
* **Human in the loop.** The agent drafts orders and people approve them. Drafts are idempotent.
* **Works offline.** No API key is needed, and an LLM failure falls back to the offline planner.
* **Isolated tests.** An autouse fixture gives each test its own SQLite file. `SUPPLY_CHAIN_DB`
  overrides the path at runtime.
