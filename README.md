# Demand Forecasting Agent: Supply Chain Replenishment

An agentic workflow for the **Latentsig review, problem statement 2 (Demand Forecasting Agent
Workflow)**. A **LightGBM** model trained on Walmart **M5** sales is the agent's core tool. The
agent finds store/department series that will run out before their next scheduled delivery,
explains why (promotions, demand spikes, Google Trends), sizes expedite orders, splits them
across suppliers, and drafts purchase orders for a person to approve.

📄 **Full report:** [docs/REPORT.pdf](docs/REPORT.pdf) ([Markdown version](docs/REPORT.md)): problem,
architecture, agent workflow, model evaluation, UI screenshots, a 5-minute demo and the AI tools
used. Components are summarised in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

![Dashboard](docs/images/dashboard.png)

## How it meets the brief

| The brief asks for | What this project does |
|---|---|
| ML tool: LightGBM or Prophet on M5 / Walmart | Global **LightGBM** model on **Kaggle M5** (5 stores × 3 departments, 2011–2016) |
| Detect sudden stockout risks | Forecast-driven risk check: stock + inbound vs demand until the next delivery, plus safety stock |
| Detect demand spikes | Last 7 days and the next 7 forecast days vs a 90-day baseline |
| Query external trend data | Live **Google Trends** (pytrends), cached, with an honest "unavailable" fallback |
| Query promotional schedules | SNAP benefit days and retail events from the M5 calendar, 56 days ahead |
| Recommend automated re-orders | Expedite quantities and **draft purchase orders** with a human approve/reject step |
| Supplier allocations | Urgent units go to the fastest supplier, the rest to the best cost/reliability supplier |
| Report on architecture and workflow | [docs/REPORT.pdf](docs/REPORT.pdf) |
| Bonus: UI/UX to run the agent | Next.js dashboard, agent chat with a step-by-step reasoning trace, PO approval |

## Results

* **Forecast accuracy:** in a 3-window, 28-day chronological backtest, LightGBM reaches
  **9.5% WAPE**, against 13.4% for a seasonal-naive baseline and 32.4% for naive, and it wins on
  12 of 15 series. The model never sees future sales during evaluation.
* **Agent on the latest data:** a scan flags 2 CRITICAL and 6 WARNING series. One
  "Plan replenishment" request makes 22 tool calls and drafts supplier-split POs. Approving
  them clears the critical risks.

![Agent running the replenishment workflow](docs/images/agent.png)

## Quickstart

**Prerequisites:** Python 3.10–3.12 and Node.js 20+.

> **Tested on:** Windows 11, Python 3.12.4, Node.js 24, with pandas 2.1, numpy 1.26,
> scikit-learn 1.5, LightGBM 4.6 and FastAPI 0.139. It was verified from a fresh clone of this
> repo: all tests pass, the backend starts, and the frontend installs and builds.
>
> **macOS:** LightGBM needs OpenMP. Run `brew install libomp` once before `pip install`.
> **Linux:** if `import lightgbm` fails, install `libgomp1` (Debian/Ubuntu: `sudo apt install libgomp1`).
> A virtual environment (`python -m venv .venv`) is recommended.

```bash
pip install -r requirements.txt
python -m uvicorn src.api:app --port 8000
```

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000**. On the first start the backend trains the LightGBM model
automatically, in about 30 seconds. To retrain and re-run the backtest yourself, run
`python scripts/train.py`.

**Optional, an LLM planner:** copy `.env.example` to `.env` and add a free
[Groq](https://console.groq.com) key (`GROQ_API_KEY=...`), then restart the backend. Without a
key, an offline planner runs the same tools, so the demo works anywhere.

**Docker (backend only):** `docker build -t demand-agent .` then `docker run -p 8000:8000 demand-agent`.

### Try this
1. **Dashboard:** see which series run out before their next delivery.
2. **Agent:** click **Plan replenishment** and open the reasoning trace.
3. **Reorders:** approve the drafted POs, then watch the at-risk count drop on the Dashboard.
4. **Forecast / Report:** move the horizon slider; compare the model with the baselines.

## How it works

```
Next.js UI ──REST──▶ FastAPI ──▶ Agent (LLM tool calling or offline planner)
                                   │
                                   ▼  9 tools
        forecast_demand · detect_stockout_risk · detect_demand_spike · get_promo_calendar
        query_trends · recommend_reorder · draft_purchase_order · inventory_snapshot · run_risk_scan
                                   │
             LightGBM forecaster · risk + reorder + supplier allocation · Google Trends · SQLite
```

Main API endpoints: `POST /agent`, `GET /forecast`, `GET /dashboard/risk`, `POST /risk-scan`,
`GET /reorder`, `POST /reorder` (draft PO), `POST /pos/{id}/approve|reject`, `GET /evaluation`,
`GET /report`. The interactive API docs are at http://localhost:8000/docs.

## Data
- **Source:** Kaggle *M5 Forecasting – Accuracy*. The processed store/department table
  (`data/supply_chain_sales.csv`) and the future M5 calendar (`data/m5_calendar_future.csv`) are
  committed, so the app runs without the raw download.
- **Rebuilding from raw data:** put the Kaggle M5 `sales_train_validation.csv` and `calendar.csv`
  in `data/raw/m5/`, then run `python scripts/train.py`.
- **Simulated parts:** M5 has no inventory or supplier data. On-hand and inbound stock come from
  replaying a static weekly ordering policy over the real sales. Supplier lead times, costs and
  reliability are documented assumptions.

## Tests
```bash
pytest -q
```
49 tests cover the risk boundaries, supplier allocation, PO idempotency and approval, agent
intent routing, horizon parsing, the end-to-end replenishment workflow, a mocked LLM tool loop,
backtest leakage, the trend fallback and data safety. Each test uses its own temporary SQLite
database.

## Project layout
```
src/            data, model, eval, inventory, trends, tools, agent, db, api, report
frontend/       Next.js dashboard
scripts/        train.py (model + backtest), plot_accuracy.py, build_report_pdf.py
data/           processed M5 table, future calendar
reports/        backtest metrics and predictions
docs/           REPORT.pdf / REPORT.md, ARCHITECTURE.md, images/
tests/          pytest suite
```

## Limitations
- Inventory and supplier data are simulated or assumed (M5 has none). A real deployment would
  read on-hand and inbound stock from a WMS/ERP; the risk engine only needs those inputs.
- Forecasts are at store/department level, not item level.
- Google Trends is a weak, US-wide signal, used as context only. It is sometimes rate-limited.

## AI tools leveraged
OpenCode (Claude) and Claude Code (Claude Opus) for development, review and fixes. LightGBM,
pandas, pytrends, FastAPI and Next.js for the system itself. Groq, OpenAI or Ollama LLMs serve
as the optional agent planner. Details are in section 10 of the report.
