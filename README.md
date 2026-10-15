# Demand Forecasting Agent — Logistics Supply Chain Intelligence

Agentic workflow that forecasts store-level demand, detects stockout risk, pulls external trend
signals, and recommends (and drafts) replenishment orders. Built for the Latentsig review
(2–3 day coding assignment).

## Highlights
- **Real ML model as a tool:** scikit-learn HistGradientBoosting (global), compared against a 7-day seasonal-naive baseline, with holdout metrics (MAE/MAPE/RMSE).
- **Agentic loop:** ReAct agent with 6 tools and a full visible reasoning trace.
- **Business actions:** stockout alerts, reorder qty (EOQ-style), supplier selection, draft POs.
- **External signals:** Google Trends via pytrends (graceful offline fallback).
- **Works offline:** deterministic router when no LLM API key is configured; UI never breaks.
- **Polished UI:** Streamlit + Plotly, KPI cards, forecast bands, risk feed, chat agent, report export.

## Quickstart (Next.js frontend + FastAPI backend)
```bash
pip install -r requirements.txt
cp .env.example .env        # optional: add GROQ_API_KEY or OPENAI_API_KEY
python scripts/train.py    # one-time: trains + caches models (~2-5 min)

# Terminal 1 - backend
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000

# Terminal 2 - frontend
cd frontend
npm install
npm run dev
```
Open http://localhost:3000 (backend at http://localhost:8000)

(Legacy Streamlit UI remains available via `streamlit run app.py`.)

## Tests
```bash
pytest -q
```

## Project layout
```
app.py                 Streamlit UI
src/                   config, data, features, model, agent, tools, inventory, trends, db, report, llm
scripts/train.py       one-off training
data/                  generated supply_chain_sales.csv (auto)
models/                cached joblib models + metrics (gitignored)
docs/                  ARCHITECTURE.md, REPORT.md
tests/                 pytest
```

## AI tools leveraged
- OpenCode (Claude) for scaffolding, refactoring and code review
- scikit-learn (HistGradientBoosting) + joblib for forecasting
- pytrends for external demand signals
- Streamlit + Plotly for UI, OpenAI-compatible clients (Groq/OpenAI/Ollama) for the agent

## Data
- Default: synthetic but realistic Walmart-style store/department daily demand (auto-generated).
- Real dataset support: place the official M5 `sales_train_validation.csv` and
  `calendar.csv` files under `data/raw/m5/` (plus `sell_prices.csv` only when you
  want price joins).  The application aggregates M5 daily sales to the existing
  store/department/date schema and uses it as the primary source whenever those files
  exist.  Run `python scripts/train.py` after adding them.
- Legacy CSV support: place a normalized CSV at `data/supply_chain_sales.csv` with columns
  `Store, Dept, Date, Sales, OnHand, Supplier, LeadTime, Promo, Holiday` and rerun
  `python scripts/train.py`.  The M5 mapping lives in `src/data.py`.

## Notes
- Designed to run on 8 GB RAM: models trained once and cached; app loads from cache.
- No API keys required; offline deterministic router keeps the demo functional anywhere.
