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

## Quickstart
```bash
pip install -r requirements.txt
cp .env.example .env        # optional: add GROQ_API_KEY or OPENAI_API_KEY
python scripts/train.py    # one-time: trains + caches models (~2-5 min)
streamlit run app.py
```
Open http://localhost:8501

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

## Notes
- Designed to run on 8 GB RAM: models trained once and cached; app loads from cache.
- No API keys required; offline deterministic router keeps the demo functional anywhere.
