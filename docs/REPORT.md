# Demand Forecasting Agent — Report

## 1. Business pain point
Logistics/retail teams lose money on stockouts and overstock. Manual spreadsheet forecasting is slow and reactive. This agent automates demand forecasting, risk detection, trend awareness, and replenishment recommendations.

## 2. Agentic workflow
- Detect: parse user intent (forecast / risk / reorder / trends / inventory / demand spike).
- Plan: LLM (Groq/OpenAI/Ollama) chooses tools; offline router handles the same intents deterministically.
- Act: call tools (`forecast_demand`, `detect_stockout_risk`, `query_trends`, `recommend_reorder`, `inventory_snapshot`, `run_risk_scan`).
- Observe: tool results (forecasts, risk levels, trends) feed back into the loop.
- Respond: concise business answer + draft POs + alerts + exportable report.

## 3. Model details
- HistGradientBoostingRegressor (scikit-learn): global gradient-boosted model over lag/rolling/holiday/promo features.
- Seasonal-naive baseline (lag 7) used as challenger metric.
- Evaluation: MAE, RMSE & MAPE on the last 28 days (time-based holdout).

## 4. UI/UX (bonus)
Streamlit dashboard with KPI cards, Plotly forecast bands, inventory risk map, alert feed, external trend widget,
agent chat with reasoning trace, reorder table with one-click PO drafts, and Markdown report download.

## 5. AI tools leveraged
OpenCode (Claude) for code generation/refactoring; Meta Prophet; LightGBM; scikit-learn; pytrends; Streamlit & Plotly.

## 6. Future work
- M5 item-level hierarchy, promotional calendars, supplier lead-time variability, automated email/ERP export.
