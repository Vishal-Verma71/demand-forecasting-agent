# Architecture

## Agentic workflow
```mermaid
flowchart TD
  U[User query] --> A[Agent (ReAct)]
  A -->|tool call| T1[forecast_demand]
  A -->|tool call| T2[detect_stockout_risk]
  A -->|tool call| T3[query_trends]
  A -->|tool call| T4[recommend_reorder]
  A -->|tool call| T5[inventory_snapshot / run_risk_scan]
  T1 --> P[Prophet per-series model]
  T1 --> L[LightGBM global model]
  T2 --> I[Inventory risk engine]
  T4 --> I
  T3 --> G[pytrends / fallback]
  I --> DB[(SQLite)]
  A --> UI[Streamlit UI + report]
```

## Data flow
1. `data.generate_dataset()` produces store-level daily sales (or loads cached CSV).
2. `features.build_series()` adds lags/rolling features.
3. `model.train_model()` fits + caches one global gradient-boosting model (+ metrics).
4. App loads cached models; trains on demand if missing.
5. Agent routes queries to tools; tools read models/DB.
6. `report.build_report()` exports Markdown.

## Key decisions
- HistGradientBoosting (scikit-learn) global model for speed & robustness; seasonal-naive lag-7 baseline for comparison.
- Time-based holdout (last 28 days) for honest metrics.
- Offline deterministic router guarantees demo runs without keys.
- SQLite over a server DB for zero-ops deployment.
