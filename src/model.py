"""Demand model: scikit-learn HistGradientBoostingRegressor (global) + seasonal-naive baseline."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import MODEL_DIR, STORES, DEPARTMENTS, RANDOM_SEED
from .data import load_or_generate


FEATURES = ["store", "dept", "dow", "month", "promo",
            "lag_1", "lag_7", "lag_28", "roll_7", "roll_28"]


def _featurize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.sort_values(["Store", "Dept", "Date"])
    g = out.groupby(["Store", "Dept"])["Sales"]
    out["lag_1"] = g.shift(1)
    out["lag_7"] = g.shift(7)
    out["lag_28"] = g.shift(28)
    out["roll_7"] = g.transform(lambda s: s.shift(1).rolling(7).mean())
    out["roll_28"] = g.transform(lambda s: s.shift(1).rolling(28).mean())
    out["dow"] = out["Date"].dt.dayofweek
    out["month"] = out["Date"].dt.month
    out["promo"] = out.get("Promo", 0).fillna(0).astype(int)
    out["store"] = pd.Categorical(out["Store"], categories=STORES).codes
    out["dept"] = pd.Categorical(out["Dept"], categories=DEPARTMENTS).codes
    return out


def train_model(df: pd.DataFrame = None, holdout_days: int = 28):
    if df is None:
        df = load_or_generate()
    feat = _featurize(df).dropna(subset=["lag_28", "roll_28"])
    cutoff = feat["Date"].max() - pd.Timedelta(days=holdout_days)
    tr = feat[feat["Date"] <= cutoff]
    te = feat[feat["Date"] > cutoff]
    Xtr, ytr = tr[FEATURES].astype(float).values, tr["Sales"].astype(float).values
    Xte, yte = te[FEATURES].astype(float).values, te["Sales"].astype(float).values

    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=RANDOM_SEED)
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)
    # seasonal-naive baseline (lag_7) where available
    base_pred = te["lag_7"].values
    mask = ~np.isnan(base_pred)
    baseline_mae = float(np.mean(np.abs(te["Sales"].values[mask] - base_pred[mask]))) if mask.any() else float("nan")

    bundle = {
        "model": model,
        "features": FEATURES,
        "metrics": {
            "MAE": round(float(mean_absolute_error(yte, preds)), 2),
            "RMSE": round(float(np.sqrt(mean_squared_error(yte, preds))), 2),
            "MAPE": round(float(np.mean(np.abs((yte - preds) / np.clip(yte, 1, None))) * 100), 2),
            "Baseline_MAE_lag7": round(baseline_mae, 2),
        },
    }
    joblib.dump(bundle, MODEL_DIR / "gb_global.joblib")
    (MODEL_DIR / "gb_metrics.json").write_text(json.dumps(bundle["metrics"], indent=2))
    return bundle


def forecast_demand(store: str, dept: str, horizon: int = 28) -> pd.DataFrame:
    bundle = joblib.load(MODEL_DIR / "gb_global.joblib")
    model = bundle["model"]
    base_df = load_or_generate()
    sub = base_df[(base_df["Store"] == store) & (base_df["Dept"] == dept)].sort_values("Date").copy()
    last_date = sub["Date"].max()
    fut_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    hist = sub.copy()
    preds = []
    for d in fut_dates:
        new_row = {"Store": store, "Dept": dept, "Date": d, "Sales": np.nan,
                   "OnHand": hist["OnHand"].iloc[-1], "Promo": 0,
                   "Holiday": int((d.month, d.day) in [(11, 24), (12, 25), (7, 4), (1, 1), (10, 31)])}
        hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
        feat = _featurize(hist)
        last = feat.iloc[-1:]
        x = last[FEATURES].astype(float).values
        p = float(model.predict(x)[0])
        p = max(0.0, p)
        preds.append(p)
        hist.loc[hist.index[-1], "Sales"] = p
    out = pd.DataFrame({"Date": fut_dates, "yhat": preds})
    out["yhat_lower"] = (out["yhat"] * 0.82).clip(lower=0)
    out["yhat_upper"] = out["yhat"] * 1.18
    return out


# aliases / back-compat
train_all = train_model
train_lightgbm = train_model
predict_lightgbm = forecast_demand


def models_ready() -> bool:
    return (MODEL_DIR / "gb_global.joblib").exists()


def get_metrics():
    p = MODEL_DIR / "gb_metrics.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}
