"""Demand model: LightGBM global regressor over all store/department series.

One model is trained on every series at once (store and department are features),
and future days are forecast recursively: each prediction is fed back as the lag
input for the next day.  Calendar inputs (SNAP promotion days, M5 events) are
known in advance and come from the M5 calendar.
"""
from __future__ import annotations

import json
import threading
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import BASE_DIR, MODEL_DIR, STORES, DEPARTMENTS, RANDOM_SEED
from .data import load_or_generate, future_calendar

MODEL_NAME = "LightGBM (LGBMRegressor, global model)"
MODEL_PATH = MODEL_DIR / "lgbm_global.joblib"
METRICS_PATH = MODEL_DIR / "lgbm_metrics.json"
BACKTEST_PREDICTIONS_PATH = BASE_DIR / "reports" / "backtest_predictions.csv"

FEATURES = ["store", "dept", "dow", "month", "promo", "holiday",
            "lag_1", "lag_7", "lag_28", "roll_7", "roll_28"]

MODEL_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "verbose": -1,
}

# Fallback band when no backtest residuals are available yet.
_DEFAULT_BAND = (0.82, 1.18)


def make_regressor() -> LGBMRegressor:
    return LGBMRegressor(**MODEL_PARAMS)


_TRAIN_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _load_bundle():
    # A fresh clone has no trained model (models/ is gitignored), so train once on first use.
    with _TRAIN_LOCK:
        if not MODEL_PATH.exists():
            train_model()
    return joblib.load(MODEL_PATH)


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
    out["promo"] = out["Promo"].fillna(0).astype(int) if "Promo" in out else 0
    out["holiday"] = out["Holiday"].fillna(0).astype(int) if "Holiday" in out else 0
    out["store"] = pd.Categorical(out["Store"], categories=STORES).codes
    out["dept"] = pd.Categorical(out["Dept"], categories=DEPARTMENTS).codes
    return out


def recursive_forecast(model, history: pd.DataFrame, future: pd.DataFrame, store: str, dept: str) -> list[float]:
    """Forecast ``future`` dates one day at a time without using any future actuals.

    ``future`` must hold Date, Promo and Holiday (calendar inputs known in advance).
    Shared by the live forecaster and the backtest so both use identical logic.
    """
    hist = history[["Store", "Dept", "Date", "Sales", "Promo", "Holiday"]].copy()
    hist["Date"] = pd.to_datetime(hist["Date"])
    hist = hist.sort_values("Date").tail(45).reset_index(drop=True)
    preds = []
    for row in future.sort_values("Date").itertuples(index=False):
        new_row = {"Store": store, "Dept": dept, "Date": pd.Timestamp(row.Date), "Sales": np.nan,
                   "Promo": int(getattr(row, "Promo", 0)), "Holiday": int(getattr(row, "Holiday", 0))}
        hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
        x = _featurize(hist).iloc[-1:][FEATURES].astype(float)
        # One row per call: a single thread is faster and avoids OpenMP contention.
        p = max(0.0, float(model.predict(x, num_threads=1)[0]))
        preds.append(p)
        hist.loc[hist.index[-1], "Sales"] = p
        hist = hist.tail(45).reset_index(drop=True)
    return preds


def train_model(df: pd.DataFrame = None, holdout_days: int = 28):
    if df is None:
        df = load_or_generate()
    feat = _featurize(df).dropna(subset=["lag_28", "roll_28"])
    cutoff = feat["Date"].max() - pd.Timedelta(days=holdout_days)
    tr = feat[feat["Date"] <= cutoff]
    te = feat[feat["Date"] > cutoff]

    model = make_regressor()
    model.fit(tr[FEATURES].astype(float), tr["Sales"].astype(float))
    preds = model.predict(te[FEATURES].astype(float))
    yte = te["Sales"].astype(float).values
    base_pred = te["lag_7"].values
    mask = ~np.isnan(base_pred)
    baseline_mae = float(np.mean(np.abs(yte[mask] - base_pred[mask]))) if mask.any() else float("nan")
    metrics = {
        "evaluation": f"one-step-ahead on the last {holdout_days} days (true lags); "
                      "see reports/backtest_metrics.json for the multi-step backtest",
        "MAE": round(float(mean_absolute_error(yte, preds)), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(yte, preds))), 2),
        "MAPE": round(float(np.mean(np.abs((yte - preds) / np.clip(yte, 1, None))) * 100), 2),
        "Baseline_MAE_lag7": round(baseline_mae, 2),
    }

    # Production model is refit on all history so the forecast starts from the latest data.
    final = make_regressor()
    final.fit(feat[FEATURES].astype(float), feat["Sales"].astype(float))
    bundle = {"model": final, "features": FEATURES, "name": MODEL_NAME, "metrics": metrics}
    joblib.dump(bundle, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    clear_model_cache()
    return bundle


_FORECAST_CACHE = {}
# API requests run in a thread pool; concurrent LightGBM predictions from several threads
# can hang, and would duplicate work anyway, so forecasts are generated one at a time.
_FORECAST_LOCK = threading.Lock()
MAX_FORECAST_HORIZON = 56   # M5 calendar covers exactly 56 days past the last sales day
_MAX_FORECAST_HORIZON = MAX_FORECAST_HORIZON


def _forecast_future(store: str, dept: str, horizon: int) -> tuple[list[float], pd.DatetimeIndex]:
    """Generate exactly ``horizon`` future predictions for one store/department series."""
    model = _load_bundle()["model"]
    base_df = load_or_generate()
    sub = base_df[(base_df["Store"] == store) & (base_df["Dept"] == dept)].sort_values("Date")
    if sub.empty:
        raise ValueError(f"No sales history for {store}/{dept}")
    start = sub["Date"].max() + pd.Timedelta(days=1)
    future = future_calendar(store, start, horizon)
    preds = recursive_forecast(model, sub, future, store, dept)
    return preds, pd.DatetimeIndex(future["Date"])


@lru_cache(maxsize=1)
def _interval_ratios() -> dict:
    """80% empirical prediction-interval multipliers per series from backtest residuals.

    For every backtested day we take actual / forecast; the 10th and 90th percentiles
    of that ratio give the band.  Honest because it reflects real multi-step errors.
    """
    if not BACKTEST_PREDICTIONS_PATH.exists():
        return {}
    bt = pd.read_csv(BACKTEST_PREDICTIONS_PATH)
    bt = bt[bt["model"] > 0]
    bt["ratio"] = bt["actual"] / bt["model"]
    q = bt.groupby(["store", "dept"])["ratio"].quantile([0.1, 0.9]).unstack()
    return {(s, d): (float(r[0.1]), float(r[0.9])) for (s, d), r in q.iterrows()}


def forecast_demand(store: str, dept: str, horizon: int = 28) -> pd.DataFrame:
    # One recursive pass to the maximum horizon serves every shorter horizon by prefix,
    # so changing the horizon slider does not re-run the forecast.
    if store not in STORES or dept not in DEPARTMENTS:
        raise ValueError(f"Unknown store/department {store}/{dept}")
    if not 1 <= int(horizon) <= MAX_FORECAST_HORIZON:
        raise ValueError(f"horizon must be between 1 and {MAX_FORECAST_HORIZON} days")
    target_horizon = max(horizon, MAX_FORECAST_HORIZON)
    cache_key = (store, dept)
    with _FORECAST_LOCK:
        cached_horizon, cached_predictions, cached_dates = _FORECAST_CACHE.get(cache_key, (0, [], pd.DatetimeIndex([])))
        if cached_horizon < target_horizon:
            cached_predictions, cached_dates = _forecast_future(store, dept, target_horizon)
            _FORECAST_CACHE[cache_key] = (target_horizon, cached_predictions, cached_dates)

    out = pd.DataFrame({"Date": cached_dates[:horizon], "yhat": cached_predictions[:horizon]})
    lo, hi = _interval_ratios().get((store, dept), _DEFAULT_BAND)
    out["yhat_lower"] = (out["yhat"] * min(lo, 1.0)).clip(lower=0)
    out["yhat_upper"] = out["yhat"] * max(hi, 1.0)
    return out


def interval_method(store: str, dept: str) -> str:
    if (store, dept) in _interval_ratios():
        return "80% empirical interval from backtest residuals"
    return "fixed ±18% band (run the backtest for empirical intervals)"


train_all = train_model


def models_ready() -> bool:
    return MODEL_PATH.exists()


def get_metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return {}


def clear_model_cache():
    _load_bundle.cache_clear()
    _interval_ratios.cache_clear()
    _FORECAST_CACHE.clear()
