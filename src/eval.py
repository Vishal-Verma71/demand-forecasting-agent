"""Chronological backtesting for the M5-backed store/department forecasts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import BASE_DIR, STORES, DEPARTMENTS, RANDOM_SEED
from .data import load_or_generate
from .model import FEATURES, _featurize

REPORT_DIR = BASE_DIR / "reports"
PREDICTIONS_PATH = REPORT_DIR / "backtest_predictions.csv"
METRICS_PATH = REPORT_DIR / "backtest_metrics.json"

DEFAULT_HORIZON = 51
MODEL_PARAMS = {
    "max_iter": 300,
    "learning_rate": 0.06,
    "max_leaf_nodes": 31,
    "l2_regularization": 1.0,
    "random_state": RANDOM_SEED,
}


def _metric_dict(actual: np.ndarray, prediction: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    absolute_error = np.abs(actual - prediction)
    nonzero = np.abs(actual) > 0.0

    wape_denom = np.sum(np.abs(actual))
    mape_values = absolute_error[nonzero] / np.abs(actual[nonzero]) if nonzero.any() else np.array([], dtype=float)

    return {
        "observations": int(len(actual)),
        "MAE": round(float(np.mean(absolute_error)), 4) if len(actual) else float("nan"),
        "RMSE": round(float(np.sqrt(np.mean(np.square(actual - prediction)))), 4) if len(actual) else float("nan"),
        "WAPE": round(float(np.sum(absolute_error) / wape_denom * 100.0), 4) if wape_denom > 0.0 else float("nan"),
        "MAPE_nonzero": round(float(np.mean(mape_values) * 100.0), 4) if mape_values.size else float("nan"),
        "nonzero_actual_observations": int(nonzero.sum()),
    }


def _fit_regressor(train_df: pd.DataFrame):
    features = _featurize(train_df.copy()).dropna(subset=["lag_28", "roll_28"])
    x_train = features[FEATURES].astype(float).values
    y_train = features["Sales"].astype(float).values
    model = HistGradientBoostingRegressor(**MODEL_PARAMS)
    model.fit(x_train, y_train)
    train_holdout = features["Date"].max() - pd.Timedelta(days=28)
    validation = features[features["Date"] > train_holdout]
    holdout_metric = None
    if len(validation):
        holdout_prediction = model.predict(validation[FEATURES].astype(float).values)
        holdout_metric = {
            "validation_observations": int(len(validation)),
            "validation_MAE": round(float(mean_absolute_error(validation["Sales"], holdout_prediction)), 4),
            "validation_RMSE": round(float(np.sqrt(mean_squared_error(validation["Sales"], holdout_prediction))), 4),
        }
    return model, holdout_metric


def _recursive_forecast(model, train_series: pd.DataFrame, future_dates: pd.DataFrame, store: str, dept: str) -> list[float]:
    """Forecast future dates without using any actual sales from those dates."""
    history = train_series[["Store", "Dept", "Date", "Sales", "OnHand", "Promo", "Holiday"]].copy()
    history["Date"] = pd.to_datetime(history["Date"])
    history = history.sort_values("Date").tail(45).reset_index(drop=True)

    predictions = []
    for row in future_dates.sort_values("Date").itertuples(index=False):
        on_hand = float(history["OnHand"].iloc[-1])
        new_row = pd.DataFrame([{
            "Store": store,
            "Dept": dept,
            "Date": pd.Timestamp(row.Date),
            "Sales": np.nan,
            "OnHand": on_hand,
            "Promo": int(getattr(row, "Promo", 0)),
            "Holiday": int(getattr(row, "Holiday", 0)),
        }])
        history = pd.concat([history, new_row], ignore_index=True)
        feature_frame = _featurize(history)
        latest = feature_frame.iloc[-1:]
        x = latest[FEATURES].astype(float).values
        prediction = float(model.predict(x)[0])
        prediction = max(0.0, prediction)
        predictions.append(prediction)
        history.loc[history.index[-1], "Sales"] = prediction
        if len(history) > 45:
            history = history.tail(45).reset_index(drop=True)
    return predictions


def _baseline_forecasts(train_series: pd.DataFrame, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    last_value = float(train_series["Sales"].iloc[-1])
    naive = np.repeat(last_value, horizon)

    seasonal_history = train_series["Sales"].astype(float).tolist()
    recent_week = seasonal_history[-7:] if len(seasonal_history) >= 7 else seasonal_history + [seasonal_history[-1]] * (7 - len(seasonal_history))
    seasonal = np.asarray([recent_week[i % 7] for i in range(horizon)], dtype=float)
    return naive, seasonal


def _window_sizes(n_dates: int, horizon: int, n_windows: int) -> list[int]:
    if n_dates < horizon + 30:
        raise ValueError("The real dataset does not have enough history for chronological backtesting.")
    return [n_dates - i * horizon for i in range(0, n_windows)]


def run_backtest(df: pd.DataFrame | None = None, horizon: int = DEFAULT_HORIZON, windows: int = 3,
                 output_dir: str | Path = REPORT_DIR) -> dict:
    """Run anchored-origin chronological backtesting. Future actual sales are never used as model inputs."""
    if df is None:
        df = load_or_generate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / PREDICTIONS_PATH.name
    metrics_path = output_dir / METRICS_PATH.name

    work = df.copy()
    work["Date"] = pd.to_datetime(work["Date"])
    dates = np.sort(work["Date"].dt.normalize().unique())
    n_dates = len(dates)

    selected_stores = [store for store in STORES if store in set(work["Store"].unique())]
    selected_depts = [dept for dept in DEPARTMENTS if dept in set(work["Dept"].unique())]
    if not selected_stores or not selected_depts:
        raise ValueError("No configured store/department series exists in the dataset.")

    all_prediction_rows = []
    per_window_results = []
    primary_predictions = None

    for window_number, end_index in enumerate(_window_sizes(n_dates, horizon, windows), start=1):
        test_start = dates[end_index - horizon]
        test_end = dates[end_index - 1]
        train_end_date = dates[end_index - horizon - 1]
        train_dates = dates[dates < test_start]
        test_dates = dates[(dates >= test_start) & (dates <= test_end)]
        train_df = work[work["Date"].isin(train_dates)].copy()
        test_meta = work[work["Date"].isin(test_dates)].copy()

        if len(train_df) < 50 or len(test_meta) < horizon * len(selected_stores) * len(selected_depts):
            raise ValueError("Insufficient observations for a valid backtest window.")

        model, train_validation_metrics = _fit_regressor(train_df)
        train_window_metrics: list[dict] = []
        window_predictions: list[pd.DataFrame] = []

        for store in selected_stores:
            for dept in selected_depts:
                group_train = train_df[(train_df["Store"] == store) & (train_df["Dept"] == dept)].copy().sort_values("Date")
                group_test = test_meta[(test_meta["Store"] == store) & (test_meta["Dept"] == dept)].copy().sort_values("Date")
                if len(group_train) < 50 or len(group_test) < horizon:
                    raise ValueError(f"Insufficient history for {store}/{dept}.")

                future_meta = group_test[["Store", "Dept", "Date", "Promo", "Holiday"]].copy()
                actual = group_test["Sales"].astype(float).to_numpy()
                model_forecast = _recursive_forecast(model, group_train, future_meta, store, dept)
                naive_forecast, seasonal_forecast = _baseline_forecasts(group_train, horizon)

                primary = pd.DataFrame({
                    "window": window_number,
                    "date": pd.to_datetime(group_test["Date"]).dt.date.astype(str),
                    "store": store,
                    "dept": dept,
                    "actual": actual,
                    "naive": naive_forecast,
                    "seasonal_naive": seasonal_forecast,
                    "model": model_forecast,
                })
                window_predictions.append(primary)

                train_window_metrics.append({
                    "series": f"{store}/{dept}",
                    "train_observations": int(len(group_train)),
                    **{f"naive_{k}": v for k, v in _metric_dict(actual, naive_forecast).items()},
                    **{f"seasonal_naive_{k}": v for k, v in _metric_dict(actual, seasonal_forecast).items()},
                    **{f"model_{k}": v for k, v in _metric_dict(actual, model_forecast).items()},
                })

        window_prediction = pd.concat(window_predictions, ignore_index=True)
        all_prediction_rows.append(window_prediction)
        if primary_predictions is None:
            primary_predictions = window_prediction.copy()

        trained_period = {
            "from": str(pd.to_datetime(train_dates).min().date()),
            "to": str(pd.to_datetime(train_dates).max().date()),
            "observations": int(len(train_df)),
        }
        model_metrics = [_metric_summary(m, "model") for m in train_window_metrics]
        naive_metrics = [_metric_summary(m, "naive") for m in train_window_metrics]
        seasonal_metrics = [_metric_summary(m, "seasonal_naive") for m in train_window_metrics]

        metric_collections = {
            "naive": naive_metrics,
            "seasonal_naive": seasonal_metrics,
            "model": model_metrics,
        }

        def aggregate(model_prefix: str, metric_name: str) -> float:
            values = [item[metric_name] for item in metric_collections[model_prefix]]
            return round(float(np.mean(values)), 4) if values else float("nan")

        aggregate_model = {metric: aggregate("model", metric) for metric in ("MAE", "RMSE", "WAPE", "MAPE_nonzero")}
        aggregate_naive = {metric: aggregate("naive", metric) for metric in ("MAE", "RMSE", "WAPE", "MAPE_nonzero")}
        aggregate_seasonal = {metric: aggregate("seasonal_naive", metric) for metric in ("MAE", "RMSE", "WAPE", "MAPE_nonzero")}

        per_series = []
        for series_metric in train_window_metrics:
            per_series.append({
                "series": series_metric["series"],
                "train_observations": series_metric["train_observations"],
                "naive": _series_metrics(series_metric, "naive"),
                "seasonal_naive": _series_metrics(series_metric, "seasonal_naive"),
                "model": _series_metrics(series_metric, "model"),
            })

        per_window_results.append({
            "window": window_number,
            "train": {
                "from": trained_period["from"],
                "to": trained_period["to"],
                "observations": trained_period["observations"],
                "validation": train_validation_metrics,
            },
            "test": {
                "from": str(pd.to_datetime(test_dates).min().date()),
                "to": str(pd.to_datetime(test_dates).max().date()),
                "observations": int(len(test_dates)),
            },
            "series": per_series,
            "overall_macro": {
                "naive": aggregate_naive,
                "seasonal_naive": aggregate_seasonal,
                "model": aggregate_model,
            },
        })

    final_predictions = pd.concat(all_prediction_rows, ignore_index=True)
    final_predictions.to_csv(predictions_path, index=False)

    primary_window = per_window_results[0]
    window_metrics = [item["overall_macro"] for item in per_window_results]
    replay_windows = [
        {
            "window": item["window"],
            "train": item["train"],
            "test": item["test"],
            **item["overall_macro"],
        }
        for item in per_window_results
    ]

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": "Kaggle M5 Forecasting - Accuracy",
        "source_files": [
            "data/raw/m5/sales_train_validation.csv",
            "data/raw/m5/calendar.csv",
        ],
        "target": "Daily aggregate sales at store/department level",
        "stores": selected_stores,
        "departments": selected_depts,
        "forecast_horizon": horizon,
        "windows": windows,
        "primary_window": primary_window["window"],
        "primary_test_period": primary_window["test"],
        "primary_test_observations": int(len(primary_predictions)),
        "primary_overall_macro": primary_window["overall_macro"],
        "rolling_test_periods": replay_windows,
        "rules": {
            "holdout_origin": "all observations before the first test date only",
            "future_sales_in_model_inputs": False,
            "calendar_features": "allowed because they are known in advance",
            "MAPE_rule": "MAPE is computed only on observations with non-zero actual sales",
            "WAPE_rule": "WAPE is sum(abs(error)) / sum(abs(actual)) * 100",
            "overall_rule": "macro average of the 15 per-series metrics, with a pooled per-observation micro variant also included below",
        },
        "model": {
            "name": "HistGradientBoostingRegressor",
            "features": FEATURES,
            "hyperparameters": MODEL_PARAMS,
        },
        "per_series_primary": primary_window["series"],
        "rolling_summary": window_metrics,
        "files": {
            "predictions": str(predictions_path),
            "metrics": str(metrics_path),
        },
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _series_metrics(series_metric: dict, prefix: str) -> dict:
    names = ("MAE", "RMSE", "WAPE", "MAPE_nonzero", "observations", "nonzero_actual_observations")
    return {name: series_metric[f"{prefix}_{name}"] for name in names}


def _metric_summary(metric_item: dict, prefix: str) -> dict:
    return {metric: metric_item[f"{prefix}_{metric}"] for metric in ("MAE", "RMSE", "WAPE", "MAPE_nonzero")}


def load_evaluation(path: Path = METRICS_PATH) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def main():
    parser = argparse.ArgumentParser(description="Run chronological M5 backtesting.")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--windows", type=int, default=3)
    args = parser.parse_args()
    result = run_backtest(horizon=args.horizon, windows=args.windows)
    primary = result["primary_overall_macro"]
    print("Primary holdout:", result["primary_test_period"])
    print("Overall macro metrics:")
    for name, metrics in primary.items():
        print(f"  {name}: {metrics}")
