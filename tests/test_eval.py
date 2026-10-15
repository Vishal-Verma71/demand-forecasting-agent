import math

import numpy as np
import pandas as pd
import pytest

from src import eval


def make_series(days=60):
    rows = []
    for store in ("CA_1", "CA_2"):
        for dept in ("FOODS", "HOBBIES"):
            for i in range(days):
                rows.append({
                    "Store": store,
                    "Dept": dept,
                    "Date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=i),
                    "Sales": float((i % 7 + 1) * 10 + (1 if store == "CA_2" else 0)),
                    "OnHand": float((i % 7 + 1) * 140),
                    "Promo": int(i % 9 == 0),
                    "Holiday": int(i % 13 == 0),
                    "Supplier": "TestSupplier",
                    "LeadTime": 5,
                })
    return pd.DataFrame(rows)


def test_metric_dict():
    actual = np.array([10.0, 0.0, 20.0])
    prediction = np.array([12.0, 1.0, 16.0])
    metrics = eval._metric_dict(actual, prediction)
    assert metrics["observations"] == 3
    assert metrics["MAE"] == 2.3333
    assert metrics["nonzero_actual_observations"] == 2
    assert metrics["MAPE_nonzero"] == 20.0


def test_chronological_windows_do_not_overlap():
    ends = eval._window_sizes(120, 30, 3)
    assert ends == [120, 90, 60]
    windows = [(end - 30, end) for end in ends]
    assert [window[0] for window in windows] == [90, 60, 30]
    assert all(right[1] <= left[0] for left, right in zip(windows, windows[1:]))


def test_run_backtest_uses_only_past_training_data(monkeypatch):
    data = make_series(70)
    captured = {}

    original_fit = eval._fit_regressor

    def fit_regressor_with_capture(train_df):
        captured["max_training_date"] = pd.Timestamp(train_df["Date"].max())
        return original_fit(train_df)

    monkeypatch.setattr(eval, "_fit_regressor", fit_regressor_with_capture)
    result = eval.run_backtest(data, horizon=7, windows=1, output_dir=("reports_test_tmp"))
    first_window = result["rolling_test_periods"][0]
    assert pd.Timestamp(first_window["test"]["from"]) > captured["max_training_date"]
    assert first_window["test"]["from"] == result["primary_test_period"]["from"]
    # Metric function and serialization stay finite.
    assert math.isfinite(result["primary_overall_macro"]["model"]["MAE"])
