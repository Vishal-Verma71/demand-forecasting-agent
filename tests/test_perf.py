import pandas as pd

from src import model

EXPECTED_CACHE_HORIZON = model._MAX_FORECAST_HORIZON


def test_forecast_horizon_uses_one_cached_full_pass(monkeypatch):
    call_count = {"count": 0}

    def fake_forecast_future(store, dept, horizon):
        call_count["count"] += 1
        assert horizon == EXPECTED_CACHE_HORIZON
        return [float(i) for i in range(horizon)], pd.date_range("2030-01-01", periods=horizon, freq="D")

    model._FORECAST_CACHE.clear()
    monkeypatch.setattr(model, "_forecast_future", fake_forecast_future)

    first = model.forecast_demand("CA_1", "FOODS", 28)
    second = model.forecast_demand("CA_1", "FOODS", 51)

    assert len(first) == 28
    assert len(second) == 51
    assert call_count["count"] == 1
    assert float(first["yhat"].tolist()[-1]) == 27.0
    model._FORECAST_CACHE.clear()