"""Feature engineering for demand forecasting."""
from __future__ import annotations

import pandas as pd
import numpy as np


def build_series(df: pd.DataFrame, store: str, dept: str) -> pd.DataFrame:
    sub = df[(df["Store"] == store) & (df["Dept"] == dept)].copy()
    sub = sub.sort_values("Date").reset_index(drop=True)
    sub["Sales"] = sub["Sales"].astype(float)
    sub["dow"] = sub["Date"].dt.dayofweek
    sub["month"] = sub["Date"].dt.month
    sub["Promo"] = sub.get("Promo", 0)
    # rolling
    sub["lag_7"] = sub["Sales"].shift(7)
    sub["lag_28"] = sub["Sales"].shift(28)
    sub["roll_7"] = sub["Sales"].rolling(7).mean()
    sub["roll_28"] = sub["Sales"].rolling(28).mean()
    return sub


def train_test_split_time(df: pd.DataFrame, holdout_days: int = 28):
    cutoff = df["Date"].max() - pd.Timedelta(days=holdout_days)
    train = df[df["Date"] <= cutoff].copy()
    test = df[df["Date"] > cutoff].copy()
    return train, test
