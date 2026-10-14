"""Data loading + synthetic dataset generation (Walmart/M5-style store-level demand)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import date, timedelta

from .config import DATA_DIR, STORES, DEPARTMENTS, RANDOM_SEED

DATA_PATH = DATA_DIR / "supply_chain_sales.csv"

SUPPLIERS = {
    "FOODS": [("FreshCo", 3), ("GlobalFoods", 5)],
    "HOBBIES": [("ToyWorld", 7), ("CraftHub", 10)],
    "HOUSEHOLD": [("HomeEssentials", 5), ("CleanPro", 4)],
}

# US-ish holidays that lift demand
HOLIDAYS = {
    (11, 24), (11, 25), (11, 26), (12, 24), (12, 25), (12, 26),
    (7, 4), (2, 14), (10, 31), (1, 1),
}


def generate_dataset(days: int = 730, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = date(2023, 1, 1)
    rows = []
    # base level per store/dept
    base = {"FOODS": 1200, "HOBBIES": 350, "HOUSEHOLD": 600}
    store_factor = {"CA_1": 1.3, "CA_2": 1.1, "TX_1": 1.2, "TX_2": 0.95, "WI_1": 0.85}
    for store in STORES:
        for dept in DEPARTMENTS:
            level = base[dept] * store_factor[store]
            trend = rng.uniform(-0.0005, 0.0012)
            for i in range(days):
                d = start + timedelta(days=i)
                # seasonality
                dow = d.weekday()
                weekly = 1.0 + (0.25 if dow in (5, 6) else -0.1 if dow == 0 else 0.0)
                month = d.month
                seasonal = 1.0 + 0.15 * np.sin(2 * np.pi * (month - 3) / 12)
                holiday = 1.25 if (month, d.day) in HOLIDAYS else 1.0
                # promo occasionally
                promo = 1.0
                if rng.random() < 0.07:
                    promo = rng.uniform(1.2, 1.6)
                noise = rng.normal(0, 0.08)
                sales = level * (1 + trend * i) * weekly * seasonal * holiday * promo * (1 + noise)
                sales = max(0.0, float(sales))
                # inventory roughly 1.5x avg weekly demand
                inv = sales * rng.uniform(8, 16)
                sup, lead = SUPPLIERS[dept][rng.integers(0, 2)]
                rows.append({
                    "Store": store,
                    "Dept": dept,
                    "Date": d.isoformat(),
                    "Sales": round(sales, 2),
                    "OnHand": round(inv, 2),
                    "Supplier": sup,
                    "LeadTime": int(lead),
                    "Promo": int(promo > 1.1),
                    "Holiday": int((month, d.day) in HOLIDAYS),
                })
    df = pd.DataFrame(rows)
    return df


def load_or_generate(path: Path = DATA_PATH, force: bool = False) -> pd.DataFrame:
    if path.exists() and not force:
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"])
        return df
    df = generate_dataset()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def list_series(df: pd.DataFrame):
    return sorted(df[["Store", "Dept"]].drop_duplicates().itertuples(index=False, name=None))
