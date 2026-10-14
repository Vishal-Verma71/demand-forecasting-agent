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


def generate_dataset(days: int = 730, seed: int = RANDOM_SEED, end_date: date | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if end_date is None:
        end_date = date.today()
    start = end_date - timedelta(days=days - 1)
    rows = []
    # base level per store/dept
    base = {"FOODS": 1200, "HOBBIES": 350, "HOUSEHOLD": 600}
    store_factor = {"CA_1": 1.3, "CA_2": 1.1, "TX_1": 1.2, "TX_2": 0.95, "WI_1": 0.85}
    for store in STORES:
        for dept in DEPARTMENTS:
            level = base[dept] * store_factor[store]
            trend = rng.uniform(-0.0005, 0.0012)
            # One stable supplier/lead-time per store-department series.
            supplier, lead = SUPPLIERS[dept][int(rng.integers(0, 2))]
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
                rows.append({
                    "Store": store,
                    "Dept": dept,
                    "Date": d.isoformat(),
                    "Sales": round(sales, 2),
                    "OnHand": round(inv, 2),
                    "Supplier": supplier,
                    "LeadTime": int(lead),
                    "Promo": int(promo > 1.1),
                    "Holiday": int((month, d.day) in HOLIDAYS),
                })
    df = pd.DataFrame(rows)
    return df


REQUIRED_COLUMNS = ["Store", "Dept", "Date", "Sales"]


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    out = df.copy()
    if "OnHand" not in out.columns:
        out["OnHand"] = out["Sales"].astype(float) * 7.0
    if "Supplier" not in out.columns:
        out["Supplier"] = "UnknownSupplier"
    if "LeadTime" not in out.columns:
        out["LeadTime"] = 7
    if "Promo" not in out.columns:
        out["Promo"] = 0
    if "Holiday" not in out.columns:
        out["Holiday"] = 0
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Store", "Dept", "Date", "Sales"]).copy()
    out["Sales"] = out["Sales"].astype(float).clip(lower=0)
    out["OnHand"] = out["OnHand"].astype(float).clip(lower=0)
    out["LeadTime"] = out["LeadTime"].astype(int).clip(lower=1)
    out["Supplier"] = out["Supplier"].astype(str)
    out["Promo"] = out["Promo"].astype(int)
    out["Holiday"] = out["Holiday"].astype(int)
    return out.reset_index(drop=True)


def load_or_generate(path: Path = DATA_PATH, force: bool = False) -> pd.DataFrame:
    if path.exists() and not force:
        df = pd.read_csv(path)
        if set(REQUIRED_COLUMNS).issubset(df.columns):
            df = normalize_dataset(df)
            # Regenerate when the cached data is stale or incomplete.
            if not df.empty and df["Date"].max() >= pd.Timestamp(date.today() - timedelta(days=30)):
                return df
        force = True
    if path.exists() and force:
        try:
            path.unlink()
        except OSError:
            pass
    df = normalize_dataset(generate_dataset())
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def list_series(df: pd.DataFrame):
    return sorted(df[["Store", "Dept"]].drop_duplicates().itertuples(index=False, name=None))
