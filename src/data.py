"""Data loading and real-data adaptation for the Demand Forecasting Agent.

The application is built around a store/department daily-sales table.  When the
official M5 files in ``data/raw/m5`` are present they become the primary source.
Otherwise we retain the original lightweight generator as an explicit offline
fallback.  Synthetic fields such as on-hand stock, supplier and lead time are
documented assumptions because the M5 competition does not contain those.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
from datetime import date, timedelta

from .config import DATA_DIR, STORES, DEPARTMENTS, RANDOM_SEED

DATA_PATH = DATA_DIR / "supply_chain_sales.csv"
M5_RAW_DIR = DATA_DIR / "raw" / "m5"
M5_REQUIRED_FILES = ("sales_train_validation.csv", "calendar.csv")

SUPPLIERS = {
    "FOODS": [("FreshCo", 3), ("GlobalFoods", 5)],
    "HOBBIES": [("ToyWorld", 7), ("CraftHub", 10)],
    "HOUSEHOLD": [("HomeEssentials", 5), ("CleanPro", 4)],
}

HOLIDAYS = {
    (11, 24), (11, 25), (11, 26), (12, 24), (12, 25), (12, 26),
    (7, 4), (2, 14), (10, 31), (1, 1),
}

REQUIRED_COLUMNS = ["Store", "Dept", "Date", "Sales"]


def generate_dataset(days: int = 730, seed: int = RANDOM_SEED, end_date: date | None = None) -> pd.DataFrame:
    """Existing lightweight generator retained for offline testing/fallback only."""
    rng = np.random.default_rng(seed)
    if end_date is None:
        end_date = date.today()
    start = end_date - timedelta(days=days - 1)
    rows = []
    base = {"FOODS": 1200, "HOBBIES": 350, "HOUSEHOLD": 600}
    store_factor = {"CA_1": 1.3, "CA_2": 1.1, "TX_1": 1.2, "TX_2": 0.95, "WI_1": 0.85}
    for store in STORES:
        for dept in DEPARTMENTS:
            level = base[dept] * store_factor[store]
            trend = rng.uniform(-0.0005, 0.0012)
            supplier, lead = SUPPLIERS[dept][int(rng.integers(0, 2))]
            for i in range(days):
                d = start + timedelta(days=i)
                dow = d.weekday()
                weekly = 1.0 + (0.25 if dow in (5, 6) else -0.1 if dow == 0 else 0.0)
                month = d.month
                seasonal = 1.0 + 0.15 * np.sin(2 * np.pi * (month - 3) / 12)
                holiday = 1.25 if (month, d.day) in HOLIDAYS else 1.0
                promo = 1.0
                if rng.random() < 0.07:
                    promo = rng.uniform(1.2, 1.6)
                noise = rng.normal(0, 0.08)
                sales = level * (1 + trend * i) * weekly * seasonal * holiday * promo * (1 + noise)
                sales = max(0.0, float(sales))
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
    return pd.DataFrame(rows)


def m5_files_available(raw_dir: str | Path = M5_RAW_DIR) -> bool:
    raw = Path(raw_dir)
    return all((raw / f).is_file() for f in M5_REQUIRED_FILES)


def _supplier_for(store: str, dept: str):
    choices = SUPPLIERS.get(dept.upper(), [("UnknownSupplier", 7)])
    token = sum((i + 1) * ord(c) for i, c in enumerate(str(store)))
    return choices[token % len(choices)]


def load_m5_sales(raw_dir: str | Path = M5_RAW_DIR,
                  stores: list[str] | None = None,
                  departments: list[str] | None = None) -> pd.DataFrame:
    """Load real M5 data into the app's store/department daily-sales schema.

    M5 contains item/store/day sales.  We aggregate item sales to the
    existing store/department/date grain, map day identifiers through the M5
    calendar, and derive the promotional and holiday fields from that calendar.
    Inventory fields are intentionally not invented by M5: they use clearly
    labelled existing assumptions.
    """
    if not m5_files_available(raw_dir):
        raise FileNotFoundError(f"M5 files not found in {raw_dir}")
    raw = Path(raw_dir)
    stores = stores or STORES
    departments = departments or DEPARTMENTS

    calendar = pd.read_csv(raw / "calendar.csv")
    required = {"date", "d"}
    if not required.issubset(calendar.columns):
        raise ValueError("M5 calendar.csv is missing required date/day columns")

    calendar = calendar.drop_duplicates("d").copy()
    calendar["Date"] = pd.to_datetime(calendar["date"], errors="coerce")
    calendar["Holiday"] = 0
    if "event_name_1" in calendar.columns:
        calendar["Holiday"] = calendar["event_name_1"].notna().astype(int)
    if "event_name_2" in calendar.columns:
        calendar["Holiday"] = (calendar["Holiday"]
                               | calendar["event_name_2"].notna().astype(int))

    sales_path = raw / "sales_train_validation.csv"
    with open(sales_path, "r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split(",")
    if "store_id" not in header or "dept_id" not in header:
        raise ValueError("M5 sales file is missing store_id/dept_id columns")
    day_columns = [c for c in header if c.startswith("d_")]
    if not day_columns:
        raise ValueError("M5 sales file contains no daily sales columns")

    dtype_map = {c: "int16" for c in day_columns}
    sales = pd.read_csv(sales_path, dtype=dtype_map)
    sales = sales[sales["store_id"].astype(str).isin(stores)].copy()
    sales["dept_core"] = sales["dept_id"].astype(str).str.split("_").str[0]
    sales = sales[sales["dept_core"].isin(departments)].copy()
    if sales.empty:
        raise ValueError("No selected store/department series in the M5 file")

    grouped = sales.groupby(["store_id", "dept_core"], as_index=False, observed=True)[day_columns].sum()
    del sales
    gc.collect()

    long = grouped.melt(
        id_vars=["store_id", "dept_core"],
        value_vars=day_columns,
        var_name="d",
        value_name="Sales",
    )
    del grouped
    gc.collect()

    long = long.merge(calendar[["d", "Date", "Holiday"]], on="d", how="left", validate="many_to_one")
    if "snap_CA" in calendar.columns or "snap_TX" in calendar.columns or "snap_WI" in calendar.columns:
        long = long.merge(
            calendar[["d"] + [c for c in ("snap_CA", "snap_TX", "snap_WI") if c in calendar.columns]],
            on="d",
            how="left",
            validate="many_to_one",
        )

    long["State"] = long["store_id"].astype(str).str[:2].str.upper()
    snap_columns = [f"snap_{s}" for s in ("CA", "TX", "WI")]
    promo_series = np.zeros(len(long), dtype=int)
    for state in ("CA", "TX", "WI"):
        col = f"snap_{state}"
        if col in long.columns:
            promo_series = np.where(long["State"].to_numpy() == state,
                                    (long[col].fillna(0).to_numpy() > 0).astype(int),
                                    promo_series)
    long["Promo"] = promo_series

    out = pd.DataFrame({
        "Store": long["store_id"].astype(str),
        "Dept": long["dept_core"].astype(str),
        "Date": pd.to_datetime(long["Date"], errors="coerce"),
        "Sales": long["Sales"].astype(float).clip(lower=0),
        "OnHand": (long["Sales"].astype(float).clip(lower=0) * 14.0).clip(lower=0),
        "Promo": long["Promo"].astype(int),
        "Holiday": long["Holiday"].astype(int),
    })

    lead_times = []
    suppliers = []
    for _, row in out[["Store", "Dept"]].iterrows():
        supplier, lead = _supplier_for(row["Store"], row["Dept"])
        suppliers.append(supplier)
        lead_times.append(int(lead))
    out["Supplier"] = suppliers
    out["LeadTime"] = lead_times

    out = out.dropna(subset=["Store", "Dept", "Date", "Sales"]).copy()
    out["LeadTime"] = out["LeadTime"].astype(int).clip(lower=1)
    out["Supplier"] = out["Supplier"].astype(str)
    out["Holiday"] = out["Holiday"].astype(int)
    out["Promo"] = out["Promo"].astype(int)
    return out.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)


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


def _m5_is_newer_than_cache(raw_dir: Path, cache_path: Path) -> bool:
    try:
        cache_mtime = cache_path.stat().st_mtime
        return max((raw_dir / f).stat().st_mtime for f in M5_REQUIRED_FILES) > cache_mtime
    except OSError:
        return False


def load_or_generate(path: Path = DATA_PATH, force: bool = False, raw_dir: str | Path = M5_RAW_DIR) -> pd.DataFrame:
    raw = Path(raw_dir)

    if not force and path.exists():
        force_from_m5 = m5_files_available(raw) and _m5_is_newer_than_cache(raw, path)
        if not force_from_m5:
            df = pd.read_csv(path)
            if set(REQUIRED_COLUMNS).issubset(df.columns):
                df = normalize_dataset(df)
                if not df.empty:
                    return df
        force = True

    if m5_files_available(raw):
        df = normalize_dataset(load_m5_sales(raw))
    else:
        # Explicit fallback, used only when the real M5 files are unavailable.
        df = normalize_dataset(generate_dataset())

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, date_format="%Y-%m-%d")
    return df


def list_series(df: pd.DataFrame):
    return sorted(df[["Store", "Dept"]].drop_duplicates().itertuples(index=False, name=None))
