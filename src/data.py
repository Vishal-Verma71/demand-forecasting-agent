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

from .config import DATA_DIR, STORES, DEPARTMENTS, RANDOM_SEED, INVENTORY_PROXY_DAYS

DATA_PATH = DATA_DIR / "supply_chain_sales.csv"
M5_RAW_DIR = DATA_DIR / "raw" / "m5"
M5_REQUIRED_FILES = ("sales_train_validation.csv", "calendar.csv")

SUPPLIERS = {
    "FOODS": [("FreshCo", 3), ("GlobalFoods", 5)],
    "HOBBIES": [("ToyWorld", 7), ("CraftHub", 10)],
    "HOUSEHOLD": [("HomeEssentials", 5), ("CleanPro", 4)],
}

# Commercial terms used for supplier allocation.  M5 has no supplier data, so these
# are documented business assumptions: relative unit cost (1.00 = list price) and
# on-time delivery reliability.  Faster suppliers are deliberately more expensive.
SUPPLIER_TERMS = {
    "FreshCo": {"unit_cost": 1.08, "reliability": 0.97},
    "GlobalFoods": {"unit_cost": 1.00, "reliability": 0.92},
    "ToyWorld": {"unit_cost": 1.06, "reliability": 0.95},
    "CraftHub": {"unit_cost": 0.94, "reliability": 0.88},
    "HomeEssentials": {"unit_cost": 0.97, "reliability": 0.93},
    "CleanPro": {"unit_cost": 1.04, "reliability": 0.96},
}

# Inventory ledger simulation (see simulate_inventory).  The simulated stores run the
# "traditional static" policy the agent is meant to improve on: a weekly min/max
# order sized from the trailing 28-day average, with no forecast, no promotion
# awareness and no statistical safety stock.
REVIEW_PERIOD_DAYS = 7
FUTURE_CALENDAR_PATH = DATA_DIR / "m5_calendar_future.csv"

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


def select_supplier(store: str, dept: str):
    '''Choose the configured supplier with the shortest supplier lead time.

    M5 does not contain supplier or lead-time data.  Supplier choices are therefore
    documented business assumptions.  When more fields are available, the tie-break
    is documented and configurable, but only lead time is currently real.  Store is
    retained in the signature because a future, richer supplier table would use it.
    '''
    choices = SUPPLIERS.get(dept.upper(), [("UnknownSupplier", 7)])
    return min(choices, key=lambda choice: (choice[1], choice[0]))


def _supplier_for(store: str, dept: str):
    return select_supplier(store, dept)


def _write_future_calendar(calendar: pd.DataFrame, sales_days: set[str]) -> None:
    """Persist the part of the M5 calendar after the last sales day (events + SNAP).

    The raw M5 folder is not committed, so this small file keeps the promotional
    schedule available to the forecaster and the agent in a fresh clone.
    """
    future = calendar[~calendar["d"].isin(sales_days)].copy()
    if future.empty:
        return
    names = future[["event_name_1", "event_name_2"]].fillna("") if "event_name_1" in future.columns else None
    future["Event"] = (names.apply(lambda r: " / ".join(v for v in r if v), axis=1)
                       if names is not None else "")
    cols = ["Date", "Holiday", "Event"] + [c for c in ("snap_CA", "snap_TX", "snap_WI") if c in future.columns]
    future[cols].to_csv(FUTURE_CALENDAR_PATH, index=False, date_format="%Y-%m-%d")


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
    _write_future_calendar(calendar, set(day_columns))

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
        "OnHand": (long["Sales"].astype(float).clip(lower=0) * INVENTORY_PROXY_DAYS).clip(lower=0),
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


def simulate_inventory(sales: np.ndarray, lead_time: int, buffer_days: float,
                       phase: int, review_days: int = REVIEW_PERIOD_DAYS,
                       reliability: float = 1.0, seed: int = RANDOM_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Replay a periodic-review, order-up-to replenishment policy over actual sales.

    M5 has no stock data.  Instead of a constant "N days of sales" proxy, we run the
    policy a typical store would use: every ``review_days`` it orders up to
    (review + lead + buffer) days of trailing-average demand, and the order arrives
    ``lead_time`` days later.  With probability ``1 - reliability`` a delivery is a
    short shipment (40-90% of the order), as happens with real suppliers.  Sales
    that exceed stock are lost.  Returns end-of-day on-hand units and units on
    order (placed but not yet received).
    """
    rng = np.random.default_rng(seed)
    n = len(sales)
    on_hand = np.zeros(n)
    on_order = np.zeros(n)
    cover_days = review_days + lead_time + buffer_days
    stock = float(np.mean(sales[:28])) * cover_days if n else 0.0
    pipeline: dict[int, float] = {}
    for t in range(n):
        arriving = pipeline.pop(t, 0.0)
        if arriving and rng.random() > reliability:
            arriving *= rng.uniform(0.4, 0.9)
        stock += arriving
        stock = max(stock - float(sales[t]), 0.0)
        if (t + phase) % review_days == 0:
            avg = float(np.mean(sales[max(0, t - 27):t + 1]))
            position = stock + sum(pipeline.values())
            qty = max(0.0, avg * cover_days - position)
            if qty > 0:
                pipeline[t + lead_time] = pipeline.get(t + lead_time, 0.0) + qty
        on_hand[t] = stock
        on_order[t] = sum(pipeline.values())
    return on_hand, on_order


def add_inventory_simulation(df: pd.DataFrame) -> pd.DataFrame:
    """Attach simulated OnHand/OnOrder columns to every store/department series."""
    out = df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True).copy()
    out["OnHand"] = 0.0
    out["OnOrder"] = 0.0
    out["DaysToReview"] = 0
    for (store, dept), idx in out.groupby(["Store", "Dept"]).groups.items():
        key = sum(ord(c) for c in f"{store}{dept}")
        lead = int(out.loc[idx[0], "LeadTime"])
        supplier = str(out.loc[idx[0], "Supplier"])
        buffer_days = 0.0                         # static policy: no safety buffer
        phase = key % REVIEW_PERIOD_DAYS          # stores review on different weekdays
        reliability = SUPPLIER_TERMS.get(supplier, {}).get("reliability", 0.9)
        on_hand, on_order = simulate_inventory(out.loc[idx, "Sales"].to_numpy(dtype=float),
                                               lead, buffer_days, phase,
                                               reliability=reliability, seed=RANDOM_SEED + key)
        out.loc[idx, "OnHand"] = np.round(on_hand, 1)
        out.loc[idx, "OnOrder"] = np.round(on_order, 1)
        # Days from each date until the store's next weekly review (1..7).
        t = np.arange(len(idx))
        out.loc[idx, "DaysToReview"] = REVIEW_PERIOD_DAYS - (t + phase) % REVIEW_PERIOD_DAYS
    return out


def future_calendar(store: str, start, periods: int) -> pd.DataFrame:
    """Known-in-advance calendar (M5 events + SNAP days) for forecast dates.

    M5's calendar.csv runs 56 days past the last sales day, so promotions and
    events are genuinely known for the whole forecast horizon.  Falls back to a
    fixed holiday list when the calendar file is unavailable.
    """
    dates = pd.date_range(pd.Timestamp(start), periods=periods, freq="D")
    out = pd.DataFrame({"Date": dates, "Promo": 0, "Holiday": 0, "Event": ""})
    out["Holiday"] = [int((d.month, d.day) in HOLIDAYS) for d in dates]
    if FUTURE_CALENDAR_PATH.exists():
        cal = pd.read_csv(FUTURE_CALENDAR_PATH, parse_dates=["Date"])
        snap_col = f"snap_{store[:2].upper()}"
        cal = cal.set_index("Date")
        known = out["Date"].isin(cal.index)
        rows = cal.reindex(out.loc[known, "Date"])
        out.loc[known, "Holiday"] = rows["Holiday"].fillna(0).astype(int).to_numpy()
        out.loc[known, "Event"] = rows["Event"].fillna("").astype(str).to_numpy()
        if snap_col in cal.columns:
            out.loc[known, "Promo"] = rows[snap_col].fillna(0).astype(int).to_numpy()
    return out


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    out = df.copy()
    if "OnHand" not in out.columns:
        out["OnHand"] = out["Sales"].astype(float) * INVENTORY_PROXY_DAYS
    if "OnOrder" not in out.columns:
        out["OnOrder"] = 0.0
    if "DaysToReview" not in out.columns:
        out["DaysToReview"] = REVIEW_PERIOD_DAYS
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
    out["OnOrder"] = out["OnOrder"].astype(float).clip(lower=0)
    out["DaysToReview"] = out["DaysToReview"].astype(int).clip(1, REVIEW_PERIOD_DAYS)
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
                needs_ledger = "DaysToReview" not in df.columns  # cache predates the inventory simulation
                df = normalize_dataset(df)
                if not df.empty:
                    if needs_ledger:
                        df = add_inventory_simulation(df)
                        df.to_csv(path, index=False, date_format="%Y-%m-%d")
                    return df
        force = True

    if m5_files_available(raw):
        df = load_m5_sales(raw)
    else:
        # Explicit fallback, used only when the real M5 files are unavailable.
        df = generate_dataset()
    df = add_inventory_simulation(normalize_dataset(df))

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, date_format="%Y-%m-%d")
    return df


def data_diagnostics(raw_dir: str | Path = M5_RAW_DIR,
                    stores: list[str] | None = None,
                    departments: list[str] | None = None) -> dict:
    '''Return read-only counts that distinguish source data from selected app output.'''
    raw = Path(raw_dir)
    stores = stores or STORES
    departments = departments or DEPARTMENTS
    raw_item_rows_by_store = {}
    raw_selected_item_rows_by_store = {}
    raw_selected_item_rows_by_department = {dept: 0 for dept in departments}
    if (raw / "sales_train_validation.csv").is_file():
        import csv
        with open(raw / "sales_train_validation.csv", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                store = row.get("store_id", "")
                dept = row.get("dept_id", "").split("_")[0]
                raw_item_rows_by_store[store] = raw_item_rows_by_store.get(store, 0) + 1
                if store in stores:
                    raw_selected_item_rows_by_store[store] = raw_selected_item_rows_by_store.get(store, 0) + 1
                    if dept in raw_selected_item_rows_by_department:
                        raw_selected_item_rows_by_department[dept] += 1

    cached = load_or_generate()
    cached_by_store = cached.groupby("Store").size().astype(int).to_dict()
    cached_by_store_department = {
        f"{store}/{dept}": int(len(cached[(cached.Store == store) & (cached.Dept == dept)]))
        for store in stores for dept in departments
    }

    return {
        "source": "Kaggle M5 aggregated daily store/department demand",
        "m5_source_item_rows_by_store": raw_item_rows_by_store,
        "m5_selected_item_rows_by_store": raw_selected_item_rows_by_store,
        "m5_selected_item_rows_by_department": raw_selected_item_rows_by_department,
        "m5_daily_columns": 1913 if m5_files_available(raw) else 0,
        "cached_data_rows": int(len(cached)),
        "cached_rows_by_store": cached_by_store,
        "cached_rows_by_store_department": cached_by_store_department,
        "inventory_assumption": {
            "OnHand": f"estimated at {INVENTORY_PROXY_DAYS} days of recent demand",
            "Supplier": "deterministic existing supplier configuration, not a Kaggle field",
            "LeadTime": "existing business assumption, not a Kaggle field",
        },
    }


def list_series(df: pd.DataFrame):
    return sorted(df[["Store", "Dept"]].drop_duplicates().itertuples(index=False, name=None))
