"""One-off training script: LightGBM global model + chronological backtest."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_or_generate
from src.db import init_db
from src.eval import run_backtest
from src.inventory import seed_inventory_from_data
from src.model import train_all


def main():
    print("Loading data...")
    df = load_or_generate()
    print(f"Rows: {len(df)}")
    init_db()
    seed_inventory_from_data(df)
    print("Training LightGBM global model...")
    lb = train_all(df)
    print("Holdout metrics:", lb["metrics"])
    print("Running 3-window chronological backtest (28-day horizon)...")
    result = run_backtest(df, horizon=28, windows=3)
    for name, metrics in result["primary_overall_macro"].items():
        print(f"  {name}: {metrics}")
    print("Done. Model cached in models/, backtest in reports/.")


if __name__ == "__main__":
    main()
