"""One-off training script: Prophet (all series) + LightGBM (global)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import STORES, DEPARTMENTS
from src.data import load_or_generate
from src.inventory import seed_inventory_from_data
from src.db import init_db
from src.model import train_all


def main():
    print("Loading/generating data...")
    df = load_or_generate()
    print(f"Rows: {len(df)}")
    init_db()
    seed_inventory_from_data(df)
    print("Training LightGBM global model...")
    lb = train_all(df)
    print("LightGBM metrics:", lb["metrics"])
    print("Done. Models cached in models/.")


if __name__ == "__main__":
    main()
