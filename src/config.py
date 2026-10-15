"""Project configuration & environment."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
load_dotenv(BASE_DIR / ".env", override=False)

# Runtime state (inventory snapshot, alerts, purchase orders). Override for tests or demos.
DB_PATH = Path(os.getenv("SUPPLY_CHAIN_DB", str(BASE_DIR / "data" / "supply_chain.db")))

# LLM backend (optional; falls back to deterministic router if none set)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto")  # auto | groq | openai | ollama | none
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# Forecast
FORECAST_HORIZON_DEFAULT = 28
SERVICE_LEVEL_Z = 1.65  # safety-stock z-score (~95% cycle service level)
INVENTORY_PROXY_DAYS = 8.0  # only used when a dataset arrives without stock columns
RANDOM_SEED = 42

# Dataset
STORES = ["CA_1", "CA_2", "TX_1", "TX_2", "WI_1"]
DEPARTMENTS = ["FOODS", "HOBBIES", "HOUSEHOLD"]

MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
(MODEL_DIR / ".keep").touch(exist_ok=True)
