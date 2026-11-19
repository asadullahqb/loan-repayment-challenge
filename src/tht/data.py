"""Data-loading utilities for the THT loan-repayment challenge."""
import os
from pathlib import Path

import pandas as pd

# Absolute path to the raw parquet folder, independent of the notebook's cwd.
# Override with the THT_DATA_DIR env var (e.g. a Google Drive mount on Colab).
_env_dir = os.environ.get("THT_DATA_DIR")
if _env_dir:
    DATA_DIR = Path(_env_dir)
else:
    DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "parquet"

FILES = {
    "loan": "loan.parquet",
    "payment": "payment.parquet",
    "underwriting": "clarity_underwriting_variables.parquet",
}


def load_data():
    """Load all three parquet files into a dict of DataFrames."""
    return {name: pd.read_parquet(DATA_DIR / fname) for name, fname in FILES.items()}
