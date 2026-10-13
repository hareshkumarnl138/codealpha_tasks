"""Shared data loading, validation, and project path utilities."""
from pathlib import Path
import logging
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "default of credit card clients.xls"
PROCESSED_DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
MODELS = ROOT / "models"
TARGET = "default_payment"
ID_COLUMN = "id"

def setup_logging(name: str = "credit_scoring") -> logging.Logger:
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    return logger

def normalize_column_name(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")

def load_raw_data(path: Path = RAW_DATA) -> pd.DataFrame:
    """Load the original UCI XLS file. The source file uses its second row as headers."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Place 'default of credit card clients.xls' inside data/raw/ and try again."
        )
    try:
        df = pd.read_excel(path, header=1)
    except ImportError as exc:
        raise RuntimeError("Excel reader missing. Run: python -m pip install -r requirements.txt") from exc

    df.columns = [normalize_column_name(c) for c in df.columns]
    # Standardize the source target label.
    target_candidates = {"default_payment_next_month", "default_payment_next_month_"}
    found_target = next((c for c in df.columns if c in target_candidates), None)
    if found_target:
        df = df.rename(columns={found_target: TARGET})
    if "id" not in df.columns and "id_" in df.columns:
        df = df.rename(columns={"id_": "id"})
    if TARGET not in df.columns:
        raise ValueError(
            "Could not identify the target column 'default payment next month'. "
            f"Columns found: {df.columns.tolist()}"
        )
    df = df.dropna(how="all").copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 (no default) and 1 (default).")
    return df

def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return model features, excluding target and customer identifier."""
    return [c for c in df.columns if c not in {TARGET, ID_COLUMN}]
