"""Data loading and strict availability filtering (RULE #1: first_usable_origin_utc <= origin)."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def to_utc(value) -> pd.Timestamp:
    """Convert a timestamp or string to a UTC-aware pd.Timestamp."""
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def read_long(path: Path) -> pd.DataFrame:
    """Read a long-format data file (parquet or csv) with UTC timestamps."""
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, dtype={"dataset_id": str})
    for col in ("timestamp_utc", "first_usable_origin_utc"):
        df[col] = pd.to_datetime(df[col], utc=True)
    return df


def rows_available_at(data: pd.DataFrame, origin_utc) -> pd.DataFrame:
    """Return the data that is usable at the forecast origin as a wide table.

    RULE #1 (Mandatory Anti-Leakage Filter):
    Keeps ONLY rows with first_usable_origin_utc <= origin_utc and returns one
    row per timestamp_utc and one column per series_id.
    """
    origin = to_utc(origin_utc)
    usable = data.loc[data["first_usable_origin_utc"] <= origin]
    wide = usable.pivot(index="timestamp_utc", columns="series_id", values="value")
    return wide.sort_index()


def load_all_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load inputs.parquet (or inputs.csv) and spot_nordpool.csv into a single long DataFrame."""
    inputs_path = data_dir / "inputs.parquet"
    if not inputs_path.exists():
        inputs_path = data_dir / "inputs.csv"
    inputs = read_long(inputs_path)

    spot_path = data_dir / "spot_nordpool.csv"
    if spot_path.exists():
        spot = read_long(spot_path)
        combined = pd.concat([inputs, spot], ignore_index=True)
    else:
        combined = inputs

    return combined.sort_values(["timestamp_utc", "series_id"]).reset_index(drop=True)


def read_origins(path: Path) -> pd.DatetimeIndex:
    """Read forecast origins as a UTC DatetimeIndex."""
    df = pd.read_csv(path)
    return pd.DatetimeIndex(pd.to_datetime(df["forecast_origin_utc"], utc=True))
