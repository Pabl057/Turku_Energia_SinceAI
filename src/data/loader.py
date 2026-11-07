"""Data loading and anti-leakage filtering utilities (RULE #1: first_usable_origin_utc)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

VALIDATION_FIRST_ORIGIN = pd.Timestamp("2026-09-06T21:00:00Z")
TEST_FIRST_ORIGIN = pd.Timestamp("2026-10-04T21:00:00Z")
HORIZON_STEPS = 40
STEP_MINUTES = 15
SPIKE_MARGIN = 50.0


def to_utc(value) -> pd.Timestamp:
    """Convert any timestamp-like value to a timezone-aware UTC Timestamp."""
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def read_long(path: Path) -> pd.DataFrame:
    """Read a long-format data file (parquet or csv) with UTC timestamps."""
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, dtype={"dataset_id": str})
    for col in ("timestamp_utc", "first_usable_origin_utc"):
        df[col] = pd.to_datetime(df[col], utc=True)
    return df


def load_all_data(data_dir: Path | str = DEFAULT_DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """Load inputs.parquet (or inputs.csv) and spot_nordpool.csv.

    Returns (inputs_df, spot_df, combined_long_df).
    Note: Raw Nord Pool spot values must never be logged or committed (RULE #2).
    """
    data_dir = Path(data_dir)
    inputs_path = data_dir / "inputs.parquet"
    try:
        inputs = read_long(inputs_path)
    except (ImportError, FileNotFoundError, ValueError):
        inputs = read_long(data_dir / "inputs.csv")

    spot_path = data_dir / "spot_nordpool.csv"
    spot = read_long(spot_path) if spot_path.exists() else None

    if spot is not None and len(spot):
        combined = pd.concat([inputs, spot], ignore_index=True)
    else:
        combined = inputs.copy()
    return inputs, spot, combined


def long_rows_available_at(data: pd.DataFrame, origin_utc) -> pd.DataFrame:
    """Return long-format rows strictly satisfying first_usable_origin_utc <= origin_utc."""
    origin = to_utc(origin_utc)
    return data.loc[data["first_usable_origin_utc"] <= origin].copy()


def rows_available_at(data: pd.DataFrame, origin_utc) -> pd.DataFrame:
    """Return the data that is usable at the forecast origin as a wide table.

    Strictly enforces RULE #1: keeps only rows with first_usable_origin_utc <= origin_utc
    and returns one row per timestamp_utc and one column per series_id.
    """
    origin = to_utc(origin_utc)
    usable = data.loc[data["first_usable_origin_utc"] <= origin]
    wide = usable.pivot(index="timestamp_utc", columns="series_id", values="value")
    return wide.sort_index()


def read_origins(path: Path | str) -> pd.Series:
    """Read forecast_origin_utc timestamps from a CSV file."""
    return pd.to_datetime(pd.read_csv(path)["forecast_origin_utc"], utc=True)


def get_hourly_origins(start_utc, end_utc) -> pd.DatetimeIndex:
    """Generate hourly UTC forecast origins in [start_utc, end_utc] inclusive."""
    return pd.date_range(to_utc(start_utc), to_utc(end_utc), freq="h", tz="UTC")
