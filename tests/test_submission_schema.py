"""Rule #3 Submission Schema & Mathematical Constraints tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.train import enforce_prediction_constraints
from validate_submission import (
    ALL_COLUMNS,
    DIRECTION_COLUMNS,
    QUANTILE_COLUMNS,
    REQUIRED_COLUMNS,
    SPIKE_COLUMNS,
    validate,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_enforce_prediction_constraints_repairs_violations() -> None:
    """Verify enforce_prediction_constraints repairs quantile crossing, unnormalized direction, and spike sum > 1."""
    raw = pd.DataFrame({
        "forecast_origin_utc": ["2026-10-05T00:00:00Z", "2026-10-05T00:00:00Z"],
        "delivery_start_utc": ["2026-10-05T00:00:00Z", "2026-10-05T00:15:00Z"],
        "step": [1, 2],
        "p10": [90.0, 40.0],  # row 0 has p10 > p50
        "p50": [50.0, 60.0],
        "p90": [30.0, 55.0],  # both rows have p90 < p50
        "p_up": [0.9, -0.2],  # row 0 sum = 1.8; row 1 negative
        "p_down": [0.5, 0.7],
        "p_none": [0.4, 0.3],
        "p_spike_high": [0.8, -0.1],  # row 0 sum = 1.4; row 1 negative
        "p_spike_low": [0.6, 1.3],
        "data_cutoff_utc": ["2026-10-04T23:00:00Z", "2026-10-05T00:00:00Z"],
        "top_drivers": ["a;b;c", "d;e;f"],
    })

    fixed = enforce_prediction_constraints(raw)

    assert (fixed["p10"] <= fixed["p50"]).all()
    assert (fixed["p50"] <= fixed["p90"]).all()

    for col in DIRECTION_COLUMNS + SPIKE_COLUMNS:
        assert ((fixed[col] >= 0.0) & (fixed[col] <= 1.0)).all()

    dir_sum = fixed["p_up"] + fixed["p_down"] + fixed["p_none"]
    assert np.allclose(dir_sum, 1.0, atol=1e-6)

    spike_sum = fixed["p_spike_high"] + fixed["p_spike_low"]
    assert (spike_sum <= 1.0).all()


def test_enforce_prediction_constraints_rejects_cutoff_after_origin() -> None:
    """Verify enforce_prediction_constraints raises AssertionError when data_cutoff_utc > forecast_origin_utc."""
    bad = pd.DataFrame({
        "forecast_origin_utc": ["2026-10-05T00:00:00Z"],
        "delivery_start_utc": ["2026-10-05T00:00:00Z"],
        "step": [1],
        "p10": [10.0],
        "p50": [20.0],
        "p90": [30.0],
        "p_up": [0.3],
        "p_down": [0.3],
        "p_none": [0.4],
        "p_spike_high": [0.1],
        "p_spike_low": [0.1],
        "data_cutoff_utc": ["2026-10-05T00:15:00Z"],  # 15m after origin!
        "top_drivers": ["a;b;c"],
    })
    with pytest.raises(AssertionError, match="Leakage in data_cutoff_utc"):
        enforce_prediction_constraints(bad)


def _build_minimal_valid_submission() -> tuple[pd.DataFrame, pd.DataFrame]:
    origins = ["2026-10-05T00:00:00Z", "2026-10-05T01:00:00Z"]
    origins_df = pd.DataFrame({"forecast_origin_utc": origins})
    rows = []
    for orig_str in origins:
        orig_ts = pd.Timestamp(orig_str)
        cutoff_str = (orig_ts - pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for s in range(1, 41):
            deliv_str = (orig_ts + pd.Timedelta(minutes=15 * (s - 1))).strftime("%Y-%m-%dT%H:%M:%SZ")
            rows.append({
                "forecast_origin_utc": orig_str,
                "delivery_start_utc": deliv_str,
                "step": str(s),
                "p10": "20.000000",
                "p50": "40.000000",
                "p90": "70.000000",
                "p_up": "0.300000",
                "p_down": "0.200000",
                "p_none": "0.500000",
                "p_spike_high": "0.100000",
                "p_spike_low": "0.050000",
                "data_cutoff_utc": cutoff_str,
                "top_drivers": "day_ahead_spot_price;recent_imbalance_spread;residual_load_forecast",
            })
    return origins_df, pd.DataFrame(rows)[ALL_COLUMNS]


def test_validate_submission_catches_each_schema_and_constraint_violation(tmp_path: Path) -> None:
    """Verify official validate_submission.validate catches every schema and mathematical violation."""
    origins_df, valid_df = _build_minimal_valid_submission()
    origins_path = tmp_path / "origins.csv"
    origins_df.to_csv(origins_path, index=False)

    valid_path = tmp_path / "valid.csv"
    valid_df.to_csv(valid_path, index=False)
    report, counts = validate(valid_path, origins_path, 40, 15, 0.001)
    assert report.n_errors == 0
    assert counts["rows"] == 80

    def _codes_for(mutated: pd.DataFrame, name: str) -> set[str]:
        p = tmp_path / f"{name}.csv"
        mutated.to_csv(p, index=False)
        rep, _ = validate(p, origins_path, 40, 15, 0.001)
        return {f.code for f in rep.findings if f.level == "ERROR"}

    # 1. Swapped p10 > p50 -> INVALID_QUANTILE_ORDER
    df_q = valid_df.copy()
    df_q.loc[0, "p10"] = "55.000000"
    assert "INVALID_QUANTILE_ORDER" in _codes_for(df_q, "bad_q")

    # 2. p_up + p_down + p_none = 1.05 -> INVALID_PROBABILITY_SUM
    df_dir = valid_df.copy()
    df_dir.loc[0, "p_up"] = "0.350000"
    assert "INVALID_PROBABILITY_SUM" in _codes_for(df_dir, "bad_dir")

    # 3. p_spike_high + p_spike_low = 1.05 -> INVALID_SPIKE_SUM
    df_spk = valid_df.copy()
    df_spk.loc[0, "p_spike_high"] = "0.600000"
    df_spk.loc[0, "p_spike_low"] = "0.450000"
    assert "INVALID_SPIKE_SUM" in _codes_for(df_spk, "bad_spk")

    # 4. data_cutoff_utc > forecast_origin_utc -> DATA_CUTOFF_AFTER_ORIGIN
    df_cut = valid_df.copy()
    df_cut.loc[0, "data_cutoff_utc"] = "2026-10-05T00:15:00Z"
    assert "DATA_CUTOFF_AFTER_ORIGIN" in _codes_for(df_cut, "bad_cut")

    # 5. Missing step row -> MISSING_ROWS and ROW_COUNT
    df_miss = valid_df.iloc[1:].copy()
    miss_codes = _codes_for(df_miss, "bad_miss")
    assert "MISSING_ROWS" in miss_codes
    assert "ROW_COUNT" in miss_codes

    # 6. Duplicate (forecast_origin_utc, step) -> DUPLICATE_ROWS
    df_dup = valid_df.copy()
    df_dup.iloc[1] = df_dup.iloc[0]
    assert "DUPLICATE_ROWS" in _codes_for(df_dup, "bad_dup")

    # 7. Non-UTC or fractional-second timestamp -> INVALID_TIMESTAMP
    df_ts = valid_df.copy()
    df_ts.loc[0, "delivery_start_utc"] = "2026-10-05T00:00:00.123Z"
    assert "INVALID_TIMESTAMP" in _codes_for(df_ts, "bad_ts")


def test_official_forecast_backtest_csv_passes_validator() -> None:
    """Verify forecast_backtest.csv has 26,920 rows (673 origins x 40 steps), 13 columns, and 0 validator errors."""
    backtest_path = PROJECT_ROOT / "forecast_backtest.csv"
    origins_path = PROJECT_ROOT / "data" / "forecast_origins.csv"
    sample_path = PROJECT_ROOT / "data" / "sample_submission.csv"

    assert backtest_path.exists(), "forecast_backtest.csv is missing"
    assert origins_path.exists(), "data/forecast_origins.csv is missing"
    assert sample_path.exists(), "data/sample_submission.csv is missing"

    sample_cols = pd.read_csv(sample_path, nrows=0).columns.tolist()
    sub_cols = pd.read_csv(backtest_path, nrows=0).columns.tolist()
    assert sub_cols == sample_cols == ALL_COLUMNS
    assert set(REQUIRED_COLUMNS).issubset(sub_cols)

    report, counts = validate(backtest_path, origins_path, 40, 15, 0.001)
    assert report.n_errors == 0, f"Validator errors: {report.findings}"
    assert report.n_warnings == 0, f"Validator warnings: {report.findings}"
    assert counts["origins_expected"] == 673
    assert counts["origins_found"] == 673
    assert counts["rows_expected"] == 26920
    assert counts["rows"] == 26920
