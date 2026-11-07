"""End-to-end training, validation, scoring, and backtest submission pipeline."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import DATA_DIR, PROJECT_ROOT, load_all_data, read_origins
from src.features.build_features import FeatureBuilder
from src.models.forecaster import ImbalanceForecasterSuite, apply_temporal_ensembling

VALIDATION_FIRST_ORIGIN = pd.Timestamp("2026-09-06T21:00:00Z")
VALIDATION_LAST_ORIGIN = pd.Timestamp("2026-10-04T20:00:00Z")
TEST_FIRST_ORIGIN = pd.Timestamp("2026-10-04T21:00:00Z")
TRAIN_START_ORIGIN = pd.Timestamp("2026-06-03T00:00:00Z")


def format_and_write_submission(df: pd.DataFrame, out_path: Path, template_path: Path) -> None:
    """Write a submission file matching exact column order and ISO-8601 Z timestamp formatting."""
    columns = pd.read_csv(template_path, nrows=0).columns.tolist()
    out = df.copy()
    for col in ("forecast_origin_utc", "delivery_start_utc", "data_cutoff_utc"):
        out[col] = pd.to_datetime(out[col], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out[columns].to_csv(out_path, index=False, float_format="%.6f", lineterminator="\n", encoding="utf-8")


def run_script(*args: str) -> tuple[int, str]:
    """Run an official root script (make_templates.py, validate_submission.py, score.py) and return (rc, stdout)."""
    result = subprocess.run(
        [sys.executable, *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    output = result.stdout.rstrip()
    if result.stderr.strip():
        output += "\n" + result.stderr.rstrip()
    return result.returncode, output


def predict_with_weekly_expanding_window(
    all_feat_df: pd.DataFrame,
    eval_origins: pd.DatetimeIndex,
    refit_days: int = 7,
    compute_shap: bool = True,
) -> pd.DataFrame:
    """Predict across `eval_origins` with weekly expanding-window refits.

    Strict Anti-Leakage Check (RULE #1):
    For each weekly chunk starting at `chunk_start_origin`, the model is trained ONLY on
    historical (origin, step) rows where:
      1. `forecast_origin_utc < chunk_start_origin`
      2. `target_first_usable_utc <= chunk_start_origin`
    """
    preds_list = []
    chunk_size = 24 * refit_days
    n_origins = len(eval_origins)

    for i in range(0, n_origins, chunk_size):
        chunk_origins = eval_origins[i : i + chunk_size]
        chunk_start = chunk_origins[0]

        # Strictly filter training data known at or before chunk_start
        train_mask = (
            (all_feat_df["forecast_origin_utc"] < chunk_start)
            & (all_feat_df["target_first_usable_utc"] <= chunk_start)
        )
        train_df = all_feat_df.loc[train_mask]

        # Assert zero leakage
        max_fu = train_df["target_first_usable_utc"].max()
        assert max_fu <= chunk_start, f"LEAKAGE DETECTED: {max_fu} > {chunk_start}"

        eval_mask = all_feat_df["forecast_origin_utc"].isin(chunk_origins)
        eval_df = all_feat_df.loc[eval_mask].copy()

        suite = ImbalanceForecasterSuite()
        suite.fit(train_df)
        chunk_pred = suite.predict(eval_df, compute_shap=compute_shap, temporal_ensemble=False)
        preds_list.append(chunk_pred)

    combined = pd.concat(preds_list, ignore_index=True)
    return combined


def main() -> int:
    t0 = time.time()
    print("1. Loading dataset (inputs.parquet + spot_nordpool.csv)...")
    long_data = load_all_data(DATA_DIR)
    print(f"   Loaded {len(long_data):,} rows across {long_data['series_id'].nunique()} series in {time.time() - t0:.1f}s.")

    print("2. Initializing leak-free FeatureBuilder...")
    builder = FeatureBuilder(long_data)

    # Prepare validation templates
    val_dir = PROJECT_ROOT / "results" / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    rc, _ = run_script(
        "make_templates.py",
        "--test-start",
        "2026-09-07",
        "--test-end",
        "2026-10-04",
        "--output-dir",
        str(val_dir.relative_to(PROJECT_ROOT)),
    )
    assert rc == 0, "make_templates.py failed for validation period"

    val_origins = read_origins(val_dir / "forecast_origins.csv")
    test_origins = read_origins(DATA_DIR / "forecast_origins.csv")

    # Build hourly training origins from June 3 to Oct 4
    train_origins = pd.date_range(TRAIN_START_ORIGIN, VALIDATION_LAST_ORIGIN, freq="1h", tz="UTC")
    all_origins = train_origins.union(val_origins).union(test_origins)

    t1 = time.time()
    print(f"3. Building feature matrix X for {len(all_origins):,} hourly origins ({len(all_origins) * 40:,} rows)...")
    all_feat_df = builder.build_for_origins(all_origins, include_targets=True)
    print(f"   Built {len(all_feat_df):,} rows x {len(all_feat_df.columns)} columns in {time.time() - t1:.1f}s.")

    # =========================================================================
    # 4. VALIDATION PERIOD EVALUATION (7 Sep - 4 Oct 2026, 672 origins)
    # =========================================================================
    t2 = time.time()
    print("\n4. Running Validation Period Backtest (weekly expanding window, strict anti-leakage)...")
    val_pred_raw = predict_with_weekly_expanding_window(
        all_feat_df, val_origins, refit_days=7, compute_shap=False
    )
    val_pred_ens = apply_temporal_ensembling(val_pred_raw, min_step_for_blend=5, max_prev_weight=0.18)
    print(f"   Validation predictions generated in {time.time() - t2:.1f}s.")

    val_sub_path = val_dir / "forecast_backtest.csv"
    format_and_write_submission(val_pred_ens, val_sub_path, val_dir / "sample_submission.csv")

    rc, val_check_out = run_script(
        "validate_submission.py",
        "--backtest",
        str(val_sub_path.relative_to(PROJECT_ROOT)),
        "--origins",
        str((val_dir / "forecast_origins.csv").relative_to(PROJECT_ROOT)),
    )
    print("\n--- Validation Format Check ---")
    print(val_check_out)
    assert rc == 0, "Validation format check failed!"

    score_dir = val_dir / "score"
    rc, score_out = run_script(
        "score.py",
        "--submission",
        str(val_sub_path.relative_to(PROJECT_ROOT)),
        "--actuals",
        "data/actuals_validation.csv",
        "--spike-margin",
        "50",
        "--output-dir",
        str(score_dir.relative_to(PROJECT_ROOT)),
    )
    print("\n--- Official Validation Score (score.py) ---")
    print(score_out)
    assert rc == 0, "score.py failed!"

    # =========================================================================
    # 5. TEST PERIOD BACKTEST SUBMISSION (5 Oct - 1 Nov 2026, 673 origins)
    # =========================================================================
    t3 = time.time()
    print("\n5. Generating Final Test Submission `forecast_backtest.csv` (673 origins = 26,920 rows) + SHAP drivers...")
    test_pred_raw = predict_with_weekly_expanding_window(
        all_feat_df, test_origins, refit_days=7, compute_shap=True
    )
    test_pred_ens = apply_temporal_ensembling(test_pred_raw, min_step_for_blend=5, max_prev_weight=0.18)

    final_sub_path = PROJECT_ROOT / "forecast_backtest.csv"
    format_and_write_submission(test_pred_ens, final_sub_path, DATA_DIR / "sample_submission.csv")
    print(f"   Wrote {final_sub_path.name} ({len(test_pred_ens):,} rows) in {time.time() - t3:.1f}s.")

    rc, test_check_out = run_script(
        "validate_submission.py",
        "--backtest",
        "forecast_backtest.csv",
        "--origins",
        "data/forecast_origins.csv",
    )
    print("\n--- Final Submission Validation Check (validate_submission.py) ---")
    print(test_check_out)
    assert rc == 0, "Final submission validation failed!"

    print(f"\nPipeline completed successfully in {time.time() - t0:.1f}s!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
