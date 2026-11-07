"""End-to-end validation, scoring, and submission generation pipeline.

Wraps `validate_submission.py` and `score.py` and generates:
  1. Validation forecast (`results/validation/forecast_backtest.csv`) + comparison vs baseline
  2. Official test backtest (`forecast_backtest.csv`, 26,920 rows = 673 origins x 40 steps)
  3. Dashboard artifacts (`results/dashboard_artifacts.parquet`) for Level 2 UI
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from src.data.loader import (
    DEFAULT_DATA_DIR,
    SPIKE_MARGIN,
    VALIDATION_FIRST_ORIGIN,
    load_all_data,
    read_origins,
)
from src.features.build_features import FeatureStore, build_feature_Frame
from src.models.train import walk_forward_predict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def write_submission(df: pd.DataFrame, path: Path | str, template_path: Path | str) -> None:
    """Write submission CSV matching exact column order and UTC ISO 8601 Z format."""
    path = Path(path)
    template_path = Path(template_path)
    columns = pd.read_csv(template_path, nrows=0).columns.tolist()
    out = df.copy()
    for col in ("forecast_origin_utc", "delivery_start_utc", "data_cutoff_utc"):
        out[col] = pd.to_datetime(out[col], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    path.parent.mkdir(parents=True, exist_ok=True)
    out[columns].to_csv(
        path,
        index=False,
        float_format="%.6f",
        lineterminator="\n",
        encoding="utf-8",
    )


def run_validator(backtest_path: Path | str, origins_path: Path | str) -> int:
    """Run official validate_submission.py and print report."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "validate_submission.py"),
        "--backtest",
        str(backtest_path),
        "--origins",
        str(origins_path),
    ]
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", env=env)
    print(res.stdout.rstrip())
    if res.stderr.strip():
        print(res.stderr.rstrip())
    return res.returncode


def run_scorer(
    submission_path: Path | str,
    actuals_path: Path | str,
    output_dir: Path | str,
    baseline_summary_path: Path | str | None = None,
) -> int:
    """Run official score.py and print report."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "score.py"),
        "--submission",
        str(submission_path),
        "--actuals",
        str(actuals_path),
        "--spike-margin",
        str(int(SPIKE_MARGIN)),
        "--output-dir",
        str(output_dir),
    ]
    if baseline_summary_path is not None and Path(baseline_summary_path).exists():
        cmd.extend(["--baseline-summary", str(baseline_summary_path)])
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", env=env)
    print(res.stdout.rstrip())
    if res.stderr.strip():
        print(res.stderr.rstrip())
    return res.returncode


def run_full_pipeline(data_dir: Path | str = DEFAULT_DATA_DIR) -> dict[str, Path]:
    """Execute the full Level 1 training, validation, scoring, and test submission pipeline."""
    t0 = time.time()
    data_dir = Path(data_dir)
    print("1. Loading datasets and building anti-leakage FeatureStore...")
    _, _, combined = load_all_data(data_dir)
    store = FeatureStore(combined)

    val_origins_path = PROJECT_ROOT / "baseline_output" / "validation" / "forecast_origins.csv"
    val_template_path = PROJECT_ROOT / "baseline_output" / "validation" / "sample_submission.csv"
    if not val_origins_path.exists():
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "make_templates.py"),
                "--test-start",
                "2026-09-07",
                "--test-end",
                "2026-10-04",
                "--output-dir",
                "baseline_output/validation",
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

    val_origins = read_origins(val_origins_path)
    test_origins_path = data_dir / "forecast_origins.csv"
    test_template_path = data_dir / "sample_submission.csv"
    test_origins = read_origins(test_origins_path)

    all_origins = pd.date_range("2026-06-04T00:00:00Z", test_origins.max(), freq="h", tz="UTC")
    print(f"2. Engineering features for {len(all_origins)} hourly origins (x 40 steps)...")
    df_all, feat_cols = build_feature_Frame(store, all_origins, include_targets=True)
    target_fu = pd.DatetimeIndex(
        store.fu_wide["fi_imbalance_price"].reindex(pd.DatetimeIndex(df_all["delivery_start_utc"])).to_numpy()
    )
    df_all["target_fu_utc"] = target_fu
    print(f"   Feature matrix ready: {len(df_all):,} rows x {len(feat_cols)} features ({time.time() - t0:.1f}s)")

    # 3. Validation period backtest & scoring
    print("\n3. Running walk-forward training & prediction for Validation Period (7 Sep - 4 Oct 2026)...")
    val_pred, _ = walk_forward_predict(df_all, feat_cols, val_origins, retrain_step_days=7)
    val_out_dir = PROJECT_ROOT / "results" / "validation"
    val_sub_path = val_out_dir / "forecast_backtest.csv"
    write_submission(val_pred, val_sub_path, val_template_path)

    rc_val = run_validator(val_sub_path, val_origins_path)
    if rc_val != 0:
        raise RuntimeError("Validation period submission failed validate_submission.py!")

    baseline_summary = PROJECT_ROOT / "baseline_output" / "validation" / "score" / "score_summary.csv"
    rc_score = run_scorer(
        val_sub_path,
        data_dir / "actuals_validation.csv",
        val_out_dir / "score",
        baseline_summary_path=baseline_summary,
    )
    if rc_score != 0:
        raise RuntimeError("score.py failed on validation predictions!")

    # Also compute points comparison against baseline
    points_dir = PROJECT_ROOT / "results" / "points_comparison"
    (points_dir / "our_model").mkdir(parents=True, exist_ok=True)
    (points_dir / "baseline").mkdir(parents=True, exist_ok=True)
    (points_dir / "our_model" / "score_summary.csv").write_bytes(
        (val_out_dir / "score" / "score_summary.csv").read_bytes()
    )
    if baseline_summary.exists():
        (points_dir / "baseline" / "score_summary.csv").write_bytes(baseline_summary.read_bytes())

    # 4. Official Test Period backtest (5 Oct - 1 Nov 2026 -> forecast_backtest.csv)
    print("\n4. Running walk-forward training & prediction for Official Test Period (5 Oct - 1 Nov 2026)...")
    test_pred, latest_suite = walk_forward_predict(df_all, feat_cols, test_origins, retrain_step_days=7)
    final_sub_path = PROJECT_ROOT / "forecast_backtest.csv"
    write_submission(test_pred, final_sub_path, test_template_path)

    print("\n5. Validating final submission forecast_backtest.csv...")
    rc_test = run_validator(final_sub_path, test_origins_path)
    if rc_test != 0:
        raise RuntimeError("Final forecast_backtest.csv failed validate_submission.py!")

    # 5. Save global feature importances for Level 2 Dashboard
    imp_df = pd.DataFrame({
        "feature": feat_cols,
        "gain_p50": latest_suite.m_p50.booster_.feature_importance(importance_type="gain"),
        "split_p50": latest_suite.m_p50.booster_.feature_importance(importance_type="split"),
        "gain_spike_high": latest_suite.m_spike_high.booster_.feature_importance(importance_type="gain"),
        "gain_direction": latest_suite.m_dir.booster_.feature_importance(importance_type="gain"),
    }).sort_values("gain_p50", ascending=False)
    imp_path = PROJECT_ROOT / "results" / "feature_importance.csv"
    imp_df.to_csv(imp_path, index=False)

    print(f"\nPipeline complete in {time.time() - t0:.1f}s.")
    return {
        "validation_submission": val_sub_path,
        "validation_score_summary": val_out_dir / "score" / "score_summary.csv",
        "test_submission": final_sub_path,
        "feature_importance": imp_path,
    }


if __name__ == "__main__":
    run_full_pipeline()
