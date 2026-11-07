"""End-to-end validation, SHAP explainability, and benchmark comparison tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from score import parse_args, score_submission
from src.data.loader import load_all_data
from src.evaluation.pipeline import PROJECT_ROOT, run_full_pipeline, write_submission
from src.features.build_features import FeatureStore, build_feature_Frame
from src.models.train import INT_TO_DIRECTION, DIRECTION_TO_INT, walk_forward_predict
from validate_submission import ALL_COLUMNS, validate


def test_mini_walk_forward_pipeline_and_shap_e2e(tmp_path: Path) -> None:
    """Run a compact 7-day walk-forward training + 3-origin evaluation slice, validate, score, and verify SHAP drivers."""
    _, _, combined = load_all_data()
    store = FeatureStore(combined)

    train_origins = pd.date_range("2026-08-24T00:00:00Z", "2026-08-31T00:00:00Z", freq="h", tz="UTC")
    eval_origins = pd.date_range("2026-09-01T00:00:00Z", "2026-09-01T02:00:00Z", freq="h", tz="UTC")
    all_origins = train_origins.union(eval_origins)

    df_all, feat_cols = build_feature_Frame(store, all_origins, include_targets=True)
    df_all["target_fu_utc"] = pd.DatetimeIndex(
        store.fu_wide["fi_imbalance_price"].reindex(pd.DatetimeIndex(df_all["delivery_start_utc"])).to_numpy()
    )

    pred_df, suite = walk_forward_predict(df_all, feat_cols, eval_origins, retrain_step_days=7)
    assert len(pred_df) == 120  # 3 origins x 40 steps

    # Verify SHAP top_drivers_for_frame returns 3 semicolon-separated driver names per row
    df_eval = df_all.loc[df_all["forecast_origin_utc"].isin(eval_origins)].reset_index(drop=True)
    drivers_list, shap_vals = suite.top_drivers_for_frame(df_eval[feat_cols], top_k=3)
    assert shap_vals.shape == (120, len(feat_cols))
    assert len(drivers_list) == 120
    for d_str in drivers_list:
        parts = d_str.split(";")
        assert len(parts) == 3
        assert len(set(parts)) == 3
        assert all(len(p) > 0 for p in parts)

    # Write origins, template, and submission CSV
    origins_path = tmp_path / "forecast_origins.csv"
    pd.DataFrame({
        "forecast_origin_utc": eval_origins.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }).to_csv(origins_path, index=False)

    template_path = tmp_path / "sample_submission.csv"
    pd.DataFrame(columns=ALL_COLUMNS).to_csv(template_path, index=False)

    sub_path = tmp_path / "forecast_backtest.csv"
    write_submission(pred_df, sub_path, template_path)

    report, counts = validate(sub_path, origins_path, 40, 15, 0.001)
    assert report.n_errors == 0, f"Mini submission validation errors: {report.findings}"
    assert counts["rows"] == 120

    # Build matching actuals CSV in tmp_path from store.val_wide for the evaluated delivery timestamps
    deliv_idx = pd.DatetimeIndex(pd.to_datetime(pred_df["delivery_start_utc"], utc=True)).unique().sort_values()
    dir_num = store.val_wide["fi_mfrr_direction"].reindex(deliv_idx)
    dir_str = dir_num.map(lambda v: INT_TO_DIRECTION.get(DIRECTION_TO_INT.get(v, -1), ""))
    actuals_df = pd.DataFrame({
        "delivery_start_utc": deliv_idx.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "imbalance_price": store.val_wide["fi_imbalance_price"].reindex(deliv_idx).to_numpy(),
        "spot_price": store.val_wide["fi_spot_price"].reindex(deliv_idx).to_numpy(),
        "direction": dir_str.to_numpy(),
    })
    actuals_path = tmp_path / "actuals.csv"
    actuals_df.to_csv(actuals_path, index=False)

    score_out_dir = tmp_path / "score_out"
    args = parse_args([
        "--submission", str(sub_path),
        "--actuals", str(actuals_path),
        "--spike-margin", "50",
        "--output-dir", str(score_out_dir),
    ])
    assert score_submission(args) == 0
    summary = pd.read_csv(score_out_dir / "score_summary.csv").iloc[0]
    assert int(summary["rows_scored_price"]) == 120
    assert np.isfinite(float(summary["mae_p50"]))
    assert np.isfinite(float(summary["calibration_pinball_mean"]))
    assert np.isfinite(float(summary["brier_spike_mean"]))
    assert np.isfinite(float(summary["brier_direction"]))


def test_validation_backtest_beats_official_baseline() -> None:
    """Verify validation backtest passes validator and beats all official baseline benchmarks."""
    val_sub_path = PROJECT_ROOT / "results" / "validation" / "forecast_backtest.csv"
    val_score_path = PROJECT_ROOT / "results" / "validation" / "score" / "score_summary.csv"
    final_sub_path = PROJECT_ROOT / "forecast_backtest.csv"

    if not (val_sub_path.exists() and val_score_path.exists() and final_sub_path.exists()):
        run_full_pipeline()

    val_origins_path = PROJECT_ROOT / "baseline_output" / "validation" / "forecast_origins.csv"
    report, counts = validate(val_sub_path, val_origins_path, 40, 15, 0.001)
    assert report.n_errors == 0, f"Validation submission failed validator: {report.findings}"
    assert counts["rows"] == counts["rows_expected"] == 26880

    summary = pd.read_csv(val_score_path).iloc[0]
    mae_p50 = float(summary["mae_p50"])
    pinball_mean = float(summary["calibration_pinball_mean"])
    brier_spike = float(summary["brier_spike_mean"])
    brier_dir = float(summary["brier_direction"])
    coverage = float(summary["coverage_p10_p90"])

    # Official baseline benchmarks from AGENTS.md / baseline.ipynb
    assert mae_p50 < 66.2033, f"MAE {mae_p50} did not beat baseline 66.2033"
    assert pinball_mean < 24.1776, f"Pinball {pinball_mean} did not beat baseline 24.1776"
    assert brier_spike < 0.1132, f"Spike Brier {brier_spike} did not beat baseline 0.1132"
    assert brier_dir < 0.4509, f"Direction Brier {brier_dir} did not beat baseline 0.4509"
    assert 0.75 <= coverage <= 0.85, f"Coverage {coverage} outside [0.75, 0.85] calibration target"
