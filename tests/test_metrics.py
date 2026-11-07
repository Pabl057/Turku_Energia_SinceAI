"""Evaluation metrics & D14 points calculation tests (`score.py`)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from score import (
    SCORER_VERSION,
    compute_metrics,
    compute_points,
    criterion_points,
    no_team_beats_baseline,
    parse_args,
    pinball,
    score_submission,
)


def test_pinball_loss_asymmetry_and_zero_at_exact_forecast() -> None:
    """Verify pinball loss equals 0 at y == yhat and has exact q / (1 - q) asymmetry."""
    y_exact = pd.Series([10.0, 50.0, -20.0])
    assert pinball(y_exact, y_exact, 0.10) == pytest.approx(0.0)
    assert pinball(y_exact, y_exact, 0.50) == pytest.approx(0.0)
    assert pinball(y_exact, y_exact, 0.90) == pytest.approx(0.0)

    # Underprediction by 20: diff = y - yhat = +20 -> q * 20
    y_under = pd.Series([100.0])
    yhat_under = pd.Series([80.0])
    assert pinball(y_under, yhat_under, 0.90) == pytest.approx(18.0)
    assert pinball(y_under, yhat_under, 0.10) == pytest.approx(2.0)

    # Overprediction by 20: diff = y - yhat = -20 -> (q - 1) * (-20)
    y_over = pd.Series([80.0])
    yhat_over = pd.Series([100.0])
    assert pinball(y_over, yhat_over, 0.10) == pytest.approx(18.0)
    assert pinball(y_over, yhat_over, 0.90) == pytest.approx(2.0)


def test_compute_metrics_mae_pinball_coverage_and_brier() -> None:
    """Verify compute_metrics matches closed-form hand-calculated MAE, Pinball, Coverage, and Brier scores."""
    # Row 0: actual=120, spot=50 (high spike: 120 > 100), direction="up", inside [p10=40, p90=140], p50=100
    # Row 1: actual=-20, spot=40 (low spike: -20 < -10), direction="down", outside [p10=0, p90=60], p50=20
    df = pd.DataFrame({
        "imbalance_price": [120.0, -20.0],
        "spot_price": [50.0, 40.0],
        "direction": ["up", "down"],
        "p10": [40.0, 0.0],
        "p50": [100.0, 20.0],
        "p90": [140.0, 60.0],
        "p_up": [0.8, 0.1],
        "p_down": [0.1, 0.7],
        "p_none": [0.1, 0.2],
        "p_spike_high": [0.9, 0.2],
        "p_spike_low": [0.1, 0.6],
    })

    m = compute_metrics(df, spike_margin=50.0)

    # MAE on P50: (|120 - 100| + |-20 - 20|) / 2 = (20 + 40) / 2 = 30.0
    assert m["mae_p50"] == pytest.approx(30.0)

    # Pinball P10:
    # Row 0: diff = 120 - 40 = +80 -> 0.10 * 80 = 8.0
    # Row 1: diff = -20 - 0 = -20 -> -0.90 * -20 = 18.0
    # Mean = 13.0
    assert m["pinball_p10"] == pytest.approx(13.0)

    # Pinball P90:
    # Row 0: diff = 120 - 140 = -20 -> -0.10 * -20 = 2.0
    # Row 1: diff = -20 - 60 = -80 -> -0.10 * -80 = 8.0
    # Mean = 5.0
    assert m["pinball_p90"] == pytest.approx(5.0)
    assert m["calibration_pinball_mean"] == pytest.approx(9.0)

    # Coverage: Row 0 in [40, 140] (True), Row 1 in [0, 60] (False) -> 0.5
    assert m["coverage_p10_p90"] == pytest.approx(0.5)

    # Spike High: Row 0 high=1 (p=0.9 -> 0.01), Row 1 high=0 (p=0.2 -> 0.04) -> mean = 0.025
    assert m["brier_spike_high"] == pytest.approx(0.025)
    # Spike Low: Row 0 low=0 (p=0.1 -> 0.01), Row 1 low=1 (p=0.6 -> 0.16) -> mean = 0.085
    assert m["brier_spike_low"] == pytest.approx(0.085)
    assert m["brier_spike_mean"] == pytest.approx(0.055)

    # Direction 3-class Brier (range [0, 2]):
    # Row 0 ("up"): (0.8 - 1)^2 + (0.1 - 0)^2 + (0.1 - 0)^2 = 0.04 + 0.01 + 0.01 = 0.06
    # Row 1 ("down"): (0.1 - 0)^2 + (0.7 - 1)^2 + (0.2 - 0)^2 = 0.01 + 0.09 + 0.04 = 0.14
    # Mean = 0.10
    assert m["brier_direction"] == pytest.approx(0.10)
    assert 0.0 <= m["brier_direction"] <= 2.0


def test_criterion_points_and_mode2_points_calculation(tmp_path: Path) -> None:
    """Verify criterion_points D14 scaling, clipping, zero-tolerance rule, and Mode 2 compute_points."""
    # Best team gets full weight
    assert criterion_points(baseline=66.20, team=43.72, best=43.72, weight=30.0, tolerance=1e-6) == pytest.approx(30.0)

    # Halfway between baseline and best gets half weight
    assert criterion_points(baseline=60.0, team=50.0, best=40.0, weight=30.0, tolerance=1e-6) == pytest.approx(15.0)

    # Team worse than or equal to baseline clips to 0.0
    assert criterion_points(baseline=66.20, team=66.20, best=43.72, weight=30.0, tolerance=1e-6) == pytest.approx(0.0)
    assert criterion_points(baseline=66.20, team=75.00, best=43.72, weight=30.0, tolerance=1e-6) == pytest.approx(0.0)

    # When no team beats baseline by more than tolerance, everyone gets 0.0
    assert no_team_beats_baseline(baseline=66.20, best=66.20, tolerance=1e-6) is True
    assert criterion_points(baseline=66.20, team=66.20, best=66.20, weight=30.0, tolerance=1e-6) == pytest.approx(0.0)

    # End-to-end Mode 1 + Mode 2 test on tmp_path
    sub_df = pd.DataFrame({
        "forecast_origin_utc": ["2026-09-07T00:00:00Z"],
        "delivery_start_utc": ["2026-09-07T00:00:00Z"],
        "step": [1],
        "p10": [30.0],
        "p50": [50.0],
        "p90": [70.0],
        "p_up": [0.7],
        "p_down": [0.2],
        "p_none": [0.1],
        "p_spike_high": [0.1],
        "p_spike_low": [0.1],
    })
    act_df = pd.DataFrame({
        "delivery_start_utc": ["2026-09-07T00:00:00Z"],
        "imbalance_price": [55.0],
        "spot_price": [45.0],
        "direction": ["up"],
    })
    sub_path = tmp_path / "sub.csv"
    act_path = tmp_path / "act.csv"
    sub_df.to_csv(sub_path, index=False)
    act_df.to_csv(act_path, index=False)

    team_dir = tmp_path / "results" / "team_alpha"
    args_m1 = parse_args([
        "--submission", str(sub_path),
        "--actuals", str(act_path),
        "--spike-margin", "50",
        "--output-dir", str(team_dir),
    ])
    assert score_submission(args_m1) == 0
    assert (team_dir / "score_summary.csv").exists()

    # Create a worse baseline summary in results/baseline/score_summary.csv
    base_dir = tmp_path / "results" / "baseline"
    base_dir.mkdir(parents=True, exist_ok=True)
    base_summary = pd.read_csv(team_dir / "score_summary.csv")
    base_summary["mae_p50"] = 25.0
    base_summary["calibration_pinball_mean"] = 10.0
    base_summary["brier_spike_mean"] = 0.20
    base_summary["brier_direction"] = 0.60
    base_summary["scorer_version"] = SCORER_VERSION
    base_path = base_dir / "score_summary.csv"
    base_summary.to_csv(base_path, index=False)

    points_out = tmp_path / "results" / "points_summary.csv"
    args_m2 = parse_args([
        "--points", str(tmp_path / "results"),
        "--baseline", str(base_path),
        "--output", str(points_out),
    ])
    assert compute_points(args_m2) == 0
    pts_df = pd.read_csv(points_out)
    assert len(pts_df) == 1
    assert pts_df.loc[0, "team_id"] == "team_alpha"
    assert pts_df.loc[0, "points_automatic_total"] == pytest.approx(80.0)
