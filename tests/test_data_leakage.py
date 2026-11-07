"""Rule #1 Data Availability & Anti-Leakage tests (`first_usable_origin_utc <= origin`)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.loader import load_all_data, long_rows_available_at, rows_available_at, to_utc
from src.features.build_features import FeatureStore, build_feature_Frame


@pytest.fixture(scope="module")
def real_feature_store() -> FeatureStore:
    """Load real combined dataset once for Rule #1 availability checks."""
    _, _, combined = load_all_data()
    return FeatureStore(combined)


def _make_synthetic_long_df(origin_str: str = "2026-09-15T12:00:00Z") -> pd.DataFrame:
    origin = to_utc(origin_str)
    rows = [
        # Historical series usable 1h before origin
        {
            "timestamp_utc": origin - pd.Timedelta(hours=2),
            "series_id": "fi_imbalance_price",
            "value": 45.0,
            "first_usable_origin_utc": origin - pd.Timedelta(hours=1),
        },
        # Historical series usable exactly at origin (boundary: first_usable_origin_utc == origin)
        {
            "timestamp_utc": origin - pd.Timedelta(hours=1),
            "series_id": "fi_imbalance_price",
            "value": 50.0,
            "first_usable_origin_utc": origin,
        },
        # Future delivery day-ahead forecast whose first_usable_origin_utc <= origin (usable!)
        {
            "timestamp_utc": origin + pd.Timedelta(hours=3),
            "series_id": "fi_spot_price",
            "value": 60.0,
            "first_usable_origin_utc": origin - pd.Timedelta(hours=12),
        },
        # Row with first_usable_origin_utc == origin + 15m (MUST be excluded at origin)
        {
            "timestamp_utc": origin - pd.Timedelta(minutes=45),
            "series_id": "fi_imbalance_price",
            "value": 999.0,
            "first_usable_origin_utc": origin + pd.Timedelta(minutes=15),
        },
        # Row with first_usable_origin_utc == origin + 1h (MUST be excluded at origin)
        {
            "timestamp_utc": origin,
            "series_id": "fi_imbalance_price",
            "value": 888.0,
            "first_usable_origin_utc": origin + pd.Timedelta(hours=1),
        },
    ]
    return pd.DataFrame(rows)


def test_rows_available_at_filters_future_first_usable_rows() -> None:
    """Verify rows_available_at and long_rows_available_at retain only first_usable_origin_utc <= t."""
    origin_str = "2026-09-15T12:00:00Z"
    origin = to_utc(origin_str)
    df = _make_synthetic_long_df(origin_str)

    usable_long = long_rows_available_at(df, origin)
    assert len(usable_long) == 3
    assert (usable_long["first_usable_origin_utc"] <= origin).all()
    assert 999.0 not in usable_long["value"].values
    assert 888.0 not in usable_long["value"].values

    wide = rows_available_at(df, origin)
    # Day-ahead row at origin + 3h has first_usable_origin_utc <= origin, so it is present
    assert (origin + pd.Timedelta(hours=3)) in wide.index
    assert wide.loc[origin + pd.Timedelta(hours=3), "fi_spot_price"] == 60.0
    # Future-usable rows at origin - 45m and origin must not be present
    assert (origin - pd.Timedelta(minutes=45)) not in wide.index
    assert origin not in wide.index
    assert wide.loc[origin - pd.Timedelta(hours=1), "fi_imbalance_price"] == 50.0


def test_rows_available_at_handles_naive_and_tz_aware_origins() -> None:
    """Verify naive and UTC-aware origin timestamps produce identical filtered tables."""
    df = _make_synthetic_long_df("2026-09-15T12:00:00Z")
    wide_naive = rows_available_at(df, "2026-09-15T12:00:00")
    wide_aware = rows_available_at(df, "2026-09-15T12:00:00Z")
    pd.testing.assert_frame_equal(wide_naive, wide_aware)


def test_feature_store_lookup_at_masks_unusable_rows() -> None:
    """Verify FeatureStore.lookup_at returns NaN whenever first_usable_origin_utc > origin_utc."""
    origin = to_utc("2026-09-15T12:00:00Z")
    df = _make_synthetic_long_df("2026-09-15T12:00:00Z")
    store = FeatureStore(df)

    targets = pd.DatetimeIndex([
        origin - pd.Timedelta(hours=2),
        origin - pd.Timedelta(hours=1),
        origin - pd.Timedelta(minutes=45),
        origin,
    ])
    origins = pd.DatetimeIndex([origin] * len(targets))

    vals = store.lookup_at("fi_imbalance_price", targets, origins)
    assert vals[0] == pytest.approx(45.0)
    assert vals[1] == pytest.approx(50.0)
    assert np.isnan(vals[2])
    assert np.isnan(vals[3])


def test_feature_store_latest_usable_timestamp_respects_lag_groups(
    real_feature_store: FeatureStore,
) -> None:
    """Verify latest_usable_timestamp_per_origin respects 60m, 30m, and 120m lag boundaries."""
    origins = pd.date_range("2026-07-15T00:00:00Z", "2026-07-15T23:00:00Z", freq="h", tz="UTC")
    for series_id in ("fi_imbalance_price", "fi_consumption", "se1_imbalance_price_long"):
        ts_idx = real_feature_store.latest_usable_timestamp_per_origin(series_id, origins)
        fu_used = pd.DatetimeIndex(real_feature_store.fu_wide[series_id].reindex(ts_idx))
        assert (fu_used <= origins).all(), f"{series_id} used a row with first_usable_origin_utc > origin"

        # The very next quarter-hour (ts + 15m) must have first_usable_origin_utc > origin (or be NaN)
        next_ts = ts_idx + pd.Timedelta(minutes=15)
        fu_next = pd.DatetimeIndex(real_feature_store.fu_wide[series_id].reindex(next_ts))
        valid_next = ~fu_next.isna()
        assert (fu_next[valid_next] > origins[valid_next]).all(), (
            f"{series_id} next quarter-hour was already usable at origin"
        )


def test_build_feature_frame_target_cutoff_masks_post_cutoff_targets(
    real_feature_store: FeatureStore,
) -> None:
    """Verify target_cutoff_utc masks targets whose delivery quarter-hour is not yet usable at cutoff."""
    cutoff = pd.Timestamp("2026-09-06T21:00:00Z")
    origins = pd.date_range("2026-09-06T12:00:00Z", "2026-09-06T21:00:00Z", freq="h", tz="UTC")

    df, _ = build_feature_Frame(
        real_feature_store,
        origins,
        include_targets=True,
        target_cutoff_utc=cutoff,
    )

    # Every row must satisfy data_cutoff_utc <= forecast_origin_utc
    assert (pd.to_datetime(df["data_cutoff_utc"], utc=True) <= pd.to_datetime(df["forecast_origin_utc"], utc=True)).all()

    deliv_idx = pd.DatetimeIndex(pd.to_datetime(df["delivery_start_utc"], utc=True))
    deliv_fu = pd.DatetimeIndex(real_feature_store.fu_wide["fi_imbalance_price"].reindex(deliv_idx))
    post_cutoff_mask = deliv_fu > cutoff
    assert post_cutoff_mask.any(), "Expected at least some delivery steps to fall after cutoff"

    target_cols = [
        "target_imbalance_price",
        "target_spread",
        "target_direction",
        "target_spike_high",
        "target_spike_low",
    ]
    for col in target_cols:
        assert df.loc[post_cutoff_mask, col].isna().all(), (
            f"Target column {col} leaked post-cutoff labels!"
        )


def test_realtime_3min_resampling_and_availability_lag() -> None:
    """Verify 3-min Fingrid series (181/188/191/177) resample cleanly and enforce first_usable_origin_utc."""
    from src.data.extra_data import REALTIME_FINGRID_SERIES, resample_3min_to_quarter_hour

    spec_wind = REALTIME_FINGRID_SERIES[0]  # 181 Wind RT
    raw = pd.DataFrame([
        {"startTime": "2026-09-10T10:00:00Z", "value": 1200.0},
        {"startTime": "2026-09-10T10:03:00Z", "value": 1230.0},
        {"startTime": "2026-09-10T10:06:00Z", "value": 1260.0},
        {"startTime": "2026-09-10T10:15:00Z", "value": 1400.0},
    ])

    # Default cautious lag = 30m after quarter end (10:00..10:15 -> usable at 10:45)
    df_30 = resample_3min_to_quarter_hour(raw, spec_wind, rt_lag_minutes=30)
    assert len(df_30) == 2
    assert df_30.loc[0, "value"] == pytest.approx(1230.0)
    assert df_30.loc[0, "first_usable_origin_utc"] == pd.Timestamp("2026-09-10T10:45:00Z")

    # Optional 10m lag (10:00..10:15 -> usable at 10:25)
    df_10 = resample_3min_to_quarter_hour(raw, spec_wind, rt_lag_minutes=10)
    assert df_10.loc[0, "first_usable_origin_utc"] == pd.Timestamp("2026-09-10T10:25:00Z")

