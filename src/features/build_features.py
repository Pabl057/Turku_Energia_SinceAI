"""Anti-leakage feature engineering pipeline for Turku Energia Imbalance Price Forecasting.

Strictly enforces RULE #1: for any forecast origin t, only rows with
first_usable_origin_utc <= t are ever accessed. Every lookup is guarded by the
per-series first_usable_origin_utc matrix (mathematically equivalent to rows_available_at).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.loader import HORIZON_STEPS, SPIKE_MARGIN, STEP_MINUTES, to_utc

# Human-readable labels for SHAP top_drivers explanations
FEATURE_DRIVER_NAMES: dict[str, str] = {
    "spot_at_delivery": "day_ahead_spot_price",
    "last_imbalance_spread": "recent_imbalance_spread",
    "last_imbalance_price": "last_imbalance_price",
    "imb_spread_mean_4q": "imbalance_spread_1h_mean",
    "imb_spread_mean_16q": "imbalance_spread_4h_mean",
    "imb_price_mean_4q": "imbalance_price_1h_mean",
    "decay_spread_short": "short_horizon_spread_inertia",
    "decay_spread_medium": "medium_horizon_spread_inertia",
    "wind_da_at_delivery": "wind_power_forecast",
    "solar_da_at_delivery": "solar_power_forecast",
    "cons_da_at_delivery": "consumption_forecast",
    "residual_load_da_at_delivery": "residual_load_forecast",
    "residual_load_ramp_1h": "residual_load_1h_ramp",
    "residual_load_ramp_3h": "residual_load_3h_ramp",
    "wind_da_ramp_1h": "wind_forecast_1h_ramp",
    "wind_da_ramp_3h": "wind_forecast_3h_ramp",
    "spot_ramp_1h": "spot_price_1h_ramp",
    "spot_ramp_3h": "spot_price_3h_ramp",
    "cons_forecast_error_30m": "realtime_consumption_forecast_error",
    "cons_forecast_error_mean_4q": "consumption_forecast_error_1h_mean",
    "renewable_error_proxy_change_1h": "wind_generation_deviation_1h",
    "production_ramp_1h": "realtime_production_1h_ramp",
    "net_import_30m": "realtime_net_import",
    "total_cross_border_flow_30m": "cross_border_net_flow",
    "last_mfrr_direction": "recent_mfrr_direction",
    "mfrr_dir_mean_4q": "mfrr_direction_1h_trend",
    "mfrr_dir_mean_16q": "mfrr_direction_4h_trend",
    "last_mfrr_up_activated": "mfrr_up_regulation_volume",
    "last_mfrr_down_activated": "mfrr_down_regulation_volume",
    "last_mfrr_net_activated": "mfrr_net_regulation_volume",
    "mfrr_net_act_mean_4q": "mfrr_net_volume_1h_mean",
    "last_mfrr_up_spread": "mfrr_up_price_spread",
    "last_mfrr_down_spread": "mfrr_down_price_spread",
    "last_afrr_up_price": "afrr_up_price",
    "last_afrr_down_price": "afrr_down_price",
    "last_afrr_up_spread": "afrr_up_price_spread",
    "last_afrr_down_spread": "afrr_down_price_spread",
    "spot_lead_1h": "spot_price_1h_lead_ramp",
    "se1_load_da_at_delivery": "sweden_se1_load_forecast",
    "se3_load_da_at_delivery": "sweden_se3_load_forecast",
    "ee_load_da_at_delivery": "estonia_ee_load_forecast",
    "se1_imb_spread_120m": "sweden_se1_imbalance_spread",
    "se3_imb_spread_120m": "sweden_se3_imbalance_spread",
    "ee_imb_spread_120m": "estonia_ee_imbalance_spread",
    "imb_spread_lag_24h": "imbalance_spread_24h_lag",
    "imb_spread_lag_168h": "imbalance_spread_7d_lag",
    "mfrr_dir_lag_24h": "mfrr_direction_24h_lag",
    "step": "forecast_horizon_step",
    "cal_hour_local": "time_of_day",
    "cal_weekday_local": "day_of_week",
}


class FeatureStore:
    """Pre-indexed wide value and availability matrices for leak-free feature extraction."""

    def __init__(self, combined_long: pd.DataFrame) -> None:
        self.val_wide = (
            combined_long.pivot(index="timestamp_utc", columns="series_id", values="value")
            .sort_index()
            .astype(float)
        )
        self.fu_wide = (
            combined_long.pivot(
                index="timestamp_utc", columns="series_id", values="first_usable_origin_utc"
            ).sort_index()
        )

    def lookup_at(
        self,
        series_id: str,
        target_timestamps: pd.DatetimeIndex | pd.Series,
        origin_timestamps: pd.DatetimeIndex | pd.Series,
    ) -> np.ndarray:
        """Look up `series_id` at `target_timestamps` strictly masked by `first_usable_origin_utc <= origin_timestamps`."""
        if series_id not in self.val_wide.columns:
            return np.full(len(target_timestamps), np.nan, dtype=float)

        vals = self.val_wide[series_id].reindex(target_timestamps).to_numpy(dtype=float)
        fus = pd.DatetimeIndex(self.fu_wide[series_id].reindex(target_timestamps))
        orig_idx = pd.DatetimeIndex(origin_timestamps)
        valid_mask = (~fus.isna()) & (fus <= orig_idx)
        return np.where(valid_mask, vals, np.nan)
    def latest_usable_timestamp_per_origin(
        self, series_id: str, origins: pd.DatetimeIndex
    ) -> pd.DatetimeIndex:
        """For each origin t, return the latest timestamp_utc of series_id where first_usable_origin_utc <= t."""
        s_fu = self.fu_wide[series_id].dropna()
        s_val = self.val_wide[series_id].reindex(s_fu.index)
        s_fu = s_fu[s_val.notna()]
        pos = s_fu.searchsorted(origins, side="right") - 1
        if (pos < 0).any():
            raise ValueError(f"No usable {series_id} row before {origins[pos < 0][0]}")
        ts_idx = s_fu.index[pos]
        used_fu = pd.DatetimeIndex(s_fu.iloc[pos])
        if not (used_fu <= origins).all():
            raise AssertionError(f"Data leakage detected in {series_id}!")
        return pd.DatetimeIndex(ts_idx)


def _rolling_backward_stats(series: pd.Series, windows: list[int]) -> dict[str, pd.Series]:
    """Compute backward-looking rolling stats on a 15-min indexed series."""
    out: dict[str, pd.Series] = {}
    for w in windows:
        r = series.rolling(window=w, min_periods=1)
        out[f"mean_{w}q"] = r.mean()
        out[f"std_{w}q"] = r.std().fillna(0.0)
        out[f"min_{w}q"] = r.min()
        out[f"max_{w}q"] = r.max()
    return out


def build_feature_Frame(
    store: FeatureStore,
    origins: pd.DatetimeIndex | pd.Series,
    horizon_steps: int = HORIZON_STEPS,
    include_targets: bool = False,
    target_cutoff_utc: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the full feature matrix X (and optional targets) for given origins and 40 steps."""
    origins_idx = pd.DatetimeIndex(pd.to_datetime(origins, utc=True))
    n_origins = len(origins_idx)

    # 1. Precompute origin-level lagged features (strictly usable at t)
    t60_idx = store.latest_usable_timestamp_per_origin("fi_imbalance_price", origins_idx)
    t30_idx = store.latest_usable_timestamp_per_origin("fi_consumption", origins_idx)
    t120_idx = store.latest_usable_timestamp_per_origin("se1_imbalance_price_long", origins_idx)

    imb_s = store.val_wide["fi_imbalance_price"]
    spot_s = store.val_wide["fi_spot_price"]
    spread_s = imb_s - spot_s
    dir_s = store.val_wide["fi_mfrr_direction"]
    mfrr_up_act_s = store.val_wide["fi_mfrr_up_activated"].fillna(0.0)
    mfrr_dn_act_s = store.val_wide["fi_mfrr_down_activated"].fillna(0.0)
    mfrr_net_act_s = mfrr_up_act_s - mfrr_dn_act_s
    mfrr_up_p_s = store.val_wide["fi_mfrr_up_price"]
    mfrr_dn_p_s = store.val_wide["fi_mfrr_down_price"]
    afrr_up_p_s = store.val_wide["fi_afrr_up_price"]
    afrr_dn_p_s = store.val_wide["fi_afrr_down_price"]
    spike_high_s = (spread_s > SPIKE_MARGIN).astype(float)
    spike_low_s = (spread_s < -SPIKE_MARGIN).astype(float)
    dir_up_s = (dir_s == 1.0).astype(float)
    dir_dn_s = (dir_s == -1.0).astype(float)
    dir_none_s = (dir_s == 0.0).astype(float)

    imb_stats = _rolling_backward_stats(imb_s, [4, 8, 16, 32, 96])
    spread_stats = _rolling_backward_stats(spread_s, [4, 8, 16, 32, 96])

    ocols: dict[str, Any] = {
        "forecast_origin_utc": origins_idx,
        "data_cutoff_utc": t60_idx,
        "last_imbalance_price": imb_s.reindex(t60_idx).to_numpy(),
        "last_spot_at_60m": spot_s.reindex(t60_idx).to_numpy(),
        "last_imbalance_spread": spread_s.reindex(t60_idx).to_numpy(),
    }

    for lag_q in (1, 2, 3, 4, 8):
        lag_ts = t60_idx - pd.Timedelta(minutes=15 * lag_q)
        ocols[f"imb_price_lag_{lag_q}q"] = imb_s.reindex(lag_ts).to_numpy()
        ocols[f"imb_spread_lag_{lag_q}q"] = spread_s.reindex(lag_ts).to_numpy()
        ocols[f"mfrr_dir_lag_{lag_q}q"] = dir_s.reindex(lag_ts).to_numpy()

    for k, s_stat in imb_stats.items():
        ocols[f"imb_price_{k}"] = s_stat.reindex(t60_idx).to_numpy()
    for k, s_stat in spread_stats.items():
        ocols[f"imb_spread_{k}"] = s_stat.reindex(t60_idx).to_numpy()

    ocols["imb_spread_momentum_1h"] = ocols["last_imbalance_spread"] - ocols["imb_spread_mean_4q"]
    ocols["imb_spread_momentum_4h"] = ocols["imb_spread_mean_4q"] - ocols["imb_spread_mean_16q"]
    ocols["imb_spread_range_4q"] = ocols["imb_spread_max_4q"] - ocols["imb_spread_min_4q"]
    ocols["imb_spread_range_16q"] = ocols["imb_spread_max_16q"] - ocols["imb_spread_min_16q"]

    ocols["last_mfrr_direction"] = dir_s.reindex(t60_idx).to_numpy()
    for w in (4, 8, 16, 32, 96):
        ocols[f"mfrr_dir_mean_{w}q"] = dir_s.rolling(w, min_periods=1).mean().reindex(t60_idx).to_numpy()
        ocols[f"mfrr_up_share_{w}q"] = dir_up_s.rolling(w, min_periods=1).mean().reindex(t60_idx).to_numpy()
        ocols[f"mfrr_dn_share_{w}q"] = dir_dn_s.rolling(w, min_periods=1).mean().reindex(t60_idx).to_numpy()
        ocols[f"mfrr_none_share_{w}q"] = dir_none_s.rolling(w, min_periods=1).mean().reindex(t60_idx).to_numpy()
        ocols[f"spike_high_share_{w}q"] = spike_high_s.rolling(w, min_periods=1).mean().reindex(t60_idx).to_numpy()
        ocols[f"spike_low_share_{w}q"] = spike_low_s.rolling(w, min_periods=1).mean().reindex(t60_idx).to_numpy()

    ocols["last_mfrr_up_activated"] = mfrr_up_act_s.reindex(t60_idx).to_numpy()
    ocols["last_mfrr_down_activated"] = mfrr_dn_act_s.reindex(t60_idx).to_numpy()
    ocols["last_mfrr_net_activated"] = mfrr_net_act_s.reindex(t60_idx).to_numpy()
    for w in (4, 16):
        ocols[f"mfrr_up_act_mean_{w}q"] = mfrr_up_act_s.rolling(w, min_periods=1).mean().reindex(t60_idx).to_numpy()
        ocols[f"mfrr_dn_act_mean_{w}q"] = mfrr_dn_act_s.rolling(w, min_periods=1).mean().reindex(t60_idx).to_numpy()
        ocols[f"mfrr_net_act_mean_{w}q"] = mfrr_net_act_s.rolling(w, min_periods=1).mean().reindex(t60_idx).to_numpy()

    ocols["last_mfrr_up_price"] = mfrr_up_p_s.reindex(t60_idx).to_numpy()
    ocols["last_mfrr_down_price"] = mfrr_dn_p_s.reindex(t60_idx).to_numpy()
    ocols["last_mfrr_up_spread"] = ocols["last_mfrr_up_price"] - ocols["last_spot_at_60m"]
    ocols["last_mfrr_down_spread"] = ocols["last_mfrr_down_price"] - ocols["last_spot_at_60m"]

    afrr_up_ff = afrr_up_p_s.ffill(limit=8)
    afrr_dn_ff = afrr_dn_p_s.ffill(limit=8)
    ocols["last_afrr_up_price"] = afrr_up_ff.reindex(t60_idx).to_numpy()
    ocols["last_afrr_down_price"] = afrr_dn_ff.reindex(t60_idx).to_numpy()
    ocols["last_afrr_up_spread"] = ocols["last_afrr_up_price"] - ocols["last_spot_at_60m"]
    ocols["last_afrr_down_spread"] = ocols["last_afrr_down_price"] - ocols["last_spot_at_60m"]
    ocols["last_afrr_up_active"] = afrr_up_p_s.notna().astype(float).rolling(4, min_periods=1).mean().reindex(t60_idx).to_numpy()
    ocols["last_afrr_down_active"] = afrr_dn_p_s.notna().astype(float).rolling(4, min_periods=1).mean().reindex(t60_idx).to_numpy()

    # 30m lag features
    cons_s = store.val_wide["fi_consumption"].ffill(limit=4)
    prod_s = store.val_wide["fi_production"].ffill(limit=4)
    cons_da_s = store.val_wide["fi_consumption_fc_da"]
    wind_da_s = store.val_wide["fi_wind_fc_da"]
    solar_da_s = store.val_wide["fi_solar_fc_da"]
    cons_err_s = cons_s - cons_da_s
    net_import_s = cons_s - prod_s
    ren_err_proxy_s = prod_s - (wind_da_s + solar_da_s)

    se1_fl_s = store.val_wide["fi_se1_flow"].ffill(limit=4)
    se3_fl_s = store.val_wide["fi_se3_flow"].ffill(limit=4)
    ee_fl_s = store.val_wide["fi_ee_flow"].ffill(limit=4)
    no4_fl_s = store.val_wide["fi_no4_flow"].ffill(limit=4)
    tot_flow_s = se1_fl_s + se3_fl_s + ee_fl_s + no4_fl_s

    ocols["last_consumption_30m"] = cons_s.reindex(t30_idx).to_numpy()
    ocols["last_production_30m"] = prod_s.reindex(t30_idx).to_numpy()
    ocols["net_import_30m"] = net_import_s.reindex(t30_idx).to_numpy()
    ocols["cons_forecast_error_30m"] = cons_err_s.reindex(t30_idx).to_numpy()
    ocols["cons_forecast_error_mean_4q"] = cons_err_s.rolling(4, min_periods=1).mean().reindex(t30_idx).to_numpy()
    ocols["cons_forecast_error_mean_16q"] = cons_err_s.rolling(16, min_periods=1).mean().reindex(t30_idx).to_numpy()
    ocols["cons_forecast_error_change_1h"] = (
        cons_err_s.reindex(t30_idx).to_numpy()
        - cons_err_s.reindex(t30_idx - pd.Timedelta(hours=1)).to_numpy()
    )
    ocols["production_ramp_1h"] = (
        prod_s.reindex(t30_idx).to_numpy()
        - prod_s.reindex(t30_idx - pd.Timedelta(hours=1)).to_numpy()
    )
    ocols["renewable_error_proxy_change_1h"] = (
        ren_err_proxy_s.reindex(t30_idx).to_numpy()
        - ren_err_proxy_s.reindex(t30_idx - pd.Timedelta(hours=1)).to_numpy()
    )
    ocols["renewable_error_proxy_change_3h"] = (
        ren_err_proxy_s.reindex(t30_idx).to_numpy()
        - ren_err_proxy_s.reindex(t30_idx - pd.Timedelta(hours=3)).to_numpy()
    )

    ocols["fi_se1_flow_30m"] = se1_fl_s.reindex(t30_idx).to_numpy()
    ocols["fi_se3_flow_30m"] = se3_fl_s.reindex(t30_idx).to_numpy()
    ocols["fi_ee_flow_30m"] = ee_fl_s.reindex(t30_idx).to_numpy()
    ocols["fi_no4_flow_30m"] = no4_fl_s.reindex(t30_idx).to_numpy()
    ocols["total_cross_border_flow_30m"] = tot_flow_s.reindex(t30_idx).to_numpy()
    ocols["total_flow_ramp_1h"] = (
        tot_flow_s.reindex(t30_idx).to_numpy()
        - tot_flow_s.reindex(t30_idx - pd.Timedelta(hours=1)).to_numpy()
    )

    # 120m lag features (neighboring areas)
    se1_imb_s = store.val_wide["se1_imbalance_price_long"]
    se3_imb_s = store.val_wide["se3_imbalance_price_long"]
    ee_imb_s = (
        0.5 * store.val_wide["ee_imbalance_price_long"]
        + 0.5 * store.val_wide["ee_imbalance_price_short"]
    ).ffill(limit=8)
    spot_at_120m = spot_s.reindex(t120_idx).to_numpy()
    ocols["se1_imb_price_120m"] = se1_imb_s.reindex(t120_idx).to_numpy()
    ocols["se3_imb_price_120m"] = se3_imb_s.reindex(t120_idx).to_numpy()
    ocols["ee_imb_price_120m"] = ee_imb_s.reindex(t120_idx).to_numpy()
    ocols["se1_imb_spread_120m"] = ocols["se1_imb_price_120m"] - spot_at_120m
    ocols["se3_imb_spread_120m"] = ocols["se3_imb_price_120m"] - spot_at_120m
    ocols["ee_imb_spread_120m"] = ocols["ee_imb_price_120m"] - spot_at_120m

    # 2. Expand to (n_origins * horizon_steps) grid & add step/target-time features
    steps = np.arange(1, horizon_steps + 1, dtype=int)
    origin_rep = np.repeat(np.arange(n_origins), horizon_steps)
    step_tile = np.tile(steps, n_origins)

    scols: dict[str, Any] = {k: np.asarray(v)[origin_rep] for k, v in ocols.items()}
    scols["step"] = step_tile
    offsets = pd.to_timedelta((step_tile - 1) * STEP_MINUTES, unit="min")
    orig_expanded_idx = origins_idx[origin_rep]
    delivery_idx = orig_expanded_idx + offsets
    t60_expanded = t60_idx[origin_rep]
    scols["delivery_start_utc"] = delivery_idx

    scols["horizon_hours"] = (step_tile - 1) * 0.25
    scols["quarter_in_hour"] = (step_tile - 1) % 4
    scols["origin_hour_utc"] = orig_expanded_idx.hour.to_numpy()

    cal_hour = store.lookup_at("cal_hour_local", delivery_idx, orig_expanded_idx)
    cal_dow = store.lookup_at("cal_weekday_local", delivery_idx, orig_expanded_idx)
    scols["cal_hour_local"] = cal_hour
    scols["cal_weekday_local"] = cal_dow
    scols["cal_is_holiday_fi"] = store.lookup_at("cal_is_holiday_fi", delivery_idx, orig_expanded_idx)
    scols["cal_is_dst"] = store.lookup_at("cal_is_dst", delivery_idx, orig_expanded_idx)
    scols["is_weekend"] = (cal_dow >= 6.0).astype(float)

    hour_frac = cal_hour + (delivery_idx.minute.to_numpy() / 60.0)
    scols["sin_hour"] = np.sin(2.0 * np.pi * hour_frac / 24.0)
    scols["cos_hour"] = np.cos(2.0 * np.pi * hour_frac / 24.0)
    scols["sin_dow"] = np.sin(2.0 * np.pi * cal_dow / 7.0)
    scols["cos_dow"] = np.cos(2.0 * np.pi * cal_dow / 7.0)

    spot_deliv = store.lookup_at("fi_spot_price", delivery_idx, orig_expanded_idx)
    wind_deliv = store.lookup_at("fi_wind_fc_da", delivery_idx, orig_expanded_idx)
    solar_deliv = store.lookup_at("fi_solar_fc_da", delivery_idx, orig_expanded_idx)
    cons_deliv = store.lookup_at("fi_consumption_fc_da", delivery_idx, orig_expanded_idx)
    fi_ee_cap = store.lookup_at("fi_ee_capacity_da", delivery_idx, orig_expanded_idx)
    ee_fi_cap = store.lookup_at("ee_fi_capacity_da", delivery_idx, orig_expanded_idx)

    ren_deliv = wind_deliv + solar_deliv
    res_deliv = cons_deliv - ren_deliv

    scols["spot_at_delivery"] = spot_deliv
    scols["wind_da_at_delivery"] = wind_deliv
    scols["solar_da_at_delivery"] = solar_deliv
    scols["cons_da_at_delivery"] = cons_deliv
    scols["fi_ee_cap_at_delivery"] = fi_ee_cap
    scols["ee_fi_cap_at_delivery"] = ee_fi_cap
    scols["renewable_da_at_delivery"] = ren_deliv
    scols["residual_load_da_at_delivery"] = res_deliv
    scols["wind_share_da_at_delivery"] = wind_deliv / (cons_deliv + 1.0)
    scols["net_ee_cap_at_delivery"] = ee_fi_cap - fi_ee_cap

    scols["se1_load_da_at_delivery"] = store.lookup_at("se1_load_fc_da", delivery_idx, orig_expanded_idx)
    scols["se3_load_da_at_delivery"] = store.lookup_at("se3_load_fc_da", delivery_idx, orig_expanded_idx)
    scols["ee_load_da_at_delivery"] = store.lookup_at("ee_load_fc_da", delivery_idx, orig_expanded_idx)
    # Day-ahead ramps leading up to delivery_start_utc
    deliv_m1h = delivery_idx - pd.Timedelta(hours=1)
    deliv_m3h = delivery_idx - pd.Timedelta(hours=3)
    spot_m1h = store.lookup_at("fi_spot_price", deliv_m1h, orig_expanded_idx)
    spot_m3h = store.lookup_at("fi_spot_price", deliv_m3h, orig_expanded_idx)
    wind_m1h = store.lookup_at("fi_wind_fc_da", deliv_m1h, orig_expanded_idx)
    wind_m3h = store.lookup_at("fi_wind_fc_da", deliv_m3h, orig_expanded_idx)
    solar_m1h = store.lookup_at("fi_solar_fc_da", deliv_m1h, orig_expanded_idx)
    solar_m3h = store.lookup_at("fi_solar_fc_da", deliv_m3h, orig_expanded_idx)
    cons_m1h = store.lookup_at("fi_consumption_fc_da", deliv_m1h, orig_expanded_idx)
    cons_m3h = store.lookup_at("fi_consumption_fc_da", deliv_m3h, orig_expanded_idx)

    res_m1h = cons_m1h - (wind_m1h + solar_m1h)
    res_m3h = cons_m3h - (wind_m3h + solar_m3h)

    scols["spot_ramp_1h"] = spot_deliv - spot_m1h
    scols["spot_ramp_3h"] = spot_deliv - spot_m3h
    deliv_p1h = delivery_idx + pd.Timedelta(hours=1)
    spot_p1h = store.lookup_at("fi_spot_price", deliv_p1h, orig_expanded_idx)
    scols["spot_lead_1h"] = np.where(~np.isnan(spot_p1h), spot_p1h - spot_deliv, 0.0)
    scols["wind_da_ramp_1h"] = wind_deliv - wind_m1h
    scols["wind_da_ramp_3h"] = wind_deliv - wind_m3h
    scols["cons_da_ramp_1h"] = cons_deliv - cons_m1h
    scols["residual_load_ramp_1h"] = res_deliv - res_m1h
    scols["residual_load_ramp_3h"] = res_deliv - res_m3h

    wind_at_t60 = store.lookup_at("fi_wind_fc_da", t60_expanded, orig_expanded_idx)
    cons_at_t60 = store.lookup_at("fi_consumption_fc_da", t60_expanded, orig_expanded_idx)
    solar_at_t60 = store.lookup_at("fi_solar_fc_da", t60_expanded, orig_expanded_idx)
    res_at_t60 = cons_at_t60 - (wind_at_t60 + solar_at_t60)

    scols["spot_diff_from_60m"] = spot_deliv - scols["last_spot_at_60m"]
    scols["wind_da_diff_from_60m"] = wind_deliv - wind_at_t60
    scols["cons_da_diff_from_60m"] = cons_deliv - cons_at_t60
    scols["residual_load_diff_from_60m"] = res_deliv - res_at_t60

    spot_mat = spot_deliv.reshape(n_origins, horizon_steps)
    res_mat = res_deliv.reshape(n_origins, horizon_steps)
    spot_h_mean = np.nanmean(spot_mat, axis=1, keepdims=True)
    spot_h_std = np.nanstd(spot_mat, axis=1, keepdims=True) + 1.0
    res_h_mean = np.nanmean(res_mat, axis=1, keepdims=True)
    res_h_std = np.nanstd(res_mat, axis=1, keepdims=True) + 1.0

    scols["spot_zscore_in_horizon"] = ((spot_mat - spot_h_mean) / spot_h_std).ravel()
    scols["residual_load_zscore_in_horizon"] = ((res_mat - res_h_mean) / res_h_std).ravel()

    deliv_m24h = delivery_idx - pd.Timedelta(hours=24)
    deliv_m168h = delivery_idx - pd.Timedelta(hours=168)
    imb_24h = store.lookup_at("fi_imbalance_price", deliv_m24h, orig_expanded_idx)
    spot_24h = store.lookup_at("fi_spot_price", deliv_m24h, orig_expanded_idx)
    imb_168h = store.lookup_at("fi_imbalance_price", deliv_m168h, orig_expanded_idx)
    spot_168h = store.lookup_at("fi_spot_price", deliv_m168h, orig_expanded_idx)

    scols["imb_price_lag_24h"] = imb_24h
    scols["imb_spread_lag_24h"] = imb_24h - spot_24h
    scols["imb_spread_lag_168h"] = imb_168h - spot_168h
    scols["mfrr_dir_lag_24h"] = store.lookup_at("fi_mfrr_direction", deliv_m24h, orig_expanded_idx)

    step_arr = step_tile.astype(float)
    scols["decay_spread_short"] = scols["last_imbalance_spread"] * np.exp(-(step_arr - 1.0) / 6.0)
    scols["decay_spread_medium"] = scols["imb_spread_mean_4q"] * np.exp(-(step_arr - 1.0) / 14.0)
    scols["decay_mfrr_dir"] = scols["mfrr_dir_mean_4q"] * np.exp(-(step_arr - 1.0) / 8.0)
    meta_cols = {"forecast_origin_utc", "delivery_start_utc", "step", "data_cutoff_utc"}
    feature_cols = ["step"] + [c for c in scols.keys() if c not in meta_cols]

    if include_targets:
        t_imb = store.val_wide["fi_imbalance_price"].reindex(delivery_idx).to_numpy(dtype=float)
        t_spot = store.val_wide["fi_spot_price"].reindex(delivery_idx).to_numpy(dtype=float)
        t_dir = store.val_wide["fi_mfrr_direction"].reindex(delivery_idx).to_numpy(dtype=float)
        t_fu_s = store.fu_wide["fi_imbalance_price"].reindex(delivery_idx)

        if target_cutoff_utc is not None:
            cutoff_ts = to_utc(target_cutoff_utc)
            t_fu_idx = pd.DatetimeIndex(t_fu_s)
            valid_target = (~t_fu_idx.isna()) & (t_fu_idx <= cutoff_ts)
            t_imb = np.where(valid_target, t_imb, np.nan)
            t_dir = np.where(valid_target, t_dir, np.nan)

        scols["target_imbalance_price"] = t_imb
        scols["target_spot_price"] = t_spot
        scols["target_spread"] = t_imb - t_spot
        scols["target_direction"] = t_dir
        scols["target_spike_high"] = np.where(
            np.isfinite(t_imb) & np.isfinite(t_spot),
            (t_imb > t_spot + SPIKE_MARGIN).astype(float),
            np.nan,
        )
        scols["target_spike_low"] = np.where(
            np.isfinite(t_imb) & np.isfinite(t_spot),
            (t_imb < t_spot - SPIKE_MARGIN).astype(float),
            np.nan,
        )

    df = pd.DataFrame(scols)
    return df, feature_cols
