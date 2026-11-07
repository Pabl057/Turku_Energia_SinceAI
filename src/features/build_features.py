"""Leak-free feature engineering pipeline for Turku Energia Imbalance Price Forecasting.

Strictly enforces RULE #1: For any forecast origin t, ONLY rows with
first_usable_origin_utc <= t are ever accessed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.loader import rows_available_at, to_utc

HORIZON_STEPS = 40
STEP_MINUTES = 15


class FeatureBuilder:
    """Fast, strictly leak-free feature builder indexed by timestamp_utc and first_usable_origin_utc."""

    def __init__(self, long_data: pd.DataFrame):
        # Verify no duplicate (series_id, timestamp_utc)
        self.long_data = long_data
        self.values_wide = long_data.pivot(index="timestamp_utc", columns="series_id", values="value").sort_index()
        self.usable_wide = long_data.pivot(
            index="timestamp_utc", columns="series_id", values="first_usable_origin_utc"
        ).sort_index()
        self.index = self.values_wide.index

        # Precompute derived series on the wide grid (still masked by usable_wide at query time!)
        # Note: a derived value at timestamp tau is only usable when ALL constituent series at tau have first_usable_origin_utc <= origin!
        self._init_series_arrays()

    def _init_series_arrays(self) -> None:
        """Cache numpy arrays and usability timestamps (in int64 ns) for fast leak-checked slicing."""
        self.ts_ns = self.index.asi8
        self.series_val: dict[str, np.ndarray] = {}
        self.series_fu_ns: dict[str, np.ndarray] = {}

        for col in self.values_wide.columns:
            self.series_val[col] = self.values_wide[col].to_numpy(dtype=np.float64, copy=True)
            fu = self.usable_wide[col]
            # Missing usability timestamps -> max int64 (never usable)
            fu_ns = np.where(fu.notna(), fu.astype("int64"), np.iinfo(np.int64).max)
            self.series_fu_ns[col] = fu_ns

    def usable_series_up_to(self, col: str, origin_ns: int, max_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (timestamps_ns, values) of series `col` up to `max_idx` where first_usable_origin_utc <= origin_ns."""
        vals = self.series_val[col][:max_idx]
        fu = self.series_fu_ns[col][:max_idx]
        mask = (fu <= origin_ns) & (~np.isnan(vals))
        return self.ts_ns[:max_idx][mask], vals[mask]

    def value_at_target(self, col: str, target_idx: np.ndarray, origin_ns: int) -> np.ndarray:
        """Return values of `col` at `target_idx` ONLY IF first_usable_origin_utc <= origin_ns, else NaN."""
        out = np.full(len(target_idx), np.nan, dtype=np.float64)
        valid_idx = (target_idx >= 0) & (target_idx < len(self.ts_ns))
        idx_clamped = np.clip(target_idx, 0, len(self.ts_ns) - 1)
        vals = self.series_val[col][idx_clamped]
        fu = self.series_fu_ns[col][idx_clamped]
        usable = valid_idx & (fu <= origin_ns)
        out[usable] = vals[usable]
        return out

    def build_for_origins(
        self,
        origins: pd.DatetimeIndex,
        include_targets: bool = False,
        spike_margin: float = 50.0,
    ) -> pd.DataFrame:
        """Build the 40-step feature matrix X (and targets y if requested) for a sequence of forecast origins."""
        rows_list = []
        step_arange = np.arange(1, HORIZON_STEPS + 1, dtype=np.int32)
        step_offsets_ns = (step_arange - 1).astype(np.int64) * (STEP_MINUTES * 60 * 1_000_000_000)
        quarter_ns = STEP_MINUTES * 60 * 1_000_000_000
        day_ns = 96 * quarter_ns

        for origin in origins:
            origin_ts = to_utc(origin)
            origin_ns = origin_ts.value

            # Index of origin in our 15-min master grid
            origin_idx = int(np.searchsorted(self.ts_ns, origin_ns, side="right"))

            # Target timestamps and indices for steps 1..40
            delivery_ns = origin_ns + step_offsets_ns
            target_idx = np.searchsorted(self.ts_ns, delivery_ns)
            # Verify exact match
            exact_match = (target_idx < len(self.ts_ns)) & (self.ts_ns[np.clip(target_idx, 0, len(self.ts_ns) - 1)] == delivery_ns)
            target_idx = np.where(exact_match, target_idx, -1)

            # Same quarter-hour 24h earlier (always usable for D-1 forecasts and spot)
            prev_day_idx = np.where(target_idx >= 96, target_idx - 96, -1)
            prev_1h_idx = np.where(target_idx >= 4, target_idx - 4, -1)
            prev_2h_idx = np.where(target_idx >= 8, target_idx - 8, -1)
            prev_3h_idx = np.where(target_idx >= 12, target_idx - 12, -1)
            next_1h_idx = np.where((target_idx >= 0) & (target_idx + 4 < len(self.ts_ns)), target_idx + 4, -1)

            # =========================================================================
            # 1. KNOWN FUTURE FEATURES AT TARGET DELIVERY TIME (strictly usable at t)
            # =========================================================================
            spot_target = self.value_at_target("fi_spot_price", target_idx, origin_ns)
            spot_prev_day = self.value_at_target("fi_spot_price", prev_day_idx, origin_ns)
            spot_known = (~np.isnan(spot_target)).astype(np.float64)
            spot_eff = np.where(~np.isnan(spot_target), spot_target, spot_prev_day)

            spot_prev_1h = self.value_at_target("fi_spot_price", prev_1h_idx, origin_ns)
            spot_prev_2h = self.value_at_target("fi_spot_price", prev_2h_idx, origin_ns)
            spot_next_1h = self.value_at_target("fi_spot_price", next_1h_idx, origin_ns)
            spot_ramp_1h = spot_eff - np.where(~np.isnan(spot_prev_1h), spot_prev_1h, spot_eff)
            spot_ramp_2h = spot_eff - np.where(~np.isnan(spot_prev_2h), spot_prev_2h, spot_eff)
            spot_lead_1h = np.where(~np.isnan(spot_next_1h), spot_next_1h, spot_eff) - spot_eff

            wind_fc = self.value_at_target("fi_wind_fc_da", target_idx, origin_ns)
            wind_fc_prev_day = self.value_at_target("fi_wind_fc_da", prev_day_idx, origin_ns)
            wind_fc_eff = np.where(~np.isnan(wind_fc), wind_fc, wind_fc_prev_day)

            solar_fc = self.value_at_target("fi_solar_fc_da", target_idx, origin_ns)
            solar_fc_prev_day = self.value_at_target("fi_solar_fc_da", prev_day_idx, origin_ns)
            solar_fc_eff = np.where(~np.isnan(solar_fc), solar_fc, solar_fc_prev_day)

            cons_fc = self.value_at_target("fi_consumption_fc_da", target_idx, origin_ns)
            cons_fc_prev_day = self.value_at_target("fi_consumption_fc_da", prev_day_idx, origin_ns)
            cons_fc_eff = np.where(~np.isnan(cons_fc), cons_fc, cons_fc_prev_day)

            # Previous 1h & 3h forecast values for ramps
            wind_fc_1h = self.value_at_target("fi_wind_fc_da", prev_1h_idx, origin_ns)
            wind_fc_3h = self.value_at_target("fi_wind_fc_da", prev_3h_idx, origin_ns)
            cons_fc_1h = self.value_at_target("fi_consumption_fc_da", prev_1h_idx, origin_ns)
            cons_fc_3h = self.value_at_target("fi_consumption_fc_da", prev_3h_idx, origin_ns)

            res_load_fc = cons_fc_eff - wind_fc_eff - solar_fc_eff
            res_load_1h = (
                np.where(~np.isnan(cons_fc_1h), cons_fc_1h, cons_fc_eff)
                - np.where(~np.isnan(wind_fc_1h), wind_fc_1h, wind_fc_eff)
                - solar_fc_eff
            )
            res_load_3h = (
                np.where(~np.isnan(cons_fc_3h), cons_fc_3h, cons_fc_eff)
                - np.where(~np.isnan(wind_fc_3h), wind_fc_3h, wind_fc_eff)
                - solar_fc_eff
            )
            res_load_ramp_1h = res_load_fc - res_load_1h
            res_load_ramp_3h = res_load_fc - res_load_3h
            wind_ramp_1h = wind_fc_eff - np.where(~np.isnan(wind_fc_1h), wind_fc_1h, wind_fc_eff)
            cons_ramp_1h = cons_fc_eff - np.where(~np.isnan(cons_fc_1h), cons_fc_1h, cons_fc_eff)
            wind_share_fc = wind_fc_eff / np.maximum(cons_fc_eff, 100.0)

            # Transfer capacities & Neighbour DA load forecasts at target
            fi_ee_cap = self.value_at_target("fi_ee_capacity_da", target_idx, origin_ns)
            ee_fi_cap = self.value_at_target("ee_fi_capacity_da", target_idx, origin_ns)
            se1_load_fc = self.value_at_target("se1_load_fc_da", target_idx, origin_ns)
            se3_load_fc = self.value_at_target("se3_load_fc_da", target_idx, origin_ns)
            ee_load_fc = self.value_at_target("ee_load_fc_da", target_idx, origin_ns)

            # Calendar features at target
            cal_hour = self.value_at_target("cal_hour_local", target_idx, origin_ns)
            cal_weekday = self.value_at_target("cal_weekday_local", target_idx, origin_ns)
            cal_holiday = self.value_at_target("cal_is_holiday_fi", target_idx, origin_ns)
            cal_dst = self.value_at_target("cal_is_dst", target_idx, origin_ns)

            sin_hour = np.sin(2.0 * np.pi * cal_hour / 24.0)
            cos_hour = np.cos(2.0 * np.pi * cal_hour / 24.0)
            delivery_dt = pd.to_datetime(delivery_ns, utc=True)
            quarter_in_hour = (delivery_dt.minute // 15).to_numpy(dtype=np.float64)
            origin_hour_utc = float(origin_ts.hour)

            # Daily spot context (from usable spot prices in [t - 24h, t + 24h])
            lo_spot = max(0, origin_idx - 96)
            hi_spot = min(len(self.ts_ns), origin_idx + 96)
            spot_window_vals = self.series_val["fi_spot_price"][lo_spot:hi_spot]
            spot_window_fu = self.series_fu_ns["fi_spot_price"][lo_spot:hi_spot]
            usable_spots = spot_window_vals[(spot_window_fu <= origin_ns) & (~np.isnan(spot_window_vals))]
            if len(usable_spots) > 0:
                spot_24h_mean = float(np.mean(usable_spots))
                spot_24h_std = float(np.std(usable_spots))
                spot_24h_max = float(np.max(usable_spots))
                spot_24h_min = float(np.min(usable_spots))
            else:
                spot_24h_mean = spot_24h_std = spot_24h_max = spot_24h_min = np.nan

            spot_vs_24h_mean = spot_eff - spot_24h_mean
            spot_vs_24h_max = spot_eff - spot_24h_max

            # Spot at origin t
            origin_exact_idx = np.array([origin_idx - 1 if origin_idx > 0 else 0])
            spot_at_origin_arr = self.value_at_target("fi_spot_price", origin_exact_idx, origin_ns)
            spot_at_origin = float(spot_at_origin_arr[0]) if not np.isnan(spot_at_origin_arr[0]) else spot_24h_mean

            # =========================================================================
            # 2. REAL-TIME GRID STATE & RESIDUAL/DRIFT LAGS BEFORE ORIGIN t
            # =========================================================================
            # Imbalance price history usable at origin t (60 min lag -> latest is t - 75m)
            imb_ts, imb_vals = self.usable_series_up_to("fi_imbalance_price", origin_ns, origin_idx)
            cutoff_ns = int(imb_ts[-1]) if len(imb_ts) > 0 else origin_ns

            def _tail_stat(arr: np.ndarray, k: int, fn, default=np.nan) -> float:
                if len(arr) == 0:
                    return default
                sub = arr[-k:]
                return float(fn(sub))

            def _lag(arr: np.ndarray, lag_idx: int, default=np.nan) -> float:
                if len(arr) >= lag_idx:
                    return float(arr[-lag_idx])
                return float(arr[0]) if len(arr) > 0 else default

            imb_last = _lag(imb_vals, 1)
            imb_lag_2 = _lag(imb_vals, 2)
            imb_lag_3 = _lag(imb_vals, 3)
            imb_lag_4 = _lag(imb_vals, 4)
            imb_lag_5 = _lag(imb_vals, 5)
            imb_lag_8 = _lag(imb_vals, 8)
            imb_lag_16 = _lag(imb_vals, 16)

            imb_mean_4 = _tail_stat(imb_vals, 4, np.mean)
            imb_std_4 = _tail_stat(imb_vals, 4, np.std)
            imb_min_4 = _tail_stat(imb_vals, 4, np.min)
            imb_max_4 = _tail_stat(imb_vals, 4, np.max)

            imb_mean_12 = _tail_stat(imb_vals, 12, np.mean)
            imb_std_12 = _tail_stat(imb_vals, 12, np.std)
            imb_min_12 = _tail_stat(imb_vals, 12, np.min)
            imb_max_12 = _tail_stat(imb_vals, 12, np.max)
            imb_mean_24 = _tail_stat(imb_vals, 24, np.mean)
            imb_mean_96 = _tail_stat(imb_vals, 96, np.mean)

            # Realized Spread (Imbalance Price - Spot Price) at usable timestamps before t (HEBEL 1!)
            if len(imb_ts) > 0:
                recent_ts = imb_ts[-24:]
                recent_imb = imb_vals[-24:]
                recent_idx = np.searchsorted(self.ts_ns, recent_ts)
                recent_spot = self.value_at_target("fi_spot_price", recent_idx, origin_ns)
                spread_vals = recent_imb - recent_spot
                spread_vals = spread_vals[~np.isnan(spread_vals)]
            else:
                spread_vals = np.array([], dtype=np.float64)

            spread_last = _lag(spread_vals, 1, 0.0)
            spread_lag_2 = _lag(spread_vals, 2, 0.0)
            spread_lag_3 = _lag(spread_vals, 3, 0.0)
            spread_lag_4 = _lag(spread_vals, 4, 0.0)
            spread_lag_5 = _lag(spread_vals, 5, 0.0)
            spread_lag_8 = _lag(spread_vals, 8, 0.0)

            spread_mean_4 = _tail_stat(spread_vals, 4, np.mean, 0.0)
            spread_std_4 = _tail_stat(spread_vals, 4, np.std, 0.0)
            spread_mean_12 = _tail_stat(spread_vals, 12, np.mean, 0.0)
            spread_max_12 = _tail_stat(spread_vals, 12, np.max, 0.0)
            spread_min_12 = _tail_stat(spread_vals, 12, np.min, 0.0)
            spread_mean_24 = _tail_stat(spread_vals, 24, np.mean, 0.0)

            # Drift of realized spread over the last 30m, 1h, 2h
            spread_drift_30m = spread_last - spread_lag_3
            spread_drift_1h = spread_last - spread_lag_5
            spread_drift_2h = spread_last - spread_lag_8

            # Same delivery quarter-hour 24h earlier imbalance price & spread (always usable since lag <= 2h)
            imb_prev_day = self.value_at_target("fi_imbalance_price", prev_day_idx, origin_ns)
            spread_prev_day = imb_prev_day - spot_prev_day

            # mFRR direction & volumes usable at origin t
            _, dir_vals = self.usable_series_up_to("fi_mfrr_direction", origin_ns, origin_idx)
            dir_last = _lag(dir_vals, 1, 0.0)
            dir_lag_2 = _lag(dir_vals, 2, 0.0)
            dir_lag_4 = _lag(dir_vals, 4, 0.0)
            dir_mean_4 = _tail_stat(dir_vals, 4, np.mean, 0.0)
            dir_mean_12 = _tail_stat(dir_vals, 12, np.mean, 0.0)
            dir_mean_24 = _tail_stat(dir_vals, 24, np.mean, 0.0)

            dir_up_share_4 = float(np.mean(dir_vals[-4:] > 0)) if len(dir_vals) > 0 else 0.35
            dir_down_share_4 = float(np.mean(dir_vals[-4:] < 0)) if len(dir_vals) > 0 else 0.64
            dir_up_share_12 = float(np.mean(dir_vals[-12:] > 0)) if len(dir_vals) > 0 else 0.35
            dir_down_share_12 = float(np.mean(dir_vals[-12:] < 0)) if len(dir_vals) > 0 else 0.64
            dir_up_share_24 = float(np.mean(dir_vals[-24:] > 0)) if len(dir_vals) > 0 else 0.35
            dir_down_share_24 = float(np.mean(dir_vals[-24:] < 0)) if len(dir_vals) > 0 else 0.64

            # Same quarter-hour yesterday mFRR direction
            dir_prev_day = self.value_at_target("fi_mfrr_direction", prev_day_idx, origin_ns)

            _, mfrr_up_vol = self.usable_series_up_to("fi_mfrr_up_activated", origin_ns, origin_idx)
            _, mfrr_down_vol = self.usable_series_up_to("fi_mfrr_down_activated", origin_ns, origin_idx)
            _, mfrr_up_prc = self.usable_series_up_to("fi_mfrr_up_price", origin_ns, origin_idx)
            _, mfrr_down_prc = self.usable_series_up_to("fi_mfrr_down_price", origin_ns, origin_idx)

            mfrr_up_vol_last = _lag(mfrr_up_vol, 1, 0.0)
            mfrr_down_vol_last = _lag(mfrr_down_vol, 1, 0.0)
            # Note: Fingrid mfrr_down_activated is negative or positive? Let's preserve raw & compute net
            mfrr_up_vol_mean_4 = _tail_stat(mfrr_up_vol, 4, np.mean, 0.0)
            mfrr_down_vol_mean_4 = _tail_stat(mfrr_down_vol, 4, np.mean, 0.0)
            mfrr_up_prc_last = _lag(mfrr_up_prc, 1, spot_at_origin)
            mfrr_down_prc_last = _lag(mfrr_down_prc, 1, spot_at_origin)
            mfrr_up_prc_spread = mfrr_up_prc_last - spot_at_origin
            mfrr_down_prc_spread = mfrr_down_prc_last - spot_at_origin

            # aFRR prices usable at origin t
            _, afrr_up_prc = self.usable_series_up_to("fi_afrr_up_price", origin_ns, origin_idx)
            _, afrr_down_prc = self.usable_series_up_to("fi_afrr_down_price", origin_ns, origin_idx)
            afrr_up_prc_last = _lag(afrr_up_prc, 1, spot_at_origin)
            afrr_down_prc_last = _lag(afrr_down_prc, 1, spot_at_origin)
            afrr_up_spread = afrr_up_prc_last - spot_at_origin
            afrr_down_spread = afrr_down_prc_last - spot_at_origin

            # Real-time Consumption & Production (30 min lag -> 30m fresher than imbalance price!)
            cons_ts, cons_vals = self.usable_series_up_to("fi_consumption", origin_ns, origin_idx)
            _, prod_vals = self.usable_series_up_to("fi_production", origin_ns, origin_idx)

            cons_last = _lag(cons_vals, 1)
            prod_last = _lag(prod_vals, 1)
            net_deficit_last = cons_last - prod_last
            net_deficit_lag_4 = _lag(cons_vals, 5) - _lag(prod_vals, 5)
            net_deficit_drift_1h = net_deficit_last - net_deficit_lag_4

            # Realized Consumption Forecast Error (HEBEL 1: True consumption - Fingrid DA forecast)
            if len(cons_ts) > 0:
                c_ts = cons_ts[-12:]
                c_val = cons_vals[-12:]
                c_idx = np.searchsorted(self.ts_ns, c_ts)
                c_fc = self.value_at_target("fi_consumption_fc_da", c_idx, origin_ns)
                c_err = c_val - c_fc
                c_err = c_err[~np.isnan(c_err)]
            else:
                c_err = np.array([], dtype=np.float64)

            cons_err_last = _lag(c_err, 1, 0.0)
            cons_err_mean_4 = _tail_stat(c_err, 4, np.mean, 0.0)
            cons_err_drift_1h = cons_err_last - _lag(c_err, 5, 0.0)

            # Cross-border flows (30 min lag)
            _, flow_se1 = self.usable_series_up_to("fi_se1_flow", origin_ns, origin_idx)
            _, flow_se3 = self.usable_series_up_to("fi_se3_flow", origin_ns, origin_idx)
            _, flow_ee = self.usable_series_up_to("fi_ee_flow", origin_ns, origin_idx)
            _, flow_no4 = self.usable_series_up_to("fi_no4_flow", origin_ns, origin_idx)

            flow_se1_last = _lag(flow_se1, 1, 0.0)
            flow_se3_last = _lag(flow_se3, 1, 0.0)
            flow_ee_last = _lag(flow_ee, 1, 0.0)
            flow_no4_last = _lag(flow_no4, 1, 0.0)
            total_flow_last = flow_se1_last + flow_se3_last + flow_ee_last + flow_no4_last
            total_flow_lag_4 = (
                _lag(flow_se1, 5, 0.0)
                + _lag(flow_se3, 5, 0.0)
                + _lag(flow_ee, 5, 0.0)
                + _lag(flow_no4, 5, 0.0)
            )
            total_flow_drift_1h = total_flow_last - total_flow_lag_4

            # Neighbour imbalance prices (120 min lag)
            _, se1_imb = self.usable_series_up_to("se1_imbalance_price_long", origin_ns, origin_idx)
            _, se3_imb = self.usable_series_up_to("se3_imbalance_price_long", origin_ns, origin_idx)
            _, ee_imb = self.usable_series_up_to("ee_imbalance_price_long", origin_ns, origin_idx)
            se1_imb_last = _lag(se1_imb, 1, spot_at_origin)
            se3_imb_last = _lag(se3_imb, 1, spot_at_origin)
            ee_imb_last = _lag(ee_imb, 1, spot_at_origin)

            # Horizon interaction features (decaying current shock into future known spot price)
            decay_fast = np.exp(-(step_arange - 1) / 4.0)
            decay_med = np.exp(-(step_arange - 1) / 10.0)
            blended_spot_imb_fast = spot_eff + spread_last * decay_fast
            blended_spot_imb_med = spot_eff + spread_mean_4 * decay_med
            imb_last_minus_spot_target = imb_last - spot_eff
            spot_target_minus_spot_origin = spot_eff - spot_at_origin

            df_origin = pd.DataFrame(
                {
                    "forecast_origin_utc": origin_ts,
                    "delivery_start_utc": delivery_dt,
                    "data_cutoff_utc": pd.Timestamp(cutoff_ns, tz="UTC"),
                    "step": step_arange,
                    # Known future features at delivery_start_utc
                    "spot_target": spot_target,
                    "spot_prev_day": spot_prev_day,
                    "spot_known": spot_known,
                    "spot_eff": spot_eff,
                    "spot_ramp_1h": spot_ramp_1h,
                    "spot_ramp_2h": spot_ramp_2h,
                    "spot_lead_1h": spot_lead_1h,
                    "spot_vs_24h_mean": spot_vs_24h_mean,
                    "spot_vs_24h_max": spot_vs_24h_max,
                    "spot_24h_std": spot_24h_std,
                    "wind_fc_eff": wind_fc_eff,
                    "solar_fc_eff": solar_fc_eff,
                    "cons_fc_eff": cons_fc_eff,
                    "res_load_fc": res_load_fc,
                    "res_load_ramp_1h": res_load_ramp_1h,
                    "res_load_ramp_3h": res_load_ramp_3h,
                    "wind_ramp_1h": wind_ramp_1h,
                    "cons_ramp_1h": cons_ramp_1h,
                    "wind_share_fc": wind_share_fc,
                    "fi_ee_cap": fi_ee_cap,
                    "ee_fi_cap": ee_fi_cap,
                    "se1_load_fc": se1_load_fc,
                    "se3_load_fc": se3_load_fc,
                    "ee_load_fc": ee_load_fc,
                    "cal_hour": cal_hour,
                    "cal_weekday": cal_weekday,
                    "cal_holiday": cal_holiday,
                    "cal_dst": cal_dst,
                    "sin_hour": sin_hour,
                    "cos_hour": cos_hour,
                    "quarter_in_hour": quarter_in_hour,
                    "origin_hour_utc": origin_hour_utc,
                    # Real-time lag & residual drift features at origin t
                    "imb_last": imb_last,
                    "imb_lag_2": imb_lag_2,
                    "imb_lag_3": imb_lag_3,
                    "imb_lag_4": imb_lag_4,
                    "imb_lag_8": imb_lag_8,
                    "imb_lag_16": imb_lag_16,
                    "imb_mean_4": imb_mean_4,
                    "imb_std_4": imb_std_4,
                    "imb_min_4": imb_min_4,
                    "imb_max_4": imb_max_4,
                    "imb_mean_12": imb_mean_12,
                    "imb_std_12": imb_std_12,
                    "imb_min_12": imb_min_12,
                    "imb_max_12": imb_max_12,
                    "imb_mean_24": imb_mean_24,
                    "imb_mean_96": imb_mean_96,
                    "imb_prev_day": imb_prev_day,
                    "spread_last": spread_last,
                    "spread_lag_2": spread_lag_2,
                    "spread_lag_3": spread_lag_3,
                    "spread_lag_4": spread_lag_4,
                    "spread_lag_8": spread_lag_8,
                    "spread_mean_4": spread_mean_4,
                    "spread_std_4": spread_std_4,
                    "spread_mean_12": spread_mean_12,
                    "spread_max_12": spread_max_12,
                    "spread_min_12": spread_min_12,
                    "spread_mean_24": spread_mean_24,
                    "spread_drift_30m": spread_drift_30m,
                    "spread_drift_1h": spread_drift_1h,
                    "spread_drift_2h": spread_drift_2h,
                    "spread_prev_day": spread_prev_day,
                    "dir_last": dir_last,
                    "dir_lag_2": dir_lag_2,
                    "dir_lag_4": dir_lag_4,
                    "dir_mean_4": dir_mean_4,
                    "dir_mean_12": dir_mean_12,
                    "dir_mean_24": dir_mean_24,
                    "dir_up_share_4": dir_up_share_4,
                    "dir_down_share_4": dir_down_share_4,
                    "dir_up_share_12": dir_up_share_12,
                    "dir_down_share_12": dir_down_share_12,
                    "dir_up_share_24": dir_up_share_24,
                    "dir_down_share_24": dir_down_share_24,
                    "dir_prev_day": dir_prev_day,
                    "mfrr_up_vol_last": mfrr_up_vol_last,
                    "mfrr_down_vol_last": mfrr_down_vol_last,
                    "mfrr_up_vol_mean_4": mfrr_up_vol_mean_4,
                    "mfrr_down_vol_mean_4": mfrr_down_vol_mean_4,
                    "mfrr_up_prc_spread": mfrr_up_prc_spread,
                    "mfrr_down_prc_spread": mfrr_down_prc_spread,
                    "afrr_up_spread": afrr_up_spread,
                    "afrr_down_spread": afrr_down_spread,
                    "cons_last": cons_last,
                    "prod_last": prod_last,
                    "net_deficit_last": net_deficit_last,
                    "net_deficit_drift_1h": net_deficit_drift_1h,
                    "cons_err_last": cons_err_last,
                    "cons_err_mean_4": cons_err_mean_4,
                    "cons_err_drift_1h": cons_err_drift_1h,
                    "flow_se1_last": flow_se1_last,
                    "flow_se3_last": flow_se3_last,
                    "flow_ee_last": flow_ee_last,
                    "flow_no4_last": flow_no4_last,
                    "total_flow_last": total_flow_last,
                    "total_flow_drift_1h": total_flow_drift_1h,
                    "se1_imb_last": se1_imb_last,
                    "se3_imb_last": se3_imb_last,
                    "ee_imb_last": ee_imb_last,
                    # Interaction / Decay bridges
                    "blended_spot_imb_fast": blended_spot_imb_fast,
                    "blended_spot_imb_med": blended_spot_imb_med,
                    "imb_last_minus_spot_target": imb_last_minus_spot_target,
                    "spot_target_minus_spot_origin": spot_target_minus_spot_origin,
                }
            )

            if include_targets:
                # For training: target values at target_idx
                # NOTE: When training at a cutoff T_train_end, the caller MUST filter rows where
                # target_first_usable_utc <= T_train_end so no target from after T_train_end is ever used!
                idx_clamped = np.clip(target_idx, 0, len(self.ts_ns) - 1)
                valid_t = (target_idx >= 0) & (target_idx < len(self.ts_ns))

                y_imb = np.where(valid_t, self.series_val["fi_imbalance_price"][idx_clamped], np.nan)
                y_spot = np.where(valid_t, self.series_val["fi_spot_price"][idx_clamped], np.nan)
                y_dir = np.where(valid_t, self.series_val["fi_mfrr_direction"][idx_clamped], np.nan)

                fu_imb = np.where(valid_t, self.series_fu_ns["fi_imbalance_price"][idx_clamped], np.iinfo(np.int64).max)
                fu_dir = np.where(valid_t, self.series_fu_ns["fi_mfrr_direction"][idx_clamped], np.iinfo(np.int64).max)
                fu_target_ns = np.maximum(fu_imb, fu_dir)

                df_origin["y_imbalance_price"] = y_imb
                df_origin["y_spot_price"] = y_spot
                df_origin["y_spread"] = y_imb - y_spot
                df_origin["y_direction"] = y_dir
                df_origin["y_spike_high"] = np.where(
                    (~np.isnan(y_imb)) & (~np.isnan(y_spot)),
                    (y_imb > y_spot + spike_margin).astype(np.float64),
                    np.nan,
                )
                df_origin["y_spike_low"] = np.where(
                    (~np.isnan(y_imb)) & (~np.isnan(y_spot)),
                    (y_imb < y_spot - spike_margin).astype(np.float64),
                    np.nan,
                )
                df_origin["target_first_usable_utc"] = pd.to_datetime(fu_target_ns, utc=True)

            rows_list.append(df_origin)

        return pd.concat(rows_list, ignore_index=True)


FEATURE_COLUMNS = [
    "step",
    "spot_target",
    "spot_prev_day",
    "spot_known",
    "spot_eff",
    "spot_ramp_1h",
    "spot_ramp_2h",
    "spot_lead_1h",
    "spot_vs_24h_mean",
    "spot_vs_24h_max",
    "spot_24h_std",
    "wind_fc_eff",
    "solar_fc_eff",
    "cons_fc_eff",
    "res_load_fc",
    "res_load_ramp_1h",
    "res_load_ramp_3h",
    "wind_ramp_1h",
    "cons_ramp_1h",
    "wind_share_fc",
    "fi_ee_cap",
    "ee_fi_cap",
    "se1_load_fc",
    "se3_load_fc",
    "ee_load_fc",
    "cal_hour",
    "cal_weekday",
    "cal_holiday",
    "cal_dst",
    "sin_hour",
    "cos_hour",
    "quarter_in_hour",
    "origin_hour_utc",
    "imb_last",
    "imb_lag_2",
    "imb_lag_3",
    "imb_lag_4",
    "imb_lag_8",
    "imb_lag_16",
    "imb_mean_4",
    "imb_std_4",
    "imb_min_4",
    "imb_max_4",
    "imb_mean_12",
    "imb_std_12",
    "imb_min_12",
    "imb_max_12",
    "imb_mean_24",
    "imb_mean_96",
    "imb_prev_day",
    "spread_last",
    "spread_lag_2",
    "spread_lag_3",
    "spread_lag_4",
    "spread_lag_8",
    "spread_mean_4",
    "spread_std_4",
    "spread_mean_12",
    "spread_max_12",
    "spread_min_12",
    "spread_mean_24",
    "spread_drift_30m",
    "spread_drift_1h",
    "spread_drift_2h",
    "spread_prev_day",
    "dir_last",
    "dir_lag_2",
    "dir_lag_4",
    "dir_mean_4",
    "dir_mean_12",
    "dir_mean_24",
    "dir_up_share_4",
    "dir_down_share_4",
    "dir_up_share_12",
    "dir_down_share_12",
    "dir_up_share_24",
    "dir_down_share_24",
    "dir_prev_day",
    "mfrr_up_vol_last",
    "mfrr_down_vol_last",
    "mfrr_up_vol_mean_4",
    "mfrr_down_vol_mean_4",
    "mfrr_up_prc_spread",
    "mfrr_down_prc_spread",
    "afrr_up_spread",
    "afrr_down_spread",
    "cons_last",
    "prod_last",
    "net_deficit_last",
    "net_deficit_drift_1h",
    "cons_err_last",
    "cons_err_mean_4",
    "cons_err_drift_1h",
    "flow_se1_last",
    "flow_se3_last",
    "flow_ee_last",
    "flow_no4_last",
    "total_flow_last",
    "total_flow_drift_1h",
    "se1_imb_last",
    "se3_imb_last",
    "ee_imb_last",
    "blended_spot_imb_fast",
    "blended_spot_imb_med",
    "imb_last_minus_spot_target",
    "spot_target_minus_spot_origin",
]
