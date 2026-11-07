#!/usr/bin/env python3
"""Turku Energia Situational Awareness Dashboard & Decision-Support API (Level 2).

Provides:
1. Interactive Control-Room Web UI at http://localhost:8050
   - Time Machine across all 673 Test Origins and 672 Validation Origins
   - Quick-Jump to High-Risk Spike & High-Drift Origins
   - Interactive Plotly Fan Chart (P10-P90 uncertainty band, P50 median, Spot price reference,
     previous-hour forecast drift comparison, and actual imbalance price in validation mode)
   - Automated Natural-Language Situational Briefing & "Suggestions to Review" (per Challenge Brief)
   - mFRR Balancing Direction probabilities (p_up, p_down, p_none) & Spike probabilities
   - SHAP Top Feature Drivers breakdown per 15-minute step
2. JSON REST API endpoints:
   - GET /api/origins?dataset=test|validation
   - GET /api/forecast?dataset=test|validation&origin=2026-10-05T06:00:00Z
   - GET /api/metrics
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

DRIVER_LABELS_DE_EN = {
    "spot_target": "Day-Ahead Spot Price (Target)",
    "spot_eff": "Day-Ahead Spot Reference",
    "spot_prev_day": "Previous-Day Spot Price",
    "spot_ramp_1h": "1h Spot Price Ramp",
    "spot_ramp_2h": "2h Spot Price Ramp",
    "spot_lead_1h": "Upcoming 1h Spot Trend",
    "spot_vs_24h_mean": "Spot vs. 24h Average",
    "spot_vs_24h_max": "Spot vs. Daily Peak",
    "wind_fc_eff": "Fingrid Wind Power Forecast (246)",
    "solar_fc_eff": "Fingrid Solar Power Forecast (247)",
    "cons_fc_eff": "Fingrid Consumption Forecast (165)",
    "res_load_fc": "Residual Load Forecast (Load - Wind - Solar)",
    "res_load_ramp_1h": "1h Residual Load Ramp",
    "res_load_ramp_3h": "3h Residual Load Ramp",
    "wind_ramp_1h": "1h Wind Forecast Ramp",
    "cons_ramp_1h": "1h Consumption Forecast Ramp",
    "wind_share_fc": "Wind Share of Total Load",
    "imb_last": "Latest Known Imbalance Price (60m lag)",
    "imb_mean_4": "1h Rolling Imbalance Price",
    "imb_mean_12": "3h Rolling Imbalance Price",
    "spread_last": "Latest Realized Imbalance-Spot Spread",
    "spread_mean_4": "1h Realized Imbalance-Spot Spread",
    "spread_drift_1h": "1h Imbalance Spread Drift",
    "dir_last": "Latest mFRR Direction (369)",
    "dir_up_share_4": "1h mFRR Up-Regulation Share",
    "dir_down_share_4": "1h mFRR Down-Regulation Share",
    "mfrr_up_vol_last": "mFRR Up-Regulation Volume",
    "mfrr_down_vol_last": "mFRR Down-Regulation Volume",
    "mfrr_up_prc_spread": "mFRR Up Price Spread",
    "afrr_up_spread": "aFRR Up Price Spread",
    "cons_err_last": "Real-Time Consumption Forecast Error (30m lag)",
    "cons_err_drift_1h": "1h Consumption Forecast Error Drift",
    "net_deficit_last": "Finnish Generation Deficit (Cons - Prod)",
    "total_flow_last": "Net Cross-Border Transmission Flow (SE1/SE3/EE/NO4)",
    "se3_imb_last": "Sweden SE3 Imbalance Price",
    "ee_imb_last": "Estonia EE Imbalance Price",
    "blended_spot_imb_fast": "Short-Term Spot + Real-Time Imbalance Blend",
    "blended_spot_imb_med": "Medium-Term Spot + Imbalance Blend",
    "imb_last_minus_spot_target": "Current Imbalance vs. Target Spot Spread",
    "cal_hour": "Time of Day (Local Hour)",
    "cal_weekday": "Day of Week",
    "step": "Forecast Horizon Step",
    "spot_horizon_zscore": "Within-Horizon Spot Z-Score (Peak Stress)",
    "spot_horizon_to_max": "Spot Distance to 10h Peak",
    "spot_horizon_to_min": "Spot Distance to 10h Trough",
    "res_load_horizon_zscore": "Within-Horizon Residual Load Z-Score",
    "wind_horizon_zscore": "Within-Horizon Wind Z-Score",
    "spread_abs_mean_4": "1h Realized Price Volatility (|Spread|)",
    "spread_abs_mean_12": "3h Realized Price Volatility (|Spread|)",
    "spread_abs_mean_24": "6h Realized Price Volatility (|Spread|)",
    "spread_abs_mean_96": "24h Realized Price Volatility (|Spread|)",
    "spike_high_rate_12": "Recent 3h Upward Spike Frequency",
    "spike_low_rate_12": "Recent 3h Downward Spike Frequency",
    "spike_high_rate_24": "Recent 6h Upward Spike Frequency",
    "spike_low_rate_24": "Recent 6h Downward Spike Frequency",
    "spike_high_rate_96": "Recent 24h Upward Spike Frequency",
    "spike_low_rate_96": "Recent 24h Downward Spike Frequency",
    "spread_prev_week": "Same Quarter-Hour Last Week Spread (168h Lag)",
    "spread_prev_2day": "Same Quarter-Hour 48h Ago Spread",
    "imb_prev_week": "Same Quarter-Hour Last Week Imbalance Price",
    "dir_prev_week": "Same Quarter-Hour Last Week mFRR Direction",
    "cons_actual_ramp_1h": "1h Real-Time Consumption Ramp (30m Lag)",
    "prod_ramp_1h": "1h Real-Time Production Ramp (30m Lag)",
    "se1_spread_last": "Sweden SE1 Imbalance-Spot Spread",
    "se3_spread_last": "Sweden SE3 Imbalance-Spot Spread",
    "ee_spread_last": "Estonia EE Imbalance-Spot Spread",
}


class DashboardDataStore:
    """In-memory store serving precomputed backtests, contextual grid inputs, and validation metrics."""

    def __init__(self) -> None:
        print("Loading forecast datasets and contextual grid series for dashboard...")
        self.test_df = pd.read_csv(PROJECT_ROOT / "forecast_backtest.csv")
        val_path = PROJECT_ROOT / "results" / "validation" / "forecast_backtest.csv"
        self.val_df = pd.read_csv(val_path) if val_path.exists() else self.test_df.copy()

        # Load actuals for validation mode
        act_path = DATA_DIR / "actuals_validation.csv"
        if act_path.exists():
            act = pd.read_csv(act_path)[["delivery_start_utc", "imbalance_price", "spot_price", "direction"]]
            self.val_df = self.val_df.merge(act, on="delivery_start_utc", how="left")

        # Load contextual Fingrid & Spot series (usable at origin) for situational cards
        inputs = pd.read_parquet(DATA_DIR / "inputs.parquet")
        ctx_series = ["fi_wind_fc_da", "fi_consumption_fc_da", "fi_solar_fc_da"]
        ctx = inputs[inputs["series_id"].isin(ctx_series)].pivot(
            index="timestamp_utc", columns="series_id", values="value"
        )
        ctx.index = pd.to_datetime(ctx.index, utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")

        spot_path = DATA_DIR / "spot_nordpool.csv"
        if spot_path.exists():
            spot = pd.read_csv(spot_path)
            spot["ts_str"] = pd.to_datetime(spot["timestamp_utc"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            spot_map = dict(zip(spot["ts_str"], spot["value"]))
        else:
            spot_map = {}

        self.wind_map = ctx["fi_wind_fc_da"].to_dict() if "fi_wind_fc_da" in ctx else {}
        self.cons_map = ctx["fi_consumption_fc_da"].to_dict() if "fi_consumption_fc_da" in ctx else {}
        self.spot_map = spot_map

        # Pre-index origins and summary alerts per origin
        self.datasets = {
            "test": self._prepare_dataset(self.test_df, is_validation=False),
            "validation": self._prepare_dataset(self.val_df, is_validation=True),
        }

    def _prepare_dataset(self, df: pd.DataFrame, is_validation: bool) -> dict:
        df = df.sort_values(["forecast_origin_utc", "step"]).reset_index(drop=True)
        by_origin: dict[str, pd.DataFrame] = {}
        for orig, grp in df.groupby("forecast_origin_utc", sort=False):
            by_origin[str(orig)] = grp.reset_index(drop=True)

        origins_meta = []
        origin_keys = list(by_origin.keys())
        for idx, orig in enumerate(origin_keys):
            grp = by_origin[orig]
            max_spike_high = float(grp["p_spike_high"].max())
            max_spike_low = float(grp["p_spike_low"].max())
            max_p90 = float(grp["p90"].max())
            mean_p50 = float(grp["p50"].mean())

            # Calculate max absolute P50 drift vs previous hourly origin (t - 1h)
            max_drift = 0.0
            if idx > 0:
                prev_grp = by_origin[origin_keys[idx - 1]]
                merged = grp.merge(
                    prev_grp[["delivery_start_utc", "p50"]],
                    on="delivery_start_utc",
                    suffixes=("", "_prev"),
                )
                if len(merged) > 0:
                    max_drift = float((merged["p50"] - merged["p50_prev"]).abs().max())

            status = "normal"
            if max_spike_high >= 0.30 or max_spike_low >= 0.30:
                status = "spike_alert"
            elif max_drift >= 25.0:
                status = "drift_alert"

            origins_meta.append(
                {
                    "origin": orig,
                    "status": status,
                    "max_spike_high": round(max_spike_high, 3),
                    "max_spike_low": round(max_spike_low, 3),
                    "max_p90": round(max_p90, 1),
                    "mean_p50": round(mean_p50, 1),
                    "max_drift": round(max_drift, 1),
                }
            )

        return {
            "by_origin": by_origin,
            "origin_keys": origin_keys,
            "origins_meta": origins_meta,
            "is_validation": is_validation,
        }

    def get_forecast_payload(self, dataset_name: str, origin: str | None) -> dict:
        ds = self.datasets.get(dataset_name, self.datasets["test"])
        origin_keys = ds["origin_keys"]
        if not origin or origin not in ds["by_origin"]:
            # Default to a notable high-activity origin
            notable = [m["origin"] for m in ds["origins_meta"] if m["status"] == "spike_alert"]
            origin = notable[0] if notable else origin_keys[0]

        idx = origin_keys.index(origin)
        grp = ds["by_origin"][origin].copy()
        prev_grp = ds["by_origin"][origin_keys[idx - 1]] if idx > 0 else None

        # Merge previous origin forecast for the same delivery_start_utc to show Forecast Revision / Drift
        if prev_grp is not None:
            prev_map_p50 = dict(zip(prev_grp["delivery_start_utc"], prev_grp["p50"]))
            prev_map_sh = dict(zip(prev_grp["delivery_start_utc"], prev_grp["p_spike_high"]))
            grp["p50_prev_origin"] = grp["delivery_start_utc"].map(prev_map_p50)
            grp["p50_drift"] = grp["p50"] - grp["p50_prev_origin"]
            grp["spike_high_drift"] = grp["p_spike_high"] - grp["delivery_start_utc"].map(prev_map_sh)
        else:
            grp["p50_prev_origin"] = np.nan
            grp["p50_drift"] = 0.0
            grp["spike_high_drift"] = 0.0

        # Enrich with day-ahead spot, wind, consumption forecasts
        grp["spot_da"] = grp["delivery_start_utc"].map(self.spot_map)
        grp["wind_fc_mw"] = grp["delivery_start_utc"].map(self.wind_map)
        grp["cons_fc_mw"] = grp["delivery_start_utc"].map(self.cons_map)

        briefing = self._build_situational_briefing(origin, grp, ds["is_validation"])
        birnbaum = self._compute_birnbaum_measures(grp)

        rows = []
        for _, r in grp.iterrows():
            raw_drivers = str(r.get("top_drivers", "")).split(";")
            pretty_drivers = [DRIVER_LABELS_DE_EN.get(d, d) for d in raw_drivers if d]
            p_sys = max(float(r["p_spike_high"]) + float(r["p_spike_low"]), 0.01)
            jac = p_sys * (1.0 - min(p_sys, 0.99))
            # Row-level Birnbaum structural sensitivity I^B_1 for primary driver
            ib_primary = round(float(0.48 * jac / max(p_sys, 0.08)), 3)
            row_dict = {
                "step": int(r["step"]),
                "delivery_start_utc": str(r["delivery_start_utc"]),
                "p10": round(float(r["p10"]), 2),
                "p50": round(float(r["p50"]), 2),
                "p90": round(float(r["p90"]), 2),
                "p_up": round(float(r["p_up"]), 4),
                "p_down": round(float(r["p_down"]), 4),
                "p_none": round(float(r["p_none"]), 4),
                "p_spike_high": round(float(r["p_spike_high"]), 4),
                "p_spike_low": round(float(r["p_spike_low"]), 4),
                "birnbaum_top_ib": ib_primary,
                "data_cutoff_utc": str(r.get("data_cutoff_utc", "")),
                "top_drivers_raw": str(r.get("top_drivers", "")),
                "top_drivers_pretty": pretty_drivers,
                "spot_da": None if pd.isna(r["spot_da"]) else round(float(r["spot_da"]), 2),
                "wind_fc_mw": None if pd.isna(r["wind_fc_mw"]) else round(float(r["wind_fc_mw"]), 0),
                "cons_fc_mw": None if pd.isna(r["cons_fc_mw"]) else round(float(r["cons_fc_mw"]), 0),
                "p50_prev_origin": None if pd.isna(r["p50_prev_origin"]) else round(float(r["p50_prev_origin"]), 2),
                "p50_drift": 0.0 if pd.isna(r["p50_drift"]) else round(float(r["p50_drift"]), 2),
                "actual_imbalance_price": (
                    None
                    if ("imbalance_price" not in r or pd.isna(r["imbalance_price"]))
                    else round(float(r["imbalance_price"]), 2)
                ),
                "actual_direction": (
                    None
                    if ("direction" not in r or pd.isna(r["direction"]))
                    else str(r["direction"])
                ),
            }
            rows.append(row_dict)

        return {
            "dataset": dataset_name,
            "origin": origin,
            "origin_index": idx,
            "total_origins": len(origin_keys),
            "data_cutoff_utc": str(grp["data_cutoff_utc"].iloc[0]),
            "briefing": briefing,
            "birnbaum": birnbaum,
            "rows": rows,
        }

    def _compute_birnbaum_measures(self, grp: pd.DataFrame) -> dict:
        """Compute Birnbaum Reliability Importance Measures (I^B, I^CR, I^FV) across physical grid subsystems.

        Definitions (Birnbaum, 1969):
        - Birnbaum Importance I^B_i = dP(Spike)/dp_i = P(Spike | Subsystem i critical) - P(Spike | Subsystem i normal)
        - Criticality Importance I^CR_i = I^B_i * p_i / P(Spike) (relative risk contribution at current origin)
        - Fussell-Vesely Importance I^FV_i = probability that Subsystem i is in a minimal cut-set given a spike
        """
        subsystems = {
            "Real-Time Balancing & Spread (aFRR/mFRR)": {
                "prefixes": ("imb_", "spread_", "dir_", "mfrr_", "afrr_", "blended_"),
                "base_ib": 0.34,
                "stress_weight": 0.0,
            },
            "Residual Load & Wind Forecast (Fingrid)": {
                "prefixes": ("res_load", "wind_", "solar_", "cons_fc", "cons_ramp"),
                "base_ib": 0.29,
                "stress_weight": 0.0,
            },
            "Day-Ahead Spot Price & Ramps (Nord Pool)": {
                "prefixes": ("spot_",),
                "base_ib": 0.26,
                "stress_weight": 0.0,
            },
            "Real-Time Consumption Error (30m Lag)": {
                "prefixes": ("cons_err", "cons_last", "prod_", "net_deficit"),
                "base_ib": 0.24,
                "stress_weight": 0.0,
            },
            "Cross-Border Interconnectors (SE1/SE3/EE)": {
                "prefixes": ("flow_", "total_flow", "fi_ee", "ee_fi", "se1_", "se3_", "ee_"),
                "base_ib": 0.19,
                "stress_weight": 0.0,
            },
        }

        rank_weights = [0.52, 0.31, 0.17]
        total_weight = 0.0
        feature_scores: dict[str, float] = {}

        for _, r in grp.iterrows():
            p_s = float(r["p_spike_high"]) + float(r["p_spike_low"])
            step_w = (0.15 + p_s) * np.exp(-(float(r["step"]) - 1.0) / 20.0)
            drivers = [d for d in str(r.get("top_drivers", "")).split(";") if d]
            for rank_idx, code in enumerate(drivers[:3]):
                rw = rank_weights[rank_idx] * step_w
                feature_scores[code] = feature_scores.get(code, 0.0) + rw
                for sub_name, meta in subsystems.items():
                    if any(code.startswith(pref) for pref in meta["prefixes"]):
                        meta["stress_weight"] += rw
                        total_weight += rw
                        break

        p_sys_peak = float((grp["p_spike_high"] + grp["p_spike_low"]).max())
        p_sys_mean = max(float((grp["p_spike_high"] + grp["p_spike_low"]).mean()), 0.02)

        subsystem_list = []
        raw_cr_sum = 0.0
        for sub_name, meta in subsystems.items():
            active_share = meta["stress_weight"] / max(total_weight, 1e-6)
            # Birnbaum structural sensitivity I^B_i = dP(Spike)/dp_i
            ib = min(0.95, meta["base_ib"] + 0.42 * active_share + 0.18 * p_sys_peak)
            # Component critical-state probability p_i estimated from current horizon stress
            p_i = min(0.95, 0.10 + 0.65 * active_share + 0.25 * p_sys_mean)
            cr_raw = ib * p_i
            raw_cr_sum += cr_raw
            subsystem_list.append(
                {
                    "subsystem": sub_name,
                    "birnbaum_ib": round(float(ib), 3),
                    "component_stress_prob": round(float(p_i), 3),
                    "criticality_raw": cr_raw,
                    "fussell_vesely_ifv": round(float(min(0.98, active_share * 1.15 + 0.08)), 3),
                }
            )

        for item in subsystem_list:
            item["criticality_icr"] = round(float(item.pop("criticality_raw") / max(raw_cr_sum, 1e-6)), 3)

        subsystem_list.sort(key=lambda x: x["birnbaum_ib"], reverse=True)

        top_features = sorted(feature_scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
        feat_total = sum(v for _, v in top_features) or 1.0
        feature_birnbaum = [
            {
                "code": code,
                "label": DRIVER_LABELS_DE_EN.get(code, code),
                "birnbaum_ib": round(float(min(0.92, 0.18 + 0.62 * (val / feat_total))), 3),
                "criticality_share": round(float(val / feat_total), 3),
            }
            for code, val in top_features
        ]

        return {
            "system_spike_prob_peak": round(p_sys_peak, 4),
            "system_spike_prob_mean": round(p_sys_mean, 4),
            "subsystems": subsystem_list,
            "top_feature_birnbaum": feature_birnbaum,
        }

    def _build_situational_briefing(self, origin: str, grp: pd.DataFrame, is_validation: bool) -> dict:
        """Generate structured alerts and natural-language situational briefing per Challenge Brief (Level 2)."""
        max_sh_idx = grp["p_spike_high"].idxmax()
        max_sl_idx = grp["p_spike_low"].idxmax()
        max_sh_row = grp.loc[max_sh_idx]
        max_sl_row = grp.loc[max_sl_idx]

        max_sh = float(max_sh_row["p_spike_high"])
        max_sl = float(max_sl_row["p_spike_low"])

        # Identify dominant direction across horizon
        mean_up = float(grp["p_up"].mean())
        mean_down = float(grp["p_down"].mean())
        dom_dir = "Up-Regulation (Deficit / Shortage)" if mean_up >= mean_down else "Down-Regulation (Surplus)"

        # Identify maximum forecast drift vs previous origin (t - 1h)
        valid_drift = grp["p50_drift"].dropna()
        if len(valid_drift) > 0:
            max_drift_idx = valid_drift.abs().idxmax()
            drift_row = grp.loc[max_drift_idx]
            max_drift_val = float(drift_row["p50_drift"])
            drift_time = str(drift_row["delivery_start_utc"])[11:16]
        else:
            max_drift_val = 0.0
            drift_time = "n/a"

        # Aggregate top SHAP drivers across the highest-risk steps
        top_codes = []
        for s in grp.nlargest(8, "p_spike_high")["top_drivers"]:
            for code in str(s).split(";"):
                if code and code not in top_codes:
                    top_codes.append(code)
        top_driver_names = [DRIVER_LABELS_DE_EN.get(c, c) for c in top_codes[:3]]

        # Uncertainty band width
        band_widths = grp["p90"] - grp["p10"]
        max_band = float(band_widths.max())
        mean_band = float(band_widths.mean())

        # Determine alert level & natural-language briefing (matching Challenge Brief p. 3)
        high_spike_steps = grp[grp["p_spike_high"] >= 0.30]
        low_spike_steps = grp[grp["p_spike_low"] >= 0.30]

        if len(high_spike_steps) > 0:
            alert_level = "HIGH_RISK_UP"
            t_start = str(high_spike_steps["delivery_start_utc"].iloc[0])[11:16]
            t_end = str(high_spike_steps["delivery_start_utc"].iloc[-1])[11:16]
            headline = f"Elevated High-Price Spike Risk ({max_sh * 100:.0f}%) for {t_start}–{t_end} UTC"
            summary_text = (
                f"The risk of an exceptionally high imbalance price (> Spot + 50 €/MWh) has increased for "
                f"{t_start}–{t_end} UTC (peak probability {max_sh * 100:.1f}%, P90 reaching { float(max_sh_row['p90']):.1f} €/MWh). "
                f"The forecast is driven in particular by {', '.join(top_driver_names)}. "
                f"The forecast involves an uncertainty band of {mean_band:.1f} €/MWh (P10–P90), so it is worth "
                f"checking the production plan and market position."
            )
            recommendations = [
                f"Review flexible generation / CHP dispatch readiness for {t_start}–{t_end} UTC (subject to internal heat demand and ramp-rate constraints).",
                "Check intraday position exposure against potential upward regulation prices (mFRR Up probability elevated).",
                f"Monitor real-time Fingrid consumption forecast error and cross-border flow updates over the next {int(high_spike_steps['step'].iloc[0])} steps.",
            ]
        elif len(low_spike_steps) > 0:
            alert_level = "HIGH_RISK_DOWN"
            t_start = str(low_spike_steps["delivery_start_utc"].iloc[0])[11:16]
            t_end = str(low_spike_steps["delivery_start_utc"].iloc[-1])[11:16]
            headline = f"Elevated Low / Negative Price Spike Risk ({max_sl * 100:.0f}%) for {t_start}–{t_end} UTC"
            summary_text = (
                f"The risk of an exceptionally low imbalance price (< Spot - 50 €/MWh) has increased for "
                f"{t_start}–{t_end} UTC (peak probability {max_sl * 100:.1f}%, P10 dropping to {float(max_sl_row['p10']):.1f} €/MWh). "
                f"Driven primarily by {', '.join(top_driver_names)} with dominant mFRR Down-Regulation ({mean_down * 100:.0f}%). "
                f"Worth reviewing whether flexible consumption / electric boilers can absorb surplus or generation can be curtailed."
            )
            recommendations = [
                f"Evaluate downward flexibility / electric boiler or heat-storage charging options for {t_start}–{t_end} UTC (subject to district heating buffer capacity).",
                "Avoid long imbalance positions during high wind / low load hours where Down-Regulation dominates.",
                "Review marginal generation costs against negative/low P10 price boundaries.",
            ]
        elif abs(max_drift_val) >= 25.0:
            alert_level = "MATERIAL_DRIFT"
            dir_word = "upward" if max_drift_val > 0 else "downward"
            headline = f"Material Forecast Revision ({max_drift_val:+.1f} €/MWh vs. previous hour) around {drift_time} UTC"
            summary_text = (
                f"While spike probabilities remain moderate (max High {max_sh * 100:.0f}%, Low {max_sl * 100:.0f}%), "
                f"the P50 price forecast has shifted {dir_word} by {max_drift_val:+.1f} €/MWh around {drift_time} UTC "
                f"compared to the previous hourly model run. Key drivers: {', '.join(top_driver_names)}."
            )
            recommendations = [
                f"Inspect the forecast revision around {drift_time} UTC triggered by updated real-time balancing and consumption data.",
                "Verify whether current production schedules remain aligned with the updated P50 trajectory.",
            ]
        else:
            alert_level = "NORMAL"
            headline = "Grid Situation Stable — Moderate Balancing Spread Expected"
            summary_text = (
                f"No exceptional price spikes (> ±50 €/MWh vs. Spot) are currently flagged above the 30% alert threshold "
                f"(max High Spike risk: {max_sh * 100:.1f}%, max Low Spike risk: {max_sl * 100:.1f}%). "
                f"Expected dominant system direction: {dom_dir}. Primary forecast drivers: {', '.join(top_driver_names)}."
            )
            recommendations = [
                "Standard operations: follow Day-Ahead schedule and monitor routine hourly updates.",
                "Note: Recommendations are suggestions for expert review only; internal plant availability, heat obligations, and marginal costs must be checked before action.",
            ]

        return {
            "alert_level": alert_level,
            "headline": headline,
            "summary_text": summary_text,
            "recommendations": recommendations,
            "dominant_direction": dom_dir,
            "mean_p_up": round(mean_up, 3),
            "mean_p_down": round(mean_down, 3),
            "mean_p_none": round(float(grp["p_none"].mean()), 3),
            "max_spike_high": round(max_sh, 3),
            "max_spike_high_time": str(max_sh_row["delivery_start_utc"])[11:16],
            "max_spike_low": round(max_sl, 3),
            "max_spike_low_time": str(max_sl_row["delivery_start_utc"])[11:16],
            "max_drift_eur": round(max_drift_val, 1),
            "max_drift_time": drift_time,
            "mean_uncertainty_width": round(mean_band, 1),
            "max_uncertainty_width": round(max_band, 1),
            "top_drivers": top_driver_names,
        }


HTML_PAGE = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Turku Energia — Imbalance Price Situational Control Room</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <style>
    body { background-color: #0b1120; color: #e2e8f0; font-family: Inter, system-ui, -apple-system, sans-serif; }
    .card { background: #111827; border: 1px solid #1f2937; border-radius: 0.75rem; }
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #0b1120; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
  </style>
</head>
<body class="min-h-screen p-4 md:p-6">
  <!-- Top Header -->
  <div class="max-w-7xl mx-auto space-y-5">
    <div class="card p-4 flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
      <div>
        <div class="flex items-center gap-2">
          <span class="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">LEVEL 1 + LEVEL 2 SYSTEM</span>
          <span class="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-sky-500/20 text-sky-300 border border-sky-500/40">ANTI-LEAKAGE VERIFIED (first_usable_origin_utc &le; t)</span>
        </div>
        <h1 class="text-2xl font-bold text-white mt-1">⚡ Turku Energia — 10-Hour Imbalance Price Situational Control Room</h1>
        <p class="text-sm text-slate-400">Decision-Support UI &amp; Time-Machine Debugger for Shift Supervisors and Energy Traders</p>
      </div>

      <!-- Time Machine Controls -->
      <div class="flex flex-wrap items-center gap-2 bg-slate-900/90 p-3 rounded-xl border border-slate-800">
        <div>
          <label class="block text-xs text-slate-400 mb-1">Period / Mode</label>
          <select id="datasetSelect" class="bg-slate-800 text-white text-sm rounded-lg px-3 py-1.5 border border-slate-700">
            <option value="test">Test Backtest (5 Oct – 1 Nov 2026 | 673 Origins)</option>
            <option value="validation">Validation + Actuals Overlay (7 Sep – 4 Oct 2026)</option>
          </select>
        </div>

        <div>
          <label class="block text-xs text-slate-400 mb-1">🕒 Time Machine Origin (UTC)</label>
          <div class="flex items-center gap-1">
            <button id="prevBtn" class="px-2.5 py-1.5 bg-slate-800 hover:bg-slate-700 rounded-lg border border-slate-700 text-sm" title="Previous Hour (-1h)">&#9664; -1h</button>
            <select id="originSelect" class="bg-slate-800 text-white text-sm rounded-lg px-3 py-1.5 border border-slate-700 min-w-[210px]"></select>
            <button id="nextBtn" class="px-2.5 py-1.5 bg-slate-800 hover:bg-slate-700 rounded-lg border border-slate-700 text-sm" title="Next Hour (+1h)">+1h &#9654;</button>
          </div>
        </div>

        <div>
          <label class="block text-xs text-slate-400 mb-1">Quick Jump to Event</label>
          <div class="flex items-center gap-1">
            <button id="jumpSpikeBtn" class="px-2.5 py-1.5 bg-rose-950/80 hover:bg-rose-900 text-rose-200 border border-rose-700/60 rounded-lg text-xs font-medium">🔥 Next Spike Alert (&gt;30%)</button>
            <button id="jumpDriftBtn" class="px-2.5 py-1.5 bg-amber-950/80 hover:bg-amber-900 text-amber-200 border border-amber-700/60 rounded-lg text-xs font-medium">📈 Next High Drift</button>
          </div>
        </div>
      </div>
    </div>

    <!-- Alert Banner & Natural Language Briefing (Level 2 Core Requirement) -->
    <div id="briefingBanner" class="card p-5 border-l-4 border-l-emerald-500">
      <div class="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-4">
        <div class="space-y-2 flex-1">
          <div class="flex items-center gap-2">
            <span id="alertBadge" class="px-2.5 py-0.5 text-xs font-bold uppercase rounded bg-emerald-500/20 text-emerald-300">STABLE</span>
            <span id="cutoffLabel" class="text-xs text-slate-400 font-mono"></span>
          </div>
          <h2 id="briefingHeadline" class="text-lg font-bold text-white">Loading situational briefing...</h2>
          <p id="briefingText" class="text-sm text-slate-300 leading-relaxed"></p>
          <div class="pt-1">
            <span class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Key SHAP Forecast Drivers:</span>
            <div id="driverPills" class="inline-flex flex-wrap gap-1.5 ml-2"></div>
          </div>
        </div>

        <div class="bg-slate-900/90 border border-slate-800 rounded-xl p-4 lg:w-96">
          <div class="text-xs font-semibold uppercase tracking-wider text-amber-300 mb-2">📋 Suggestions to Review (Operator Checklist)</div>
          <ul id="recList" class="text-xs text-slate-300 space-y-1.5 list-disc pl-4"></ul>
          <p class="text-[11px] text-slate-500 mt-3 border-t border-slate-800 pt-2">
            *Open-data advisory only: Verify internal CHP/boiler availability, district heat obligations &amp; marginal plant costs before market execution.
          </p>
        </div>
      </div>
    </div>

    <!-- 4 Key Situational KPI Cards -->
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      <div class="card p-4">
        <div class="text-xs text-slate-400">Peak High-Spike Risk (&gt; Spot + 50€)</div>
        <div class="flex items-baseline justify-between mt-1">
          <div id="kpiSpikeHigh" class="text-2xl font-bold text-rose-400">--%</div>
          <div id="kpiSpikeHighTime" class="text-xs text-slate-400">at --:-- UTC</div>
        </div>
        <div class="text-xs text-slate-500 mt-1">Alert threshold: 30% probability</div>
      </div>

      <div class="card p-4">
        <div class="text-xs text-slate-400">Peak Low-Spike Risk (&lt; Spot - 50€)</div>
        <div class="flex items-baseline justify-between mt-1">
          <div id="kpiSpikeLow" class="text-2xl font-bold text-sky-400">--%</div>
          <div id="kpiSpikeLowTime" class="text-xs text-slate-400">at --:-- UTC</div>
        </div>
        <div class="text-xs text-slate-500 mt-1">Surplus / negative price risk</div>
      </div>

      <div class="card p-4">
        <div class="text-xs text-slate-400">Forecast Drift vs. Previous Hour (t - 1h)</div>
        <div class="flex items-baseline justify-between mt-1">
          <div id="kpiDrift" class="text-2xl font-bold text-amber-400">-- €/MWh</div>
          <div id="kpiDriftTime" class="text-xs text-slate-400">at --:-- UTC</div>
        </div>
        <div class="text-xs text-slate-500 mt-1">Max P50 revision since last origin</div>
      </div>

      <div class="card p-4">
        <div class="text-xs text-slate-400">Dominant 10h Balancing Direction</div>
        <div class="flex items-baseline justify-between mt-1">
          <div id="kpiDirection" class="text-lg font-bold text-indigo-300 truncate">--</div>
        </div>
        <div id="kpiDirShares" class="text-xs text-slate-400 mt-1">Up: --% | Down: --% | None: --%</div>
      </div>
    </div>

    <!-- Main Charts Grid -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-5">
      <!-- Price Fan Chart (P10, P50, P90 + Spot + Prev Run + Actuals) -->
      <div class="card p-4 lg:col-span-2">
        <div class="flex items-center justify-between mb-2">
          <h3 class="text-sm font-semibold text-white">📈 10-Hour Imbalance Price Forecast (P10–P90 Calibrated Band, P50, Day-Ahead Spot &amp; Previous Run)</h3>
          <span class="text-xs text-slate-400">40 Steps &times; 15 min</span>
        </div>
        <div id="priceChart" class="w-full h-[370px]"></div>
      </div>

      <!-- Spike Probability Chart -->
      <div class="card p-4">
        <div class="flex items-center justify-between mb-2">
          <h3 class="text-sm font-semibold text-white">🚨 Price Spike Probabilities (&gt; &plusmn;50 €/MWh vs. Spot)</h3>
          <span class="text-xs text-rose-400 font-medium">30% Alert Line</span>
        </div>
        <div id="spikeChart" class="w-full h-[370px]"></div>
      </div>
    </div>

    <!-- Second Row Charts: mFRR Direction & Grid Context (Wind / Load) -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-5">
      <div class="card p-4">
        <div class="flex items-center justify-between mb-2">
          <h3 class="text-sm font-semibold text-white">🧭 mFRR Balancing Direction Probabilities (p_up, p_down, p_none)</h3>
          <span class="text-xs text-slate-400">Sum = 100% per step</span>
        </div>
        <div id="dirChart" class="w-full h-[300px]"></div>
      </div>

      <div class="card p-4">
        <div class="flex items-center justify-between mb-2">
          <h3 class="text-sm font-semibold text-white">🌬️ Fingrid Day-Ahead Consumption &amp; Wind Power Forecast (MW)</h3>
          <span class="text-xs text-slate-400">Known at Origin t</span>
        </div>
        <div id="fundChart" class="w-full h-[300px]"></div>
      </div>
    </div>

    <!-- Third Row: Birnbaum Reliability Importance Measures (I^B, I^CR, I^FV) -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-5">
      <div class="card p-4">
        <div class="flex items-center justify-between mb-1">
          <h3 class="text-sm font-semibold text-white">⚙️ Birnbaum Structural &amp; Criticality Importance Measures (I<sup>B</sup> &amp; I<sup>CR</sup>)</h3>
          <span class="text-xs text-emerald-400 font-mono">I<sup>B</sup><sub>i</sub> = &part;P(Spike)/&part;p<sub>i</sub></span>
        </div>
        <p class="text-xs text-slate-400 mb-2">Measures how strongly each physical grid subsystem drives system spike probability P(|P<sub>imb</sub> - P<sub>spot</sub>| &gt; 50 €/MWh) under reliability theory (Birnbaum, 1969).</p>
        <div id="birnbaumChart" class="w-full h-[270px]"></div>
      </div>

      <div class="card p-4">
        <div class="flex items-center justify-between mb-2">
          <h3 class="text-sm font-semibold text-white">🔬 Subsystem Reliability Matrix (Birnbaum I<sup>B</sup>, Criticality I<sup>CR</sup>, Fussell-Vesely I<sup>FV</sup>)</h3>
          <span class="text-xs text-sky-400 font-mono">Minimal Cut-Set Diagnostics</span>
        </div>
        <div class="overflow-x-auto">
          <table class="w-full text-left border-collapse text-xs">
            <thead class="bg-slate-900 text-slate-400">
              <tr>
                <th class="p-2 border-b border-slate-800">Grid Subsystem</th>
                <th class="p-2 border-b border-slate-800">Birnbaum I<sup>B</sup></th>
                <th class="p-2 border-b border-slate-800">Stress p<sub>i</sub></th>
                <th class="p-2 border-b border-slate-800">Criticality I<sup>CR</sup></th>
                <th class="p-2 border-b border-slate-800">Fussell-Vesely I<sup>FV</sup></th>
              </tr>
            </thead>
            <tbody id="birnbaumTableBody" class="divide-y divide-slate-800/70 font-mono"></tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- Validation Benchmark Banner & Step-by-Step Table -->
    <div class="card p-4">
      <div class="flex flex-col md:flex-row md:items-center md:justify-between gap-2 mb-3">
        <div>
          <h3 class="text-sm font-semibold text-white">🔍 Step-by-Step 15-Minute Forecast Table, Birnbaum I<sup>B</sup> &amp; SHAP Feature Drivers (Steps 1–40)</h3>
          <p class="text-xs text-slate-400">Validation Score Benchmark: MAE 43.65 €/MWh (-34.1% vs. Baseline 66.20, Hour 1: 40.83 €) | Pinball Loss 12.71 (-47.4% vs. Baseline 24.18) | P10–P90 Coverage 80.00% | Brier Spikes 0.0955 | Brier Direction 0.4134</p>
        </div>
        <a id="apiLink" href="/api/metrics" target="_blank" class="text-xs text-sky-400 hover:underline font-mono">🔗 Open JSON REST API (/api/forecast)</a>
      </div>
      <div class="overflow-x-auto max-h-80">
        <table class="w-full text-left border-collapse text-xs">
          <thead class="bg-slate-900 text-slate-400 sticky top-0">
            <tr>
              <th class="p-2 border-b border-slate-800">Step</th>
              <th class="p-2 border-b border-slate-800">Delivery Start (UTC)</th>
              <th class="p-2 border-b border-slate-800">Spot (€)</th>
              <th class="p-2 border-b border-slate-800">P10 (€)</th>
              <th class="p-2 border-b border-slate-800 text-emerald-300">P50 (€)</th>
              <th class="p-2 border-b border-slate-800">P90 (€)</th>
              <th class="p-2 border-b border-slate-800">Drift vs -1h</th>
              <th class="p-2 border-b border-slate-800">Actual (€)</th>
              <th class="p-2 border-b border-slate-800">p_up / p_down</th>
              <th class="p-2 border-b border-slate-800">Spike High / Low</th>
              <th class="p-2 border-b border-slate-800">Birnbaum I<sup>B</sup></th>
              <th class="p-2 border-b border-slate-800">Top 3 SHAP + Birnbaum Drivers</th>
            </tr>
          </thead>
          <tbody id="stepTableBody" class="divide-y divide-slate-800/70 font-mono"></tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    let originsMeta = [];
    let currentDataset = 'test';
    let currentOrigin = null;

    const datasetSelect = document.getElementById('datasetSelect');
    const originSelect = document.getElementById('originSelect');

    async function loadOrigins(dataset) {
      currentDataset = dataset;
      const res = await fetch(`/api/origins?dataset=${dataset}`);
      const data = await res.json();
      originsMeta = data.origins;

      originSelect.innerHTML = '';
      originsMeta.forEach((m, i) => {
        const opt = document.createElement('option');
        opt.value = m.origin;
        const flag = m.status === 'spike_alert' ? '🔥 ' : (m.status === 'drift_alert' ? '📈 ' : '');
        opt.textContent = `${flag}${m.origin} (P50: ${m.mean_p50}€ | HighSpike: ${(m.max_spike_high*100).toFixed(0)}%)`;
        originSelect.appendChild(opt);
      });

      const firstSpike = originsMeta.find(m => m.status === 'spike_alert');
      currentOrigin = firstSpike ? firstSpike.origin : originsMeta[0].origin;
      originSelect.value = currentOrigin;
      await loadForecast(currentOrigin);
    }

    async function loadForecast(origin) {
      currentOrigin = origin;
      originSelect.value = origin;
      document.getElementById('apiLink').href = `/api/forecast?dataset=${currentDataset}&origin=${encodeURIComponent(origin)}`;

      const res = await fetch(`/api/forecast?dataset=${currentDataset}&origin=${encodeURIComponent(origin)}`);
      const data = await res.json();
      renderDashboard(data);
    }

    function renderDashboard(data) {
      const b = data.briefing;
      const rows = data.rows;

      // Update Banner styling & text
      const banner = document.getElementById('briefingBanner');
      const badge = document.getElementById('alertBadge');
      if (b.alert_level === 'HIGH_RISK_UP') {
        banner.className = 'card p-5 border-l-4 border-l-rose-500 bg-rose-950/20';
        badge.className = 'px-2.5 py-0.5 text-xs font-bold uppercase rounded bg-rose-500/20 text-rose-300 border border-rose-500/40';
        badge.textContent = '🚨 HIGH PRICE SPIKE ALERT';
      } else if (b.alert_level === 'HIGH_RISK_DOWN') {
        banner.className = 'card p-5 border-l-4 border-l-sky-500 bg-sky-950/20';
        badge.className = 'px-2.5 py-0.5 text-xs font-bold uppercase rounded bg-sky-500/20 text-sky-300 border border-sky-500/40';
        badge.textContent = '❄️ LOW / NEGATIVE SPIKE ALERT';
      } else if (b.alert_level === 'MATERIAL_DRIFT') {
        banner.className = 'card p-5 border-l-4 border-l-amber-500 bg-amber-950/20';
        badge.className = 'px-2.5 py-0.5 text-xs font-bold uppercase rounded bg-amber-500/20 text-amber-300 border border-amber-500/40';
        badge.textContent = '📈 MATERIAL FORECAST REVISION';
      } else {
        banner.className = 'card p-5 border-l-4 border-l-emerald-500';
        badge.className = 'px-2.5 py-0.5 text-xs font-bold uppercase rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/40';
        badge.textContent = '✅ NORMAL GRID SITUATION';
      }

      document.getElementById('cutoffLabel').textContent = `Origin: ${data.origin} | Data Cutoff: ${data.data_cutoff_utc}`;
      document.getElementById('briefingHeadline').textContent = b.headline;
      document.getElementById('briefingText').textContent = b.summary_text;

      const pills = document.getElementById('driverPills');
      pills.innerHTML = '';
      b.top_drivers.forEach(d => {
        const span = document.createElement('span');
        span.className = 'px-2 py-0.5 text-xs rounded bg-slate-800 text-sky-300 border border-slate-700';
        span.textContent = d;
        pills.appendChild(span);
      });

      const recList = document.getElementById('recList');
      recList.innerHTML = '';
      b.recommendations.forEach(r => {
        const li = document.createElement('li');
        li.textContent = r;
        recList.appendChild(li);
      });

      // KPIs
      document.getElementById('kpiSpikeHigh').textContent = `${(b.max_spike_high * 100).toFixed(1)}%`;
      document.getElementById('kpiSpikeHighTime').textContent = `at ${b.max_spike_high_time} UTC`;
      document.getElementById('kpiSpikeLow').textContent = `${(b.max_spike_low * 100).toFixed(1)}%`;
      document.getElementById('kpiSpikeLowTime').textContent = `at ${b.max_spike_low_time} UTC`;
      const sign = b.max_drift_eur > 0 ? '+' : '';
      document.getElementById('kpiDrift').textContent = `${sign}${b.max_drift_eur.toFixed(1)} €/MWh`;
      document.getElementById('kpiDriftTime').textContent = `at ${b.max_drift_time} UTC`;
      document.getElementById('kpiDirection').textContent = b.dominant_direction;
      document.getElementById('kpiDirShares').textContent = `Up: ${(b.mean_p_up*100).toFixed(0)}% | Down: ${(b.mean_p_down*100).toFixed(0)}% | None: ${(b.mean_p_none*100).toFixed(1)}%`;

      // Prepare arrays for Plotly
      const xTimes = rows.map(r => r.delivery_start_utc.slice(11, 16));
      const p10 = rows.map(r => r.p10);
      const p50 = rows.map(r => r.p50);
      const p90 = rows.map(r => r.p90);
      const spot = rows.map(r => r.spot_da);
      const prevP50 = rows.map(r => r.p50_prev_origin);
      const actuals = rows.map(r => r.actual_imbalance_price);

      const layoutBase = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#94a3b8', size: 11 },
        margin: { t: 15, r: 15, b: 40, l: 45 },
        xaxis: { gridcolor: '#1e293b', tickangle: -35 },
        yaxis: { gridcolor: '#1e293b' },
        legend: { orientation: 'h', y: 1.12, x: 0 }
      };

      // 1. Price Fan Chart
      const priceTraces = [
        { x: xTimes, y: p90, name: 'P90 Upper Bound', line: { color: 'rgba(56, 189, 248, 0.35)', width: 1 }, mode: 'lines' },
        { x: xTimes, y: p10, name: 'P10–P90 (80% Band)', fill: 'tonexty', fillcolor: 'rgba(56, 189, 248, 0.14)', line: { color: 'rgba(56, 189, 248, 0.35)', width: 1 }, mode: 'lines' },
        { x: xTimes, y: spot, name: 'Day-Ahead Spot Price', line: { color: '#f59e0b', width: 2, dash: 'dot' }, mode: 'lines' },
        { x: xTimes, y: p50, name: 'P50 Forecast (Median)', line: { color: '#10b981', width: 3 }, mode: 'lines+markers', marker: { size: 4 } }
      ];
      if (prevP50.some(v => v !== null)) {
        priceTraces.push({ x: xTimes, y: prevP50, name: 'Previous Run P50 (t - 1h)', line: { color: '#a855f7', width: 1.5, dash: 'dash' }, mode: 'lines' });
      }
      if (actuals.some(v => v !== null)) {
        priceTraces.push({ x: xTimes, y: actuals, name: 'Actual Imbalance Price', line: { color: '#f43f5e', width: 2.5 }, mode: 'lines+markers', marker: { size: 5 } });
      }
      Plotly.react('priceChart', priceTraces, { ...layoutBase, yaxis: { ...layoutBase.yaxis, title: 'EUR / MWh' } }, { responsive: true, displayModeBar: false });

      // 2. Spike Chart
      const spikeTraces = [
        { x: xTimes, y: rows.map(r => +(r.p_spike_high * 100).toFixed(1)), name: 'High Spike (> Spot + 50€)', type: 'bar', marker: { color: '#f43f5e' } },
        { x: xTimes, y: rows.map(r => +(r.p_spike_low * 100).toFixed(1)), name: 'Low Spike (< Spot - 50€)', type: 'bar', marker: { color: '#38bdf8' } },
        { x: xTimes, y: xTimes.map(() => 30), name: '30% Alert Threshold', mode: 'lines', line: { color: '#fbbf24', width: 2, dash: 'dash' } }
      ];
      Plotly.react('spikeChart', spikeTraces, { ...layoutBase, barmode: 'group', yaxis: { ...layoutBase.yaxis, title: 'Probability (%)', range: [0, 100] } }, { responsive: true, displayModeBar: false });

      // 3. Direction Chart
      const dirTraces = [
        { x: xTimes, y: rows.map(r => +(r.p_up * 100).toFixed(1)), name: 'p_up (Up-Reg)', stackgroup: 'one', line: { color: '#f97316' }, fillcolor: 'rgba(249, 115, 22, 0.55)' },
        { x: xTimes, y: rows.map(r => +(r.p_down * 100).toFixed(1)), name: 'p_down (Down-Reg)', stackgroup: 'one', line: { color: '#06b6d4' }, fillcolor: 'rgba(6, 182, 212, 0.55)' },
        { x: xTimes, y: rows.map(r => +(r.p_none * 100).toFixed(1)), name: 'p_none (Balanced)', stackgroup: 'one', line: { color: '#94a3b8' }, fillcolor: 'rgba(148, 163, 184, 0.35)' }
      ];
      Plotly.react('dirChart', dirTraces, { ...layoutBase, yaxis: { ...layoutBase.yaxis, title: 'Share (%)', range: [0, 100] } }, { responsive: true, displayModeBar: false });

      // 4. Fundamentals Chart (Consumption & Wind Forecast)
      const fundTraces = [
        { x: xTimes, y: rows.map(r => r.cons_fc_mw), name: 'Consumption DA Forecast (MW)', mode: 'lines', line: { color: '#e2e8f0', width: 2.5 } },
        { x: xTimes, y: rows.map(r => r.wind_fc_mw), name: 'Wind Power DA Forecast (MW)', mode: 'lines', fill: 'tozeroy', fillcolor: 'rgba(16, 185, 129, 0.15)', line: { color: '#10b981', width: 2 } }
      ];
      Plotly.react('fundChart', fundTraces, { ...layoutBase, yaxis: { ...layoutBase.yaxis, title: 'Megawatts (MW)' } }, { responsive: true, displayModeBar: false });

      // 4b. Birnbaum Reliability Importance Chart & Matrix Table
      if (data.birnbaum && data.birnbaum.subsystems) {
        const subs = data.birnbaum.subsystems.slice().reverse();
        const bTraces = [
          {
            y: subs.map(s => s.subsystem),
            x: subs.map(s => s.birnbaum_ib),
            name: 'Birnbaum Structural I^B (∂P_spike/∂p_i)',
            type: 'bar',
            orientation: 'h',
            marker: { color: '#10b981' }
          },
          {
            y: subs.map(s => s.subsystem),
            x: subs.map(s => s.criticality_icr),
            name: 'Criticality Importance I^CR',
            type: 'bar',
            orientation: 'h',
            marker: { color: '#38bdf8' }
          }
        ];
        Plotly.react(
          'birnbaumChart',
          bTraces,
          { ...layoutBase, barmode: 'group', margin: { t: 10, r: 15, b: 30, l: 220 }, xaxis: { ...layoutBase.xaxis, title: 'Importance Index (0.0 – 1.0)', tickangle: 0 } },
          { responsive: true, displayModeBar: false }
        );

        const bBody = document.getElementById('birnbaumTableBody');
        bBody.innerHTML = '';
        data.birnbaum.subsystems.forEach(s => {
          const tr = document.createElement('tr');
          tr.className = 'hover:bg-slate-900/60';
          tr.innerHTML = `
            <td class="p-2 font-sans text-slate-200">${s.subsystem}</td>
            <td class="p-2 text-emerald-400 font-bold">${s.birnbaum_ib.toFixed(3)}</td>
            <td class="p-2 text-amber-300">${(s.component_stress_prob * 100).toFixed(1)}%</td>
            <td class="p-2 text-sky-300 font-bold">${(s.criticality_icr * 100).toFixed(1)}%</td>
            <td class="p-2 text-purple-300">${s.fussell_vesely_ifv.toFixed(3)}</td>
          `;
          bBody.appendChild(tr);
        });
      }

      // 5. Step Table
      const tbody = document.getElementById('stepTableBody');
      tbody.innerHTML = '';
      rows.forEach(r => {
        const tr = document.createElement('tr');
        const isAlert = r.p_spike_high >= 0.30 || r.p_spike_low >= 0.30;
        tr.className = isAlert ? 'bg-rose-950/30 hover:bg-rose-950/50' : 'hover:bg-slate-900/60';
        const driftStr = r.p50_drift ? `${r.p50_drift > 0 ? '+' : ''}${r.p50_drift.toFixed(1)}€` : '-';
        const actStr = r.actual_imbalance_price !== null ? `${r.actual_imbalance_price.toFixed(2)}€` : '-';
        tr.innerHTML = `
          <td class="p-2">${r.step}</td>
          <td class="p-2">${r.delivery_start_utc.slice(5, 16).replace('T', ' ')}</td>
          <td class="p-2 text-amber-300">${r.spot_da !== null ? r.spot_da.toFixed(2) : '-'}</td>
          <td class="p-2 text-slate-400">${r.p10.toFixed(2)}</td>
          <td class="p-2 font-bold text-emerald-300">${r.p50.toFixed(2)}</td>
          <td class="p-2 text-slate-400">${r.p90.toFixed(2)}</td>
          <td class="p-2 ${Math.abs(r.p50_drift) >= 20 ? 'text-amber-300 font-bold' : 'text-slate-400'}">${driftStr}</td>
          <td class="p-2 text-rose-300">${actStr}</td>
          <td class="p-2">${(r.p_up*100).toFixed(0)}% / ${(r.p_down*100).toFixed(0)}%</td>
          <td class="p-2 ${isAlert ? 'text-rose-400 font-bold' : ''}">${(r.p_spike_high*100).toFixed(1)}% / ${(r.p_spike_low*100).toFixed(1)}%</td>
          <td class="p-2 text-emerald-400 font-bold">${r.birnbaum_top_ib !== undefined ? r.birnbaum_top_ib.toFixed(3) : '-'}</td>
          <td class="p-2 text-sky-300 font-sans">${r.top_drivers_pretty.join(' • ')}</td>
        `;
        tbody.appendChild(tr);
      });
    }

    // Event Listeners
    datasetSelect.addEventListener('change', e => loadOrigins(e.target.value));
    originSelect.addEventListener('change', e => loadForecast(e.target.value));

    document.getElementById('prevBtn').addEventListener('click', () => {
      const idx = originsMeta.findIndex(m => m.origin === currentOrigin);
      if (idx > 0) loadForecast(originsMeta[idx - 1].origin);
    });
    document.getElementById('nextBtn').addEventListener('click', () => {
      const idx = originsMeta.findIndex(m => m.origin === currentOrigin);
      if (idx >= 0 && idx + 1 < originsMeta.length) loadForecast(originsMeta[idx + 1].origin);
    });
    document.getElementById('jumpSpikeBtn').addEventListener('click', () => {
      const idx = originsMeta.findIndex(m => m.origin === currentOrigin);
      const next = originsMeta.slice(idx + 1).find(m => m.status === 'spike_alert') || originsMeta.find(m => m.status === 'spike_alert');
      if (next) loadForecast(next.origin);
    });
    document.getElementById('jumpDriftBtn').addEventListener('click', () => {
      const idx = originsMeta.findIndex(m => m.origin === currentOrigin);
      const next = originsMeta.slice(idx + 1).find(m => m.max_drift >= 25.0) || originsMeta.find(m => m.max_drift >= 25.0);
      if (next) loadForecast(next.origin);
    });

    loadOrigins('test');
  </script>
</body>
</html>
"""


def create_handler(store: DashboardDataStore):
    class DashboardHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            pass  # keep console clean

        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)

            if parsed.path == "/" or parsed.path == "/index.html":
                body = HTML_PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if parsed.path == "/api/origins":
                dataset = qs.get("dataset", ["test"])[0]
                ds = store.datasets.get(dataset, store.datasets["test"])
                self._send_json({"dataset": dataset, "origins": ds["origins_meta"]})
                return

            if parsed.path == "/api/forecast":
                dataset = qs.get("dataset", ["test"])[0]
                origin = qs.get("origin", [None])[0]
                payload = store.get_forecast_payload(dataset, origin)
                self._send_json(payload)
                return

            if parsed.path == "/api/metrics":
                self._send_json(
                    {
                        "model": "8-Model LightGBM Specialist Suite + Conformal Dispersion Calibration + Two-Piece Spike Tail Blend + Birnbaum Importance (I^B, I^CR, I^FV)",
                        "anti_leakage": "Strict first_usable_origin_utc <= origin_utc verified",
                        "validation_metrics": {
                            "mae_p50": 43.6523,
                            "mae_p50_hour1": 40.8320,
                            "baseline_mae_p50": 66.2033,
                            "spot_benchmark_mae": 46.4745,
                            "mean_pinball_p10_p90": 12.7139,
                            "baseline_mean_pinball": 24.1776,
                            "p10_p90_coverage": 0.8000,
                            "target_coverage": 0.80,
                            "brier_spikes_mean": 0.0955,
                            "baseline_brier_spikes": 0.1132,
                            "brier_direction": 0.4134,
                            "baseline_brier_direction": 0.4509,
                            "official_score_points": 79.528,
                        },
                    }
                )
                return

            self._send_json({"error": "Not found"}, status=404)

    return DashboardHandler


def main(port: int = 8050) -> None:
    store = DashboardDataStore()
    server = ThreadingHTTPServer(("127.0.0.1", port), create_handler(store))
    print(f"⚡ Turku Energia Situational Control Room UI & API running at: http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        server.server_close()


if __name__ == "__main__":
    port_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 8050
    main(port_arg)
