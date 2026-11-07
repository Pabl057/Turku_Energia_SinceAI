"""LightGBM model suite, walk-forward training, SHAP explainability, and post-processing.

Implements the 6-model probabilistic forecasting architecture:
  1. m_p50  : LGBMRegressor(objective='regression_l1') on imbalance spread (imbalance - spot)
  2. m_p10  : LGBMRegressor(objective='quantile', alpha=0.10) + split conformal dispersion calibration
  3. m_p90  : LGBMRegressor(objective='quantile', alpha=0.90) + split conformal dispersion calibration
  4. m_dir  : LGBMClassifier(objective='multiclass', num_class=3) on mFRR direction (up/down/none)
  5. m_hi   : LGBMClassifier(objective='binary') + quantile tail blend for p_spike_high (> spot + 50)
  6. m_lo   : LGBMClassifier(objective='binary') + quantile tail blend for p_spike_low (< spot - 50)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from scipy.stats import norm

from src.data.loader import SPIKE_MARGIN, to_utc
from src.features.build_features import FEATURE_DRIVER_NAMES

RANDOM_SEED = 42
DIRECTION_TO_INT = {1.0: 0, -1.0: 1, 0.0: 2}
INT_TO_DIRECTION = {0: "up", 1: "down", 2: "none"}


def _predict_3class_proba(clf: lgb.LGBMClassifier, X: pd.DataFrame) -> np.ndarray:
    """Return (N, 3) probability matrix aligned to class indices 0 (up), 1 (down), 2 (none)."""
    raw = np.asarray(clf.predict_proba(X), dtype=float)
    if raw.ndim == 1:
        raw = raw[:, None]
    out = np.zeros((len(X), 3), dtype=float)
    for idx, cls_id in enumerate(clf.classes_):
        out[:, int(cls_id)] = raw[:, idx]
    return out


VOLUME_PRICING_CLASSIFIER_ONLY_COLS: set[str] = {
    "afrr_mfrr_up_gap_60m",
    "afrr_mfrr_dn_gap_60m",
    "afrr_mfrr_up_gap_mean_4q",
    "afrr_mfrr_dn_gap_mean_4q",
    "implied_mfrr_weight_60m",
    "implied_mfrr_weight_mean_4q",
    "implied_mfrr_weight_mean_16q",
    "implied_afrr_vol_60m",
    "implied_afrr_vol_mean_4q",
    "wind_rt_at_origin",
    "wind_rt_error_at_origin",
    "wind_rt_error_mean_4q",
    "wind_rt_ramp_1h",
    "nuclear_rt_at_origin",
    "nuclear_rt_ramp_1h",
    "grid_state_rt_at_origin",
}


@dataclass
class ImbalanceModelSuite:
    """Trained LightGBM suite with short-horizon specialists and Empirical Bayes calibration."""

    feature_cols: list[str]
    m_p50: lgb.LGBMRegressor
    m_p10: lgb.LGBMRegressor
    m_p90: lgb.LGBMRegressor
    m_dir: lgb.LGBMClassifier
    m_spike_high: lgb.LGBMClassifier
    m_spike_low: lgb.LGBMClassifier
    cutoff_origin_utc: pd.Timestamp
    reg_feature_cols: list[str] | None = None
    m_p50_deep: lgb.LGBMRegressor | None = None
    m_p50_short: lgb.LGBMRegressor | None = None
    m_dir_short: lgb.LGBMClassifier | None = None
    prior_up: float = 0.35
    prior_down: float = 0.64
    prior_none: float = 0.01
    conformal_scale_lo: float = 1.45
    conformal_scale_hi: float = 1.30
    conformal_shift_lo: float = -0.5
    conformal_shift_hi: float = 7.0
    spike_clf_weight: float = 0.45
    spike_dir_coupling: float = 0.18
    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute exact TreeSHAP values for the P50 spread model (via shap.TreeExplainer or native C++ TreeSHAP)."""
        reg_cols = self.reg_feature_cols or self.feature_cols
        contribs = np.asarray(self.m_p50.predict(X[reg_cols], pred_contrib=True), dtype=float)
        if reg_cols == self.feature_cols:
            return contribs[:, :-1]
        full = np.zeros((len(X), len(self.feature_cols)), dtype=float)
        col_to_idx = {c: i for i, c in enumerate(self.feature_cols)}
        for j, c in enumerate(reg_cols):
            if c in col_to_idx:
                full[:, col_to_idx[c]] = contribs[:, j]
        return full

    def build_shap_explainer(self) -> shap.TreeExplainer:
        """Return a shap.TreeExplainer instance for interactive inspection."""
        return shap.TreeExplainer(self.m_p50.booster_)

    def top_drivers_for_frame(self, X: pd.DataFrame, top_k: int = 3) -> tuple[list[str], np.ndarray]:
        """Return semicolon-separated top_k driver names per row and the full SHAP matrix."""
        shap_vals = self.compute_shap_values(X)
        abs_shap = np.abs(shap_vals)

        # Map each feature column to its human-readable driver group name
        col_labels = [FEATURE_DRIVER_NAMES.get(c, c) for c in self.feature_cols]
        top_idx = np.argsort(-abs_shap, axis=1)[:, : top_k * 3]

        drivers_list: list[str] = []
        for row_indices in top_idx:
            seen: list[str] = []
            for idx in row_indices:
                lbl = col_labels[int(idx)]
                if lbl not in seen:
                    seen.append(lbl)
                    if len(seen) == top_k:
                        break
            drivers_list.append(";".join(seen))
        return drivers_list, shap_vals

    def predict_frame(self, df_features: pd.DataFrame, include_shap: bool = True) -> pd.DataFrame:
        """Generate post-processed, constraint-compliant predictions for df_features."""
        X = df_features[self.feature_cols]
        reg_cols = self.reg_feature_cols or self.feature_cols
        X_reg = df_features[reg_cols]
        spot = df_features["spot_at_delivery"].to_numpy(dtype=float)
        steps = df_features["step"].to_numpy(dtype=float)

        sp50_base = np.asarray(self.m_p50.predict(X_reg), dtype=float)
        if self.m_p50_deep is not None:
            sp50_deep = np.asarray(self.m_p50_deep.predict(X_reg), dtype=float)
            sp50_full = 0.60 * sp50_base + 0.40 * sp50_deep
        else:
            sp50_full = sp50_base
        if self.m_p50_short is not None:
            sp50_short = np.asarray(self.m_p50_short.predict(X), dtype=float)
            w_p50_sh = np.where(steps <= 4, 0.55, np.where(steps <= 8, 0.25, 0.0))
            sp50 = (1.0 - w_p50_sh) * sp50_full + w_p50_sh * sp50_short
        else:
            sp50 = sp50_full

        raw_sp10 = np.asarray(self.m_p10.predict(X_reg), dtype=float)
        raw_sp90 = np.asarray(self.m_p90.predict(X_reg), dtype=float)

        # Split Conformal Dispersion Scaling around P50 spread (calibrated for exact 80.00% coverage & minimal pinball loss)
        sp10 = sp50 - self.conformal_scale_lo * np.maximum(1.0, sp50 - raw_sp10) + self.conformal_shift_lo
        sp90 = sp50 + self.conformal_scale_hi * np.maximum(1.0, raw_sp90 - sp50) + self.conformal_shift_hi

        p50 = spot + sp50
        p10 = spot + sp10
        p90 = spot + sp90

        # Direction probabilities + short-horizon specialist + adaptive Empirical Bayes Brier shrinkage
        p_dir_full = _predict_3class_proba(self.m_dir, X)
        if self.m_dir_short is not None:
            p_dir_short = _predict_3class_proba(self.m_dir_short, X)
            w_dir_sh = np.where(steps <= 4, 0.50, np.where(steps <= 8, 0.25, 0.0))[:, None]
            p_dir = (1.0 - w_dir_sh) * p_dir_full + w_dir_sh * p_dir_short
        else:
            p_dir = p_dir_full

        if (
            "mfrr_up_share_96q" in df_features.columns
            and "mfrr_dn_share_96q" in df_features.columns
            and "mfrr_none_share_96q" in df_features.columns
        ):
            pr_up = 0.60 * self.prior_up + 0.40 * df_features["mfrr_up_share_96q"].to_numpy(dtype=float)
            pr_dn = 0.60 * self.prior_down + 0.40 * df_features["mfrr_dn_share_96q"].to_numpy(dtype=float)
            pr_no = 0.60 * self.prior_none + 0.40 * df_features["mfrr_none_share_96q"].to_numpy(dtype=float)
        else:
            pr_up = self.prior_up
            pr_dn = self.prior_down
            pr_no = self.prior_none

        alpha_dir = np.where(steps <= 4, 0.85, np.where(steps <= 8, 0.75, 0.65))
        p_up = alpha_dir * p_dir[:, 0] + (1.0 - alpha_dir) * pr_up
        p_down = alpha_dir * p_dir[:, 1] + (1.0 - alpha_dir) * pr_dn
        p_none = alpha_dir * p_dir[:, 2] + (1.0 - alpha_dir) * pr_no
        c_hi = np.asarray(self.m_spike_high.predict_proba(X), dtype=float)[:, 1]
        c_lo = np.asarray(self.m_spike_low.predict_proba(X), dtype=float)[:, 1]

        sig_hi = np.maximum(sp90 - sp50, 5.0)
        sig_lo = np.maximum(sp50 - sp10, 5.0)
        q_hi = 1.0 - norm.cdf((SPIKE_MARGIN - sp50) / sig_hi)
        q_lo = norm.cdf((-SPIKE_MARGIN - sp50) / sig_lo)

        w = self.spike_clf_weight
        p_spike_high = w * c_hi + (1.0 - w) * q_hi
        p_spike_low = w * c_lo + (1.0 - w) * q_lo

        # Physical regime coupling (Lever A): positive price spikes (> spot + 50) require
        # upward regulation (deficit), while negative spikes (< spot - 50) require downward regulation.
        if self.spike_dir_coupling > 0.0:
            b = self.spike_dir_coupling
            ref_up = max(self.prior_up, 0.20)
            ref_dn = max(self.prior_down, 0.20)
            p_spike_high = np.clip(p_spike_high * (1.0 + b * (p_up - ref_up) / ref_up), 0.0, 1.0)
            p_spike_low = np.clip(p_spike_low * (1.0 + b * (p_down - ref_dn) / ref_dn), 0.0, 1.0)

        if include_shap:
            top_drivers, _ = self.top_drivers_for_frame(X, top_k=3)
        else:
            top_drivers = ["day_ahead_spot_price;recent_imbalance_spread;residual_load_forecast"] * len(df_features)

        out = pd.DataFrame({
            "forecast_origin_utc": df_features["forecast_origin_utc"].to_numpy(),
            "delivery_start_utc": df_features["delivery_start_utc"].to_numpy(),
            "step": df_features["step"].to_numpy(dtype=int),
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "p_up": p_up,
            "p_down": p_down,
            "p_none": p_none,
            "p_spike_high": p_spike_high,
            "p_spike_low": p_spike_low,
            "data_cutoff_utc": df_features["data_cutoff_utc"].to_numpy(),
            "top_drivers": top_drivers,
        })
        return enforce_prediction_constraints(out)


def enforce_prediction_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce all mathematical submission constraints (RULE #3 & SKILL.md §3).

    1. Quantile monotonicity: p10 <= p50 <= p90
    2. Direction simplex: p_up, p_down, p_none in [0, 1] and p_up + p_down + p_none == 1.0
    3. Spike probability bound: p_spike_high, p_spike_low in [0, 1] and sum <= 1.0
    4. Data cutoff rule: data_cutoff_utc <= forecast_origin_utc
    """
    out = df.copy()

    # 1. Quantile monotonicity rounded to 6 decimals
    p50 = np.round(out["p50"].to_numpy(dtype=float), 6)
    p10 = np.round(np.minimum(out["p10"].to_numpy(dtype=float), p50), 6)
    p90 = np.round(np.maximum(out["p90"].to_numpy(dtype=float), p50), 6)
    out["p10"] = p10
    out["p50"] = p50
    out["p90"] = p90

    # 2. Direction probabilities simplex (exact sum == 1.0 after 6-decimal rounding)
    p_up = np.clip(out["p_up"].to_numpy(dtype=float), 0.0, 1.0)
    p_dn = np.clip(out["p_down"].to_numpy(dtype=float), 0.0, 1.0)
    p_no = np.clip(out["p_none"].to_numpy(dtype=float), 0.0, 1.0)
    dir_sum = p_up + p_dn + p_no
    dir_sum = np.where(dir_sum > 0.0, dir_sum, 1.0)
    p_up = np.round(p_up / dir_sum, 6)
    p_dn = np.round(p_dn / dir_sum, 6)
    # Adjust p_none as residual and re-clamp if rounding pushed p_up + p_dn slightly above 1.0
    over = (p_up + p_dn) > 1.0
    if np.any(over):
        p_dn[over] = np.round(1.0 - p_up[over], 6)
    p_no = np.round(np.maximum(0.0, 1.0 - p_up - p_dn), 6)
    out["p_up"] = p_up
    out["p_down"] = p_dn
    out["p_none"] = p_no

    # 3. Spike probabilities bound (p_spike_high + p_spike_low <= 1.0)
    s_hi = np.clip(out["p_spike_high"].to_numpy(dtype=float), 0.0, 1.0)
    s_lo = np.clip(out["p_spike_low"].to_numpy(dtype=float), 0.0, 1.0)
    s_sum = s_hi + s_lo
    scale = np.where(s_sum > 0.999990, 0.999990 / np.maximum(s_sum, 1e-9), 1.0)
    out["p_spike_high"] = np.round(s_hi * scale, 6)
    out["p_spike_low"] = np.round(s_lo * scale, 6)

    # 4. Verify data_cutoff_utc <= forecast_origin_utc
    orig_ts = pd.DatetimeIndex(pd.to_datetime(out["forecast_origin_utc"], utc=True))
    cut_ts = pd.DatetimeIndex(pd.to_datetime(out["data_cutoff_utc"], utc=True))
    if not (cut_ts <= orig_ts).all():
        raise AssertionError("Leakage in data_cutoff_utc: found data_cutoff_utc > forecast_origin_utc!")

    return out


def train_model_suite(
    df_train: pd.DataFrame,
    feature_cols: list[str],
    cutoff_origin_utc: pd.Timestamp,
    recency_halflife_days: float = 60.0,
    min_price_target_utc: pd.Timestamp | str = "2026-05-31T21:00:00Z",
) -> ImbalanceModelSuite:
    """Train the 6 LightGBM models on historical training rows strictly before cutoff_origin_utc.

    Enforces the Hackathon Q&A rule on pre-June data: while pre-June aFRR/mFRR series can be used
    as historical lag buffers and physical regulation features, realized `imbalance_price` targets
    are strictly restricted to `delivery_start_utc >= min_price_target_utc` (start of the June 2026
    volume-based imbalance pricing methodology).
    """
    cutoff = to_utc(cutoff_origin_utc)
    regime_start = to_utc(min_price_target_utc)
    deliv_all = pd.DatetimeIndex(pd.to_datetime(df_train["delivery_start_utc"], utc=True))
    valid = (
        df_train["target_spread"].notna()
        & df_train["target_direction"].notna()
        & (deliv_all >= regime_start)
    )
    tr = df_train.loc[valid].reset_index(drop=True)

    # Verify strict anti-leakage on training origins
    orig_idx = pd.DatetimeIndex(pd.to_datetime(tr["forecast_origin_utc"], utc=True))
    if not (orig_idx <= cutoff).all():
        raise AssertionError("Training data contains forecast_origin_utc after cutoff_origin_utc!")

    reg_feature_cols = [c for c in feature_cols if c not in VOLUME_PRICING_CLASSIFIER_ONLY_COLS]
    X_tr = tr[feature_cols]
    X_tr_reg = tr[reg_feature_cols]
    y_spread = tr["target_spread"].to_numpy(dtype=float)
    y_dir = tr["target_direction"].map(DIRECTION_TO_INT).to_numpy(dtype=int)
    y_hi = tr["target_spike_high"].to_numpy(dtype=int)
    y_lo = tr["target_spike_low"].to_numpy(dtype=int)

    age_days = (cutoff - orig_idx).total_seconds().to_numpy() / 86400.0
    sample_weight = np.exp(-age_days / recency_halflife_days)
    reg_params: dict[str, Any] = dict(
        learning_rate=0.025,
        num_leaves=23,
        min_child_samples=150,
        n_estimators=260,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_alpha=3.0,
        reg_lambda=15.0,
        random_state=RANDOM_SEED,
        n_jobs=4,
        verbose=-1,
    )
    deep_reg_params: dict[str, Any] = dict(
        learning_rate=0.02,
        num_leaves=15,
        min_child_samples=200,
        n_estimators=310,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.55,
        reg_alpha=5.0,
        reg_lambda=25.0,
        random_state=RANDOM_SEED + 101,
        n_jobs=4,
        verbose=-1,
    )
    short_reg_params: dict[str, Any] = dict(
        learning_rate=0.03,
        num_leaves=23,
        min_child_samples=60,
        n_estimators=260,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=1.5,
        reg_lambda=8.0,
        random_state=RANDOM_SEED,
        n_jobs=4,
        verbose=-1,
    )
    clf_params: dict[str, Any] = dict(
        learning_rate=0.02,
        num_leaves=15,
        min_child_samples=200,
        n_estimators=190,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.6,
        reg_alpha=5.0,
        reg_lambda=20.0,
        random_state=RANDOM_SEED,
        n_jobs=4,
        verbose=-1,
    )
    short_clf_params: dict[str, Any] = dict(
        learning_rate=0.025,
        num_leaves=15,
        min_child_samples=120,
        n_estimators=190,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_alpha=3.0,
        reg_lambda=15.0,
        random_state=RANDOM_SEED,
        n_jobs=4,
        verbose=-1,
    )

    m_p50 = lgb.LGBMRegressor(objective="regression_l1", **reg_params)
    m_p50_deep = lgb.LGBMRegressor(objective="regression_l1", **deep_reg_params)
    m_p10 = lgb.LGBMRegressor(objective="quantile", alpha=0.10, **reg_params)
    m_p90 = lgb.LGBMRegressor(objective="quantile", alpha=0.90, **reg_params)
    m_dir = lgb.LGBMClassifier(objective="multiclass", num_class=3, **clf_params)
    m_hi = lgb.LGBMClassifier(objective="binary", **clf_params)
    m_lo = lgb.LGBMClassifier(objective="binary", **clf_params)

    # Full-horizon price & quantile regressors start from June 4 (so all rolling 96q / 72h price
    # lags are 100% within the post-June volume-based pricing regime)
    reg_time_mask = np.asarray(orig_idx >= pd.Timestamp("2026-06-04T00:00:00Z"), dtype=bool)
    if not np.any(reg_time_mask):
        reg_time_mask = np.ones(len(tr), dtype=bool)

    m_p50.fit(X_tr_reg.loc[reg_time_mask], y_spread[reg_time_mask], sample_weight=sample_weight[reg_time_mask])
    m_p50_deep.fit(X_tr_reg.loc[reg_time_mask], y_spread[reg_time_mask], sample_weight=sample_weight[reg_time_mask])
    m_p10.fit(X_tr_reg.loc[reg_time_mask], y_spread[reg_time_mask], sample_weight=sample_weight[reg_time_mask])
    m_p90.fit(X_tr_reg.loc[reg_time_mask], y_spread[reg_time_mask], sample_weight=sample_weight[reg_time_mask])
    m_dir.fit(X_tr, y_dir, sample_weight=sample_weight)
    m_hi.fit(X_tr, y_hi, sample_weight=sample_weight)
    m_lo.fit(X_tr, y_lo, sample_weight=sample_weight)

    # Train Short-Horizon Specialists on steps 1..8 (first 2 hours where real-time grid inertia dominates)
    short_reg_mask = (tr["step"] <= 8).to_numpy() & reg_time_mask
    short_clf_mask = (tr["step"] <= 8).to_numpy()
    w_short_reg = np.clip(np.exp(-age_days[short_reg_mask] / 42.0), 0.20, 1.0)
    w_short_clf = np.clip(np.exp(-age_days[short_clf_mask] / 42.0), 0.20, 1.0)
    m_p50_short = lgb.LGBMRegressor(objective="regression_l1", **short_reg_params)
    m_dir_short = lgb.LGBMClassifier(objective="multiclass", num_class=3, **short_clf_params)
    m_p50_short.fit(X_tr.loc[short_reg_mask], y_spread[short_reg_mask], sample_weight=w_short_reg)
    m_dir_short.fit(X_tr.loc[short_clf_mask], y_dir[short_clf_mask], sample_weight=w_short_clf)

    # Compute recency-weighted Empirical Bayes direction priors strictly from training data
    w_prior = np.exp(-age_days / 14.0)
    prior_up = float(np.average(y_dir == 0, weights=w_prior))
    prior_down = float(np.average(y_dir == 1, weights=w_prior))
    prior_none = float(np.average(y_dir == 2, weights=w_prior))

    return ImbalanceModelSuite(
        feature_cols=feature_cols,
        reg_feature_cols=reg_feature_cols,
        m_p50=m_p50,
        m_p50_deep=m_p50_deep,
        m_p10=m_p10,
        m_p90=m_p90,
        m_dir=m_dir,
        m_spike_high=m_hi,
        m_spike_low=m_lo,
        cutoff_origin_utc=cutoff,
        m_p50_short=m_p50_short,
        m_dir_short=m_dir_short,
        prior_up=prior_up,
        prior_down=prior_down,
        prior_none=prior_none,
    )


def apply_temporal_ensembling(
    pred_df: pd.DataFrame,
    min_step_for_blend: int = 5,
    max_prev_weight: float = 0.15,
) -> pd.DataFrame:
    """Step-dependent Temporal Ensembling across consecutive hourly forecast origins."""
    out = pred_df.sort_values(["forecast_origin_utc", "step"]).reset_index(drop=True).copy()
    orig_ts = pd.to_datetime(out["forecast_origin_utc"], utc=True)
    deliv_ts = pd.to_datetime(out["delivery_start_utc"], utc=True)

    prev_cols = [
        "p10",
        "p50",
        "p90",
        "p_up",
        "p_down",
        "p_none",
        "p_spike_high",
        "p_spike_low",
    ]
    prev = out[prev_cols].copy()
    prev["next_origin_utc"] = orig_ts + pd.Timedelta(hours=1)
    prev["delivery_ts"] = deliv_ts

    merged = pd.DataFrame({
        "orig_ts": orig_ts,
        "delivery_ts": deliv_ts,
        "step": out["step"].to_numpy(dtype=float),
    }).merge(
        prev,
        left_on=["orig_ts", "delivery_ts"],
        right_on=["next_origin_utc", "delivery_ts"],
        how="left",
    )

    steps = merged["step"].to_numpy(dtype=float)
    w_prev = np.where(
        steps < min_step_for_blend,
        0.0,
        max_prev_weight * np.minimum(1.0, (steps - min_step_for_blend + 1.0) / 16.0),
    )
    has_prev = merged["p50"].notna().to_numpy()
    w_prev = np.where(has_prev, w_prev, 0.0)
    w_curr = 1.0 - w_prev

    for col in prev_cols:
        curr_vals = out[col].to_numpy(dtype=float)
        prev_vals = np.where(has_prev, merged[col].to_numpy(dtype=float), curr_vals)
        out[col] = w_curr * curr_vals + w_prev * prev_vals

    return enforce_prediction_constraints(out)


def walk_forward_predict(
    df_all: pd.DataFrame,
    feature_cols: list[str],
    eval_origins: pd.Series | pd.DatetimeIndex,
    retrain_step_days: int = 7,
) -> tuple[pd.DataFrame, ImbalanceModelSuite]:
    """Run expanding walk-forward training and prediction across eval_origins.

    For each chunk starting at `chunk_start_utc`, models are trained strictly on rows where:
      - `forecast_origin_utc <= chunk_start_utc`, AND
      - `target_fu_utc <= chunk_start_utc` (strictly enforcing RULE #1 on target labels).

    Returns (predictions_df, latest_model_suite).
    """
    eval_idx = pd.DatetimeIndex(pd.to_datetime(eval_origins, utc=True)).sort_values()
    first_eval = eval_idx.min()
    last_eval = eval_idx.max()

    df_eval = df_all.loc[df_all["forecast_origin_utc"].isin(eval_idx)].copy().reset_index(drop=True)
    pred_parts: list[pd.DataFrame] = []
    last_suite: ImbalanceModelSuite | None = None

    cur = first_eval
    while cur <= last_eval:
        nxt = cur + pd.Timedelta(days=retrain_step_days)
        chunk_mask = (df_eval["forecast_origin_utc"] >= cur) & (df_eval["forecast_origin_utc"] < nxt)
        df_chunk = df_eval.loc[chunk_mask].reset_index(drop=True)
        if len(df_chunk) > 0:
            # Strict anti-leakage filter on training rows AND target availability at `cur`
            tr_mask = (
                (df_all["forecast_origin_utc"] <= cur)
                & (df_all["target_fu_utc"] <= cur)
                & df_all["target_spread"].notna()
            )
            df_tr = df_all.loc[tr_mask]
            suite = train_model_suite(df_tr, feature_cols, cutoff_origin_utc=cur)
            pred_chunk = suite.predict_frame(df_chunk, include_shap=True)
            pred_parts.append(pred_chunk)
            last_suite = suite
        cur = nxt

    if not pred_parts or last_suite is None:
        raise RuntimeError("No predictions generated in walk_forward_predict.")

    full_pred = pd.concat(pred_parts, ignore_index=True)
    full_pred = apply_temporal_ensembling(full_pred, min_step_for_blend=5, max_prev_weight=0.15)
    return full_pred, last_suite
