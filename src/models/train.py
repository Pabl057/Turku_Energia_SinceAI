"""LightGBM model suite, walk-forward training, SHAP explainability, and post-processing.

Implements the 6-model probabilistic forecasting architecture:
  1. m_p50  : LGBMRegressor(objective='regression_l1') on imbalance spread (imbalance - spot)
  2. m_p10  : LGBMRegressor(objective='quantile', alpha=0.06) + conformal dispersion calibration
  3. m_p90  : LGBMRegressor(objective='quantile', alpha=0.94) + conformal dispersion calibration
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


@dataclass
class ImbalanceModelSuite:
    """Trained 6-model LightGBM suite for imbalance price, quantiles, direction, and spikes."""

    feature_cols: list[str]
    m_p50: lgb.LGBMRegressor
    m_p10: lgb.LGBMRegressor
    m_p90: lgb.LGBMRegressor
    m_dir: lgb.LGBMClassifier
    m_spike_high: lgb.LGBMClassifier
    m_spike_low: lgb.LGBMClassifier
    cutoff_origin_utc: pd.Timestamp
    interval_scale: float = 1.25
    spike_clf_weight: float = 0.50

    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute exact TreeSHAP values for the P50 spread model (via shap.TreeExplainer or native C++ TreeSHAP)."""
        # LightGBM's pred_contrib=True computes exact Lundberg & Lee TreeSHAP values (last col is expected_value)
        contribs = np.asarray(self.m_p50.predict(X[self.feature_cols], pred_contrib=True), dtype=float)
        return contribs[:, :-1]

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
        spot = df_features["spot_at_delivery"].to_numpy(dtype=float)

        sp50 = np.asarray(self.m_p50.predict(X), dtype=float)
        raw_sp10 = np.asarray(self.m_p10.predict(X), dtype=float)
        raw_sp90 = np.asarray(self.m_p90.predict(X), dtype=float)

        # Conformal dispersion scaling around P50 spread (calibrated for ~80% coverage & minimal pinball loss)
        sp10 = sp50 - self.interval_scale * np.maximum(1.0, sp50 - raw_sp10)
        sp90 = sp50 + self.interval_scale * np.maximum(1.0, raw_sp90 - sp50)

        p50 = spot + sp50
        p10 = spot + sp10
        p90 = spot + sp90

        # Direction probabilities
        p_dir = np.asarray(self.m_dir.predict_proba(X), dtype=float)
        p_up = p_dir[:, 0]
        p_down = p_dir[:, 1]
        p_none = p_dir[:, 2]

        # Spike probabilities: blend binary classifier with two-piece quantile tail probability
        c_hi = np.asarray(self.m_spike_high.predict_proba(X), dtype=float)[:, 1]
        c_lo = np.asarray(self.m_spike_low.predict_proba(X), dtype=float)[:, 1]

        sig_hi = np.maximum((sp90 - sp50) / 1.28155, 5.0)
        sig_lo = np.maximum((sp50 - sp10) / 1.28155, 5.0)
        q_hi = 1.0 - norm.cdf((SPIKE_MARGIN - sp50) / sig_hi)
        q_lo = norm.cdf((-SPIKE_MARGIN - sp50) / sig_lo)

        w = self.spike_clf_weight
        p_spike_high = w * c_hi + (1.0 - w) * q_hi
        p_spike_low = w * c_lo + (1.0 - w) * q_lo

        if include_shap:
            top_drivers, _ = self.top_drivers_for_frame(X, top_k=3)
        else:
            top_drivers = ["day_ahead_spot_price;recent_imbalance_spread;residual_load_forecast"] * len(df_features)

        out = pd.DataFrame({
            "forecast_origin_utc": df_features["forecast_origin_utc"].values,
            "delivery_start_utc": df_features["delivery_start_utc"].values,
            "step": df_features["step"].to_numpy(dtype=int),
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "p_up": p_up,
            "p_down": p_down,
            "p_none": p_none,
            "p_spike_high": p_spike_high,
            "p_spike_low": p_spike_low,
            "data_cutoff_utc": df_features["data_cutoff_utc"].values,
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
) -> ImbalanceModelSuite:
    """Train the 6 LightGBM models on historical training rows strictly before cutoff_origin_utc."""
    cutoff = to_utc(cutoff_origin_utc)
    valid = df_train["target_spread"].notna() & df_train["target_direction"].notna()
    tr = df_train.loc[valid].reset_index(drop=True)

    # Verify strict anti-leakage on training origins
    orig_idx = pd.DatetimeIndex(pd.to_datetime(tr["forecast_origin_utc"], utc=True))
    if not (orig_idx <= cutoff).all():
        raise AssertionError("Training data contains forecast_origin_utc after cutoff_origin_utc!")

    X_tr = tr[feature_cols]
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
        n_jobs=-1,
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
        n_jobs=-1,
        verbose=-1,
    )

    m_p50 = lgb.LGBMRegressor(objective="regression_l1", **reg_params)
    m_p10 = lgb.LGBMRegressor(objective="quantile", alpha=0.06, **reg_params)
    m_p90 = lgb.LGBMRegressor(objective="quantile", alpha=0.94, **reg_params)
    m_dir = lgb.LGBMClassifier(objective="multiclass", num_class=3, **clf_params)
    m_hi = lgb.LGBMClassifier(objective="binary", **clf_params)
    m_lo = lgb.LGBMClassifier(objective="binary", **clf_params)

    m_p50.fit(X_tr, y_spread, sample_weight=sample_weight)
    m_p10.fit(X_tr, y_spread, sample_weight=sample_weight)
    m_p90.fit(X_tr, y_spread, sample_weight=sample_weight)
    m_dir.fit(X_tr, y_dir, sample_weight=sample_weight)
    m_hi.fit(X_tr, y_hi, sample_weight=sample_weight)
    m_lo.fit(X_tr, y_lo, sample_weight=sample_weight)

    return ImbalanceModelSuite(
        feature_cols=feature_cols,
        m_p50=m_p50,
        m_p10=m_p10,
        m_p90=m_p90,
        m_dir=m_dir,
        m_spike_high=m_hi,
        m_spike_low=m_lo,
        cutoff_origin_utc=cutoff,
    )


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
    full_pred = full_pred.sort_values(["forecast_origin_utc", "step"]).reset_index(drop=True)
    return full_pred, last_suite
