"""LightGBM 6-model suite, post-processing constraints, temporal ensembling, and SHAP explanations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap

from src.features.build_features import FEATURE_COLUMNS

RANDOM_SEED = 42


def enforce_postprocessing_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce all mathematical submission constraints on predictions."""
    out = df.copy()

    # 1. Quantile Monotonicity: p10 <= p50 <= p90
    p50 = out["p50"].to_numpy(dtype=np.float64)
    p10 = np.minimum(out["p10"].to_numpy(dtype=np.float64), p50)
    p90 = np.maximum(out["p90"].to_numpy(dtype=np.float64), p50)
    out["p10"] = np.round(p10, 6)
    out["p50"] = np.round(p50, 6)
    out["p90"] = np.round(p90, 6)

    # 2. Direction probabilities in [0, 1] and sum == 1.0
    p_up = np.clip(out["p_up"].to_numpy(dtype=np.float64), 1e-6, 1.0)
    p_down = np.clip(out["p_down"].to_numpy(dtype=np.float64), 1e-6, 1.0)
    p_none = np.clip(out["p_none"].to_numpy(dtype=np.float64), 1e-6, 1.0)
    dir_sum = p_up + p_down + p_none
    p_up = np.round(p_up / dir_sum, 6)
    p_down = np.round(p_down / dir_sum, 6)
    p_none = np.round(1.0 - p_up - p_down, 6)
    # Handle tiny rounding edge cases where 1 - p_up - p_down < 0
    neg_mask = p_none < 0.0
    if np.any(neg_mask):
        p_none[neg_mask] = 0.0
        p_up[neg_mask] = np.round(1.0 - p_down[neg_mask], 6)

    out["p_up"] = p_up
    out["p_down"] = p_down
    out["p_none"] = p_none

    # 3. Spike probabilities in [0, 1] and p_spike_high + p_spike_low <= 1.0
    p_high = np.clip(out["p_spike_high"].to_numpy(dtype=np.float64), 0.0, 1.0)
    p_low = np.clip(out["p_spike_low"].to_numpy(dtype=np.float64), 0.0, 1.0)
    spike_sum = p_high + p_low
    exceed = spike_sum > 0.999
    if np.any(exceed):
        scale = 0.999 / spike_sum[exceed]
        p_high[exceed] *= scale
        p_low[exceed] *= scale
    out["p_spike_high"] = np.round(p_high, 6)
    out["p_spike_low"] = np.round(p_low, 6)

    return out


def apply_temporal_ensembling(
    pred_df: pd.DataFrame,
    min_step_for_blend: int = 5,
    max_prev_weight: float = 0.18,
) -> pd.DataFrame:
    """Step-dependent Temporal Ensembling across consecutive hourly forecast origins (Hebel 2).

    For steps 1..4 (first hour): weight on previous origin = 0.0 (100% fresh real-time run).
    For steps >= min_step_for_blend: smoothly blends current origin prediction at `step`
    with previous hourly origin prediction (t - 1h) at `step + 4` for the exact same `delivery_start_utc`.
    """
    out = pred_df.sort_values(["forecast_origin_utc", "step"]).reset_index(drop=True).copy()

    # Build lookup from previous origin (origin + 1h == current_origin, same delivery_start_utc)
    prev = out[
        [
            "forecast_origin_utc",
            "delivery_start_utc",
            "p10",
            "p50",
            "p90",
            "p_up",
            "p_down",
            "p_none",
            "p_spike_high",
            "p_spike_low",
        ]
    ].copy()
    prev["next_origin_utc"] = prev["forecast_origin_utc"] + pd.Timedelta(hours=1)

    merged = out.merge(
        prev,
        left_on=["forecast_origin_utc", "delivery_start_utc"],
        right_on=["next_origin_utc", "delivery_start_utc"],
        how="left",
        suffixes=("", "_prev"),
    )

    # Step-dependent weight w_prev(step): 0 for steps 1..min_step-1, ramping up to max_prev_weight
    steps = merged["step"].to_numpy(dtype=np.float64)
    w_prev = np.where(
        steps < min_step_for_blend,
        0.0,
        max_prev_weight * np.minimum(1.0, (steps - min_step_for_blend + 1.0) / 16.0),
    )
    has_prev = merged["p50_prev"].notna().to_numpy()
    w_prev = np.where(has_prev, w_prev, 0.0)
    w_curr = 1.0 - w_prev

    for col in ("p10", "p50", "p90", "p_up", "p_down", "p_none", "p_spike_high", "p_spike_low"):
        curr_vals = merged[col].to_numpy(dtype=np.float64)
        prev_vals = np.where(has_prev, merged[f"{col}_prev"].to_numpy(dtype=np.float64), curr_vals)
        out[col] = w_curr * curr_vals + w_prev * prev_vals

    return enforce_postprocessing_constraints(out)


class ImbalanceForecasterSuite:
    """Suite of 6 LightGBM models for P10, P50, P90, mFRR Direction, and High/Low Spikes."""

    def __init__(self, feature_cols: list[str] | None = None, random_seed: int = RANDOM_SEED):
        self.feature_cols = feature_cols or FEATURE_COLUMNS
        self.random_seed = random_seed

        common_reg_params = dict(
            n_estimators=450,
            learning_rate=0.035,
            num_leaves=31,
            max_depth=6,
            min_child_samples=60,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=5.0,
            random_state=random_seed,
            n_jobs=-1,
            verbose=-1,
        )

        # Model 1, 2, 3: Quantile Regressors (trained on spread = y_imbalance_price - spot_eff)
        self.model_p10 = lgb.LGBMRegressor(objective="quantile", alpha=0.10, **common_reg_params)
        self.model_p50 = lgb.LGBMRegressor(objective="quantile", alpha=0.50, **common_reg_params)
        self.model_p90 = lgb.LGBMRegressor(objective="quantile", alpha=0.90, **common_reg_params)

        # Direct price P50 model (blended with spread P50 model for extra stability)
        self.model_p50_direct = lgb.LGBMRegressor(objective="quantile", alpha=0.50, **common_reg_params)

        common_clf_params = dict(
            n_estimators=350,
            learning_rate=0.035,
            num_leaves=23,
            max_depth=5,
            min_child_samples=80,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=5.0,
            random_state=random_seed,
            n_jobs=-1,
            verbose=-1,
        )

        # Model 4: Multiclass mFRR Direction (-1 -> 0: down, 0 -> 1: none, +1 -> 2: up)
        self.model_dir = lgb.LGBMClassifier(objective="multiclass", num_class=3, **common_clf_params)

        # Model 5 & 6: Binary Spike Classifiers
        self.model_spike_high = lgb.LGBMClassifier(objective="binary", **common_clf_params)
        self.model_spike_low = lgb.LGBMClassifier(objective="binary", **common_clf_params)

        self.explainer_p50: shap.TreeExplainer | None = None

    def fit(self, train_df: pd.DataFrame) -> "ImbalanceForecasterSuite":
        """Train all 6 LightGBM models on leak-filtered training rows."""
        # Train price models on valid target rows
        valid_price = train_df["y_imbalance_price"].notna() & train_df["spot_eff"].notna()
        df_p = train_df.loc[valid_price]
        X_p = df_p[self.feature_cols]
        y_spread = (df_p["y_imbalance_price"] - df_p["spot_eff"]).to_numpy(dtype=np.float64)
        y_imb = df_p["y_imbalance_price"].to_numpy(dtype=np.float64)

        self.model_p10.fit(X_p, y_spread)
        self.model_p50.fit(X_p, y_spread)
        self.model_p90.fit(X_p, y_spread)
        self.model_p50_direct.fit(X_p, y_imb)

        # Train direction model (-1 -> 0: down, 0 -> 1: none, +1 -> 2: up)
        valid_dir = train_df["y_direction"].isin([-1.0, 0.0, 1.0])
        df_d = train_df.loc[valid_dir]
        X_d = df_d[self.feature_cols]
        y_dir_mapped = df_d["y_direction"].map({-1.0: 0, 0.0: 1, 1.0: 2}).to_numpy(dtype=np.int32)
        self.model_dir.fit(X_d, y_dir_mapped)

        # Train spike models
        valid_spike = train_df["y_spike_high"].notna() & train_df["y_spike_low"].notna()
        df_s = train_df.loc[valid_spike]
        X_s = df_s[self.feature_cols]
        self.model_spike_high.fit(X_s, df_s["y_spike_high"].to_numpy(dtype=np.int32))
        self.model_spike_low.fit(X_s, df_s["y_spike_low"].to_numpy(dtype=np.int32))

        self.explainer_p50 = shap.TreeExplainer(self.model_p50)
        return self

    def compute_top_drivers(self, X: pd.DataFrame, top_k: int = 3) -> list[str]:
        """Compute top-k SHAP feature drivers per row using LightGBM native C++ TreeSHAP (pred_contrib=True)."""
        # pred_contrib returns shape (n_rows, n_features + 1); last column is expected value bias
        shap_vals = self.model_p50.predict(X, pred_contrib=True)[:, :-1]
        abs_shap = np.abs(shap_vals)
        top_idx = np.argsort(-abs_shap, axis=1)[:, :top_k]
        col_names = np.array(self.feature_cols)
        drivers = [";".join(col_names[row_idx]) for row_idx in top_idx]
        return drivers

    def predict(
        self,
        feat_df: pd.DataFrame,
        compute_shap: bool = True,
        temporal_ensemble: bool = True,
    ) -> pd.DataFrame:
        """Generate predictions for all 8 required target columns + data_cutoff_utc + top_drivers."""
        X = feat_df[self.feature_cols]
        spot_eff = feat_df["spot_eff"].to_numpy(dtype=np.float64)

        pred_spread_p10 = self.model_p10.predict(X)
        pred_spread_p50 = self.model_p50.predict(X)
        pred_spread_p90 = self.model_p90.predict(X)
        pred_direct_p50 = self.model_p50_direct.predict(X)

        # Reconstruct prices from spot_eff + spread, blending P50 with direct P50
        p50 = 0.70 * (spot_eff + pred_spread_p50) + 0.30 * pred_direct_p50
        p10 = spot_eff + pred_spread_p10
        p90 = spot_eff + pred_spread_p90

        # Direction probabilities (classes: 0=down, 1=none, 2=up)
        dir_probs = self.model_dir.predict_proba(X)
        classes = list(self.model_dir.classes_)
        p_down = dir_probs[:, classes.index(0)] if 0 in classes else np.zeros(len(X))
        p_none = dir_probs[:, classes.index(1)] if 1 in classes else np.zeros(len(X))
        p_up = dir_probs[:, classes.index(2)] if 2 in classes else np.zeros(len(X))

        # Spike probabilities
        p_spike_high = self.model_spike_high.predict_proba(X)[:, 1]
        p_spike_low = self.model_spike_low.predict_proba(X)[:, 1]

        top_drivers = (
            self.compute_top_drivers(X, top_k=3)
            if compute_shap
            else ["spot_eff;imb_last;res_load_fc"] * len(X)
        )

        out = pd.DataFrame(
            {
                "forecast_origin_utc": feat_df["forecast_origin_utc"],
                "delivery_start_utc": feat_df["delivery_start_utc"],
                "step": feat_df["step"],
                "p10": p10,
                "p50": p50,
                "p90": p90,
                "p_up": p_up,
                "p_down": p_down,
                "p_none": p_none,
                "p_spike_high": p_spike_high,
                "p_spike_low": p_spike_low,
                "data_cutoff_utc": feat_df["data_cutoff_utc"],
                "top_drivers": top_drivers,
            }
        )

        out = enforce_postprocessing_constraints(out)
        if temporal_ensemble:
            out = apply_temporal_ensembling(out)
        return out
