"""LightGBM Fine-Tuned Model Suite with:
1. Horizon-Specialist Short-Term P50 Model (Steps 1-8) + Full-Horizon Suite
2. Exponential Recency Sample Weighting (Autumn regime focus)
3. Out-of-Time Split Conformal Quantile Calibration (~80% P10-P90 coverage)
4. Step-Dependent Empirical Bayes Probability Calibration (minimizing Brier scores)
5. Step-Dependent Temporal Ensembling & Native C++ TreeSHAP top_drivers
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

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
    """Step-dependent Temporal Ensembling across consecutive hourly forecast origins (Hebel 2)."""
    out = pred_df.sort_values(["forecast_origin_utc", "step"]).reset_index(drop=True).copy()

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
    """Fine-tuned LightGBM model suite with Horizon Specialists, Recency Weights & Conformal/Brier Calibration."""

    def __init__(self, feature_cols: list[str] | None = None, random_seed: int = RANDOM_SEED):
        self.feature_cols = feature_cols or FEATURE_COLUMNS
        self.random_seed = random_seed

        common_reg_params = dict(
            n_estimators=400,
            learning_rate=0.04,
            num_leaves=31,
            max_depth=6,
            min_child_samples=60,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=5.0,
            random_state=random_seed,
            n_jobs=4,
            verbose=-1,
        )

        # Tail Quantile Regressors (lighter leaf regularization so tail quantiles expand naturally in volatile hours)
        tail_reg_params = {
            **common_reg_params,
            "num_leaves": 42,
            "max_depth": 7,
            "min_child_samples": 30,
            "reg_alpha": 0.2,
            "reg_lambda": 1.0,
        }
        self.model_p10 = lgb.LGBMRegressor(objective="quantile", alpha=0.10, **tail_reg_params)
        self.model_p50 = lgb.LGBMRegressor(objective="quantile", alpha=0.50, **common_reg_params)
        self.model_p90 = lgb.LGBMRegressor(objective="quantile", alpha=0.90, **tail_reg_params)

        # Direct price P50 model
        self.model_p50_direct = lgb.LGBMRegressor(objective="quantile", alpha=0.50, **common_reg_params)

        # Short-Horizon Specialist P50 Model (Steps 1-8: first 2 hours where real-time aFRR/mFRR dominates)
        short_reg_params = {**common_reg_params, "n_estimators": 350, "num_leaves": 23, "max_depth": 5, "min_child_samples": 40}
        self.model_p50_short = lgb.LGBMRegressor(objective="quantile", alpha=0.50, **short_reg_params)

        common_clf_params = dict(
            n_estimators=260,
            learning_rate=0.035,
            num_leaves=19,
            max_depth=5,
            min_child_samples=100,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=2.0,
            reg_lambda=8.0,
            random_state=random_seed,
            n_jobs=4,
            verbose=-1,
        )

        # Multiclass mFRR Direction (-1 -> 0: down, 0 -> 1: none, +1 -> 2: up)
        self.model_dir = lgb.LGBMClassifier(objective="multiclass", num_class=3, **common_clf_params)
        # Short-Horizon Direction Specialist (Steps 1-8)
        self.model_dir_short = lgb.LGBMClassifier(objective="multiclass", num_class=3, **common_clf_params)

        # Binary Spike Classifiers
        self.model_spike_high = lgb.LGBMClassifier(objective="binary", **common_clf_params)
        self.model_spike_low = lgb.LGBMClassifier(objective="binary", **common_clf_params)

        # Out-of-time Split Conformal Dispersion Scaling parameters (target ~80% coverage & minimum Pinball loss)
        self.conformal_scale_lo: float = 1.60
        self.conformal_scale_hi: float = 1.25
        self.conformal_shift_lo: float = -2.0
        self.conformal_shift_hi: float = 8.0

        # Historical base rates for Empirical Bayes Brier calibration (computed strictly on train_df)
        self.prior_up: float = 0.3526
        self.prior_down: float = 0.6461
        self.prior_none: float = 0.0013
        self.prior_spike_high: float = 0.081
        self.prior_spike_low: float = 0.057

    @staticmethod
    def _compute_recency_weights(origins: pd.Series, half_life_days: float = 42.0) -> np.ndarray:
        """Compute exponential time-decay sample weights so recent market regimes have higher weight."""
        max_ts = origins.max()
        age_days = (max_ts - origins).dt.total_seconds().to_numpy(dtype=np.float64) / 86400.0
        weights = np.exp(-age_days / half_life_days)
        return np.clip(weights, 0.20, 1.0)

    def fit(self, train_df: pd.DataFrame) -> "ImbalanceForecasterSuite":
        """Train all LightGBM models with recency weights and horizon specialists."""
        valid_price = train_df["y_imbalance_price"].notna() & train_df["spot_eff"].notna()
        df_p = train_df.loc[valid_price].sort_values(["forecast_origin_utc", "step"]).reset_index(drop=True)

        # Fit main price models with recency weights
        X_p = df_p[self.feature_cols]
        y_spread = (df_p["y_imbalance_price"] - df_p["spot_eff"]).to_numpy(dtype=np.float64)
        y_imb = df_p["y_imbalance_price"].to_numpy(dtype=np.float64)
        w_p = self._compute_recency_weights(df_p["forecast_origin_utc"])

        self.model_p10.fit(X_p, y_spread, sample_weight=w_p)
        self.model_p50.fit(X_p, y_spread, sample_weight=w_p)
        self.model_p90.fit(X_p, y_spread, sample_weight=w_p)
        self.model_p50_direct.fit(X_p, y_imb, sample_weight=w_p)

        # Fit Short-Horizon Specialist (steps 1..8)
        short_mask = df_p["step"] <= 8
        df_p_short = df_p.loc[short_mask]
        self.model_p50_short.fit(
            df_p_short[self.feature_cols],
            (df_p_short["y_imbalance_price"] - df_p_short["spot_eff"]).to_numpy(dtype=np.float64),
            sample_weight=self._compute_recency_weights(df_p_short["forecast_origin_utc"]),
        )

        # Fit Direction models & compute historical priors strictly from train_df
        valid_dir = train_df["y_direction"].isin([-1.0, 0.0, 1.0])
        df_d = train_df.loc[valid_dir]
        X_d = df_d[self.feature_cols]
        y_dir_mapped = df_d["y_direction"].map({-1.0: 0, 0.0: 1, 1.0: 2}).to_numpy(dtype=np.int32)
        w_d = self._compute_recency_weights(df_d["forecast_origin_utc"])

        self.prior_down = float(np.average(y_dir_mapped == 0, weights=w_d))
        self.prior_none = float(np.average(y_dir_mapped == 1, weights=w_d))
        self.prior_up = float(np.average(y_dir_mapped == 2, weights=w_d))

        self.model_dir.fit(X_d, y_dir_mapped, sample_weight=w_d)

        df_d_short = df_d.loc[df_d["step"] <= 8]
        y_dir_short = df_d_short["y_direction"].map({-1.0: 0, 0.0: 1, 1.0: 2}).to_numpy(dtype=np.int32)
        self.model_dir_short.fit(
            df_d_short[self.feature_cols],
            y_dir_short,
            sample_weight=self._compute_recency_weights(df_d_short["forecast_origin_utc"]),
        )

        # Fit Spike models & compute historical base rates
        valid_spike = train_df["y_spike_high"].notna() & train_df["y_spike_low"].notna()
        df_s = train_df.loc[valid_spike]
        X_s = df_s[self.feature_cols]
        y_sh = df_s["y_spike_high"].to_numpy(dtype=np.int32)
        y_sl = df_s["y_spike_low"].to_numpy(dtype=np.int32)
        w_s = self._compute_recency_weights(df_s["forecast_origin_utc"])

        self.prior_spike_high = float(np.average(y_sh, weights=w_s))
        self.prior_spike_low = float(np.average(y_sl, weights=w_s))

        self.model_spike_high.fit(X_s, y_sh, sample_weight=w_s)
        self.model_spike_low.fit(X_s, y_sl, sample_weight=w_s)

        return self

    def compute_birnbaum_importance(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Compute Birnbaum Reliability Importance Measures (I^B_j = dP(Spike)/dq_j) and Criticality Importance (I^CR_j).

        In reliability engineering (Birnbaum, 1969), the Birnbaum Importance I^B_j of component j
        measures the rate of change of the system failure/spike probability P(Spike) with respect
        to component j entering a stressed/critical state:
            I^B_j = P(Spike | X_j = stressed) - P(Spike | X_j = normal) = p_spike * (1 - p_spike) * phi_j
        where phi_j is the log-odds TreeSHAP contribution of feature j in the spike classifiers.
        """
        sh_contrib = self.model_spike_high.booster_.predict(X, pred_contrib=True, num_threads=1)
        sl_contrib = self.model_spike_low.booster_.predict(X, pred_contrib=True, num_threads=1)

        sh_logodds = np.sum(sh_contrib, axis=1, keepdims=True)
        sl_logodds = np.sum(sl_contrib, axis=1, keepdims=True)
        p_sh = 1.0 / (1.0 + np.exp(-np.clip(sh_logodds, -20.0, 20.0)))
        p_sl = 1.0 / (1.0 + np.exp(-np.clip(sl_logodds, -20.0, 20.0)))

        # Jacobian factor dP/d(log-odds) = P * (1 - P) converting log-odds sensitivity into probability space I^B_j
        jac_sh = p_sh * (1.0 - p_sh)
        jac_sl = p_sl * (1.0 - p_sl)

        birnbaum_high = np.abs(sh_contrib[:, :-1]) * jac_sh
        birnbaum_low = np.abs(sl_contrib[:, :-1]) * jac_sl
        birnbaum_total = birnbaum_high + birnbaum_low

        # Criticality Importance I^CR_j normalized per row
        row_sum = np.maximum(np.sum(birnbaum_total, axis=1, keepdims=True), 1e-9)
        criticality_imp = birnbaum_total / row_sum
        return birnbaum_total, criticality_imp

    def compute_top_drivers(self, X: pd.DataFrame, top_k: int = 3) -> list[str]:
        """Compute top-k feature drivers per row combining Price TreeSHAP and Birnbaum Spike Importance (I^B)."""
        shap_vals = self.model_p50.predict(X, pred_contrib=True, num_threads=1)[:, :-1]
        abs_shap = np.abs(shap_vals)
        shap_norm = abs_shap / np.maximum(np.sum(abs_shap, axis=1, keepdims=True), 1e-9)

        _, birnbaum_crit = self.compute_birnbaum_importance(X)

        # Hybrid score: 65% P50 price sensitivity + 35% Birnbaum Spike Reliability Importance
        combined_score = 0.65 * shap_norm + 0.35 * birnbaum_crit
        top_idx = np.argsort(-combined_score, axis=1)[:, :top_k]
        col_names = np.array(self.feature_cols)
        drivers = [";".join(col_names[row_idx]) for row_idx in top_idx]
        return drivers

    def predict(
        self,
        feat_df: pd.DataFrame,
        compute_shap: bool = True,
        temporal_ensemble: bool = True,
    ) -> pd.DataFrame:
        """Generate calibrated predictions for all 8 target columns + data_cutoff_utc + top_drivers."""
        X = feat_df[self.feature_cols]
        spot_eff = feat_df["spot_eff"].to_numpy(dtype=np.float64)
        steps = feat_df["step"].to_numpy(dtype=np.float64)

        pred_spread_p10 = self.model_p10.predict(X)
        pred_spread_p50_full = self.model_p50.predict(X)
        pred_spread_p50_short = self.model_p50_short.predict(X)
        pred_spread_p90 = self.model_p90.predict(X)
        pred_direct_p50 = self.model_p50_direct.predict(X)

        # Blend Short-Horizon Specialist (steps 1..8) with Full-Horizon P50
        short_weight = np.where(steps <= 4, 0.55, np.where(steps <= 8, 0.30, 0.0))
        pred_spread_p50 = (1.0 - short_weight) * pred_spread_p50_full + short_weight * pred_spread_p50_short

        # Reconstruct P50 with gentle spread damping (0.90) toward Spot on medium/long horizons
        raw_p50 = 0.72 * (spot_eff + pred_spread_p50) + 0.28 * pred_direct_p50
        damping = np.where(steps <= 4, 0.96, 0.90)
        p50 = damping * raw_p50 + (1.0 - damping) * spot_eff

        raw_p10 = np.minimum(spot_eff + pred_spread_p10, p50)
        raw_p90 = np.maximum(spot_eff + pred_spread_p90, p50)

        # Apply Conformal Quantile Calibration (targets ~80% coverage & minimizes Pinball loss)
        p10 = p50 - self.conformal_scale_lo * (p50 - raw_p10) + self.conformal_shift_lo
        p90 = p50 + self.conformal_scale_hi * (raw_p90 - p50) + self.conformal_shift_hi

        # Direction probabilities (classes: 0=down, 1=none, 2=up) + Short-Horizon Specialist + Empirical Bayes Shrinkage
        dir_probs_full = self.model_dir.predict_proba(X)
        dir_probs_short = self.model_dir_short.predict_proba(X)
        w_dir_short = np.where(steps <= 4, 0.50, np.where(steps <= 8, 0.25, 0.0))[:, None]
        dir_probs = (1.0 - w_dir_short) * dir_probs_full + w_dir_short * dir_probs_short

        classes = list(self.model_dir.classes_)
        raw_down = dir_probs[:, classes.index(0)] if 0 in classes else np.zeros(len(X))
        raw_none = dir_probs[:, classes.index(1)] if 1 in classes else np.zeros(len(X))
        raw_up = dir_probs[:, classes.index(2)] if 2 in classes else np.zeros(len(X))

        # Step-dependent Brier shrinkage: keep sharp at steps 1-4 (alpha=0.82), stronger shrinkage at steps 9-40 (alpha=0.60)
        alpha_dir = np.where(steps <= 4, 0.82, np.where(steps <= 8, 0.70, 0.60))
        p_up = alpha_dir * raw_up + (1.0 - alpha_dir) * self.prior_up
        p_down = alpha_dir * raw_down + (1.0 - alpha_dir) * self.prior_down
        p_none = alpha_dir * raw_none + (1.0 - alpha_dir) * self.prior_none

        # Spike probabilities: Binary Classifier + Two-Piece Quantile Tail Probability + Brier Shrinkage
        raw_sh = self.model_spike_high.predict_proba(X)[:, 1]
        raw_sl = self.model_spike_low.predict_proba(X)[:, 1]

        # Analytical Two-Piece Exponential Quantile Tail Probabilities P(Y > spot + 50) & P(Y < spot - 50)
        ln5 = float(np.log(5.0))  # 0.50 * exp(-ln(5) * 1.0) == 0.10 at P90 / P10
        sigma_hi = np.maximum(p90 - p50, 3.0)
        sigma_lo = np.maximum(p50 - p10, 3.0)

        thresh_hi = spot_eff + 50.0
        dist_hi = thresh_hi - p50
        p_tail_high = np.where(
            dist_hi >= 0.0,
            0.50 * np.exp(-ln5 * np.minimum(dist_hi / sigma_hi, 10.0)),
            1.0 - 0.50 * np.exp(-ln5 * np.minimum(-dist_hi / sigma_lo, 10.0)),
        )

        thresh_lo = spot_eff - 50.0
        dist_lo = p50 - thresh_lo
        p_tail_low = np.where(
            dist_lo >= 0.0,
            0.50 * np.exp(-ln5 * np.minimum(dist_lo / sigma_lo, 10.0)),
            1.0 - 0.50 * np.exp(-ln5 * np.minimum(-dist_lo / sigma_hi, 10.0)),
        )

        blended_sh = 0.80 * raw_sh + 0.20 * p_tail_high
        blended_sl = 0.80 * raw_sl + 0.20 * p_tail_low

        alpha_spike = np.where(steps <= 4, 0.92, 0.86)
        p_spike_high = alpha_spike * blended_sh + (1.0 - alpha_spike) * self.prior_spike_high
        p_spike_low = alpha_spike * blended_sl + (1.0 - alpha_spike) * self.prior_spike_low

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
