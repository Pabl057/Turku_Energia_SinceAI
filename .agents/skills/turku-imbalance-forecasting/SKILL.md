---
name: turku-imbalance-forecasting
description: >-
  Use this skill when developing data pipelines, feature engineering, LightGBM/quantile
  forecasting models, SHAP explanations, validation scripts, or generating submissions
  (`forecast_backtest.csv`) for the Turku Energia Imbalance Price Challenge.
---

# Turku Energia Imbalance Price Forecasting Skill

This skill enforces the domain workflow, data leakage prevention rules, and validation steps for the **Turku Energia Imbalance Price Forecasting Challenge (Since AI Hackathon 2026)**.

## 1. Strict Data Availability & Anti-Leakage Check (`RULE #1`)

Before engineering any feature or training/evaluating any model for forecast origin $t$:
* **Mandatory Filter**: Only use rows where `first_usable_origin_utc <= origin`.
* **Helper Function**: Always filter datasets through `rows_available_at(origin)` (see `baseline.ipynb`).
* **Never** slice or join purely on `timestamp_utc` or `delivery_start_utc` without first filtering by `first_usable_origin_utc <= origin`.

```python
def rows_available_at(df, origin_utc):
    """Return only rows known at or before origin_utc."""
    return df.loc[df["first_usable_origin_utc"] <= origin_utc].copy()
```

## 2. Project Directory Conventions

Place modular code in the designated directories:
* `src/data/`: Data loaders and `rows_available_at` filtering utilities.
* `src/features/`: Feature engineering (Fingrid wind/solar/consumption forecasts, spot price spreads, calendar/time features, cross-border flows, lag features respecting publication lag).
* `src/models/`:
  * Point & Quantile regression (`p10`, `p50`, `p90`) using LightGBM (`objective='quantile'` / `mae`, `RANDOM_SEED = 42`).
  * Multi-class classification for balancing direction (`p_up`, `p_down`, `p_none`, $\sum p = 1.0$).
  * Binary probabilistic classification for price spikes (`p_spike_high` for $> \text{spot} + 50$, `p_spike_low` for $< \text{spot} - 50$, $\sum p_{\text{spike}} \le 1.0$).
  * SHAP explainability (`shap.TreeExplainer`) to populate `top_drivers`.
* `src/evaluation/`: Local validation wrappers calling `score.py` and `validate_submission.py`.
* `notebooks/`: Exploratory data analysis and prototyping notebooks.
* `apps/`: Level 2 situational awareness dashboard & Time Machine UI.

## 3. Post-Processing & Constraint Enforcement

Always enforce mathematical constraints on predictions before saving:
1. **Quantile Monotonicity**:
   * Ensure `p10 <= p50 <= p90` (e.g., `p10 = np.minimum(p10, p50)` and `p90 = np.maximum(p90, p50)`).
2. **Direction Probabilities Simplex**:
   * Clip `p_up`, `p_down`, `p_none` to `[0.0, 1.0]` and normalize so `p_up + p_down + p_none == 1.0`.
3. **Spike Probabilities Bound**:
   * Clip `p_spike_high`, `p_spike_low` to `[0.0, 1.0]` and ensure `p_spike_high + p_spike_low <= 1.0`.

## 4. Mandatory Validation & Scoring Protocol

Every time `forecast_backtest.csv` or a validation prediction file is generated:

1. **Validate Submission Format (26,920 rows = 673 origins × 40 steps)**:
   ```bash
   python validate_submission.py --backtest forecast_backtest.csv --origins data/forecast_origins.csv
   ```
2. **Evaluate Local Validation Score**:
   ```bash
   python score.py --predictions <validation_predictions.csv> --actuals data/actuals_validation.csv
   ```
3. **Benchmark Comparison**:
   Compare metrics against the official baseline (`MAE: 66.20 €/MWh`, `Pinball: 24.18`).
