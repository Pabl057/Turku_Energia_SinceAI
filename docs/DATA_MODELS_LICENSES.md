# List of Data Sources, Models, Software Components & Usage Rights

**Project:** Turku Energia Imbalance Price Forecasting Challenge (Since AI Hackathon 2026)

---

## 1. Data Sources & Usage Rights

All model training, validation, and test backtesting use **exclusively** the official starter package datasets (`../data/inputs.parquet` and `data/spot_nordpool.csv`), strictly filtered by `first_usable_origin_utc <= forecast_origin_utc`. No external live APIs or third-party proprietary datasets are queried during inference.

| Data Source / Provider | Series Group (`availability.csv`) | Dataset / Document IDs | Availability & Publication Lag Rule (`first_usable_origin_utc`) | License & Usage Rights |
| :--- | :--- | :--- | :--- | :--- |
| **Fingrid Open Data** | `imbalance_price` | `319` | **60 min** after 15-min period end | **CC BY 4.0** (Creative Commons Attribution 4.0 International). Free to share and adapt with attribution to Fingrid Oyj. |
| **Fingrid Open Data** | `mfrr_price_volume` | `244`, `106`, `375`, `376` | **60 min** after 15-min period end | **CC BY 4.0** (Fingrid Oyj). |
| **Fingrid Open Data** | `afrr_price_volume` | `347`, `348` | **60 min** after 15-min period end | **CC BY 4.0** (Fingrid Oyj). |
| **Fingrid Open Data** | `mfrr_direction` | `369` | **60 min** after 15-min period end | **CC BY 4.0** (Fingrid Oyj). |
| **Fingrid Open Data** | `consumption_production_actual` | `124` (consumption), `74` (production) | **30 min** after 15-min period end | **CC BY 4.0** (Fingrid Oyj). |
| **Fingrid Open Data** | `cross_border_flow` | `61` (FI–SE1), `57` (FI–SE3), `60` (FI–EE), `55` (FI–NO) | **30 min** after 15-min period end | **CC BY 4.0** (Fingrid Oyj). |
| **Fingrid Open Data** | `consumption_forecast_da`, `wind_forecast_da`, `solar_forecast_da` | `165`, `246`, `247` | **D-1 at 13:00** `Europe/Helsinki` | **CC BY 4.0** (Fingrid Oyj). |
| **ENTSO-E Transparency Platform** | `neighbour_imbalance_load` | `17.1.g` (Imbalance prices SE1, SE3, EE), `6.1.b` (Load forecast SE1, SE3, EE) | **120 min** after period end | **CC BY 4.0** (ENTSO-E list of data available for free re-use; converted to UTC and 15-min intervals in starter package). |
| **ENTSO-E Transparency Platform** | `transfer_capacity` | `11.1` (Day-ahead scheduled commercial exchanges / capacity FI–EE & EE–FI) | **D-1 at 14:00** `Europe/Helsinki` | **CC BY 4.0** (ENTSO-E free re-use list). |
| **Nord Pool (via Turku Energia)** | `spot_price` (`../data/spot_nordpool.csv`) | Day-Ahead Spot Price (`fi_spot_price`, Finland) | **D-1 at 14:00** `Europe/Helsinki` | **Proprietary — Licensed to Turku Energia.** Shared exclusively for use during the Since AI 2026 Hackathon. **Restrictions:** No redistribution, no upload to public repositories (`../.gitignore` enforced and verified via `tests/test_licensing_privacy.py`), no raw spot price column exported in `forecast_backtest.csv`, and mandatory deletion after the event. |
| **Starter Package Calendar Series** | `calendar` | Derived calendar & holiday indicators (Finland) | **Known in advance** (`2026-05-31T21:00:00Z`) | **Public Domain / MIT** (Algorithmically generated calendar/time features and Finnish statutory holiday schedule). |

---

## 2. Machine Learning Models & Statistical Methods

No pre-trained foundation models, external LLM APIs, or commercial model weights are used. All models are **trained from scratch** (`RANDOM_SEED = 42`) on the starter package time series in `../src/models/train.py` (`ImbalanceModelSuite`).

| Model Identifier | Target Output | Algorithm / Architecture | Training & Calibration Methodology | Rights / Origin |
| :--- | :--- | :--- | :--- | :--- |
| **`m_p50` + `m_p50_short`** | `p50` (Median Imbalance Price, €/MWh) | LightGBM Gradient Boosted Decision Trees (`LGBMRegressor`, `objective='regression_l1'`) | Trained on the **imbalance-to-spot spread** ($\Delta P = P_{\text{imbalance}} - P_{\text{spot}}$) across 181 leak-free features. Full-horizon model (`steps 1..40`) blended with a short-horizon specialist (`steps 1..8`) and added back to `spot_at_delivery`. | Custom-trained from scratch; MIT-licensed framework (LightGBM). |
| **`m_p10` & `m_p90`** | `p10`, `p90` (80% Uncertainty Interval, €/MWh) | LightGBM Quantile Regressors (`objective='quantile'`, $\alpha \in \{0.10, 0.90\}$) + Split Conformal Dispersion Scaling | Predicts raw 10th/90th spread quantiles, calibrated via **Split Conformal Dispersion Scaling** around `p50` (`conformal_scale_lo=1.45`, `conformal_scale_hi=1.30`, shifts `-0.5` / `+7.0`) and monotonic clipping ($p_{10} \le p_{50} \le p_{90}$) to achieve exact 80.0% validation coverage. | Custom-trained from scratch; MIT-licensed framework (LightGBM). |
| **`m_dir` + `m_dir_short`** | `p_up`, `p_down`, `p_none` (Dominant mFRR Direction) | LightGBM Multiclass Classifier (`LGBMClassifier`, `objective='multiclass'`, `num_class=3`) + Empirical Bayes Shrinkage | Full-horizon + short-horizon (`steps 1..8`) ensemble with horizon-dependent **Empirical Bayes Brier shrinkage** toward base rates and simplex normalization ($\sum p = 1.0$). | Custom-trained from scratch; MIT-licensed framework (LightGBM). |
| **`m_spike_high` & `m_spike_low`** | `p_spike_high` ($>\text{spot}+50$), `p_spike_low` ($<\text{spot}-50$) | LightGBM Binary Classifiers (`objective='binary'`) + Parametric Quantile-Tail Blend | Blends calibrated GBDT binary probabilities ($w=0.45$) with a Student-t / Gaussian tail probability derived from the $(p_{10}, p_{50}, p_{90})$ spread distribution ($w=0.55$), enforcing $p_{\text{spike\_high}} + p_{\text{spike\_low}} \le 1.0$. | Custom-trained from scratch; MIT/BSD-licensed frameworks (LightGBM, SciPy). |
| **TreeSHAP Explainer** | `top_drivers` (Feature Attribution) | Exact TreeSHAP (`pred_contrib=True` / `shap.TreeExplainer`) | Computes exact Lundberg & Lee TreeSHAP values per forecast row on `m_p50`, aggregating the top 3 distinct human-readable driver groups per step. | Open-source algorithm (MIT License). |

---

## 3. Software Components, Libraries & Licenses

All software components are open-source and permissive (OSI-approved licenses), allowing unrestricted commercial use, modification, and deployment by Turku Energia.

| Component / Package | Version Used | Role in Solution | Software License |
| :--- | :--- | :--- | :--- |
| **Python** | `3.12.16` | Core runtime environment | PSF License Agreement (BSD-compatible) |
| **LightGBM** (`lightgbm`) | `4.7.0` | GBDT quantile regression, multiclass & binary classification, native C++ TreeSHAP | MIT License |
| **SHAP** (`shap`) | `0.54.1` | `TreeExplainer` model explainability & feature attribution | MIT License |
| **pandas** (`pandas`) | `3.1.1` | Time-series alignment, `rows_available_at` anti-leakage filtering, rolling/lag aggregation | BSD 3-Clause License |
| **NumPy** (`numpy`) | `2.5.4` | Vectorized array operations, post-processing constraint enforcement | BSD 3-Clause License |
| **SciPy** (`scipy`) | `1.17.1` | Normal CDF tail probability estimation (`scipy.stats.norm`) for spike calibration | BSD 3-Clause License |
| **scikit-learn** (`scikit-learn`) | `1.9.1` | Evaluation metrics and cross-validation utilities | BSD 3-Clause License |
| **PyArrow** (`pyarrow`) | `26.0.1` | Fast Parquet I/O (`../data/inputs.parquet`) | Apache License 2.0 |
| **Streamlit** (`streamlit`) | `1.67.0` | Level 2 Situational Awareness Dashboard, Alerting & Historical "Time Machine" UI (`../apps/dashboard.py`) | Apache License 2.0 |
| **Plotly** (`plotly`) | `7.1.0` | Interactive fan charts (P10–P90), spike probability heatmaps, and SHAP visualizations | MIT License |
| **Requests** (`requests`) | `2.34.2` | Starter package dataset builder dependency | Apache License 2.0 |
| **tzdata** (`tzdata`) | `2025.2` | IANA timezone definitions (`Europe/Helsinki` D-1 publication cutoffs & DST transitions) | Apache License 2.0 |
| **Pytest** (`pytest`) | `9.2.0` | Automated verification suite (anti-leakage, schema, metric improvement, and Nord Pool privacy tests) | MIT License |
| **Jupyter** (`jupyter`) | `1.1.1` | Execution of official starter baseline (`../notebooks/baseline.ipynb`) | BSD 3-Clause License |
| **Starter Package Scripts** | Official (2026) | `../build_dataset.py`, `make_templates.py`, `validate_submission.py`, `score.py` | Provided by Turku Energia / Since AI 2026 Organizer |

---

## 4. External Services & Third-Party APIs

* **External Runtime / Cloud Services Used:** **None.**
* The entire Level 1 forecasting pipeline (`../src`) and Level 2 situational dashboard (`apps/dashboard.py`) run **100% locally and offline** on standard CPU hardware using the starter package files.

---

## 5. Compliance, IP & Data Governance Summary

1. **Nord Pool Data Protection (`RULE #2`):**
   * `../data/spot_nordpool.csv` and all `/data/`, `/baseline_output/`, and `/results/` directories are explicitly blocked by `.gitignore` and verified untracked via automated test (`tests/test_licensing_privacy.py`).
   * The submission file `../forecast_backtest.csv` contains only the 13 official schema columns and exposes no raw Nord Pool spot price series.
   * All licensed Nord Pool data files will be deleted from local storage immediately after the hackathon concludes.
2. **Strict Anti-Leakage Compliance (`RULE #1`):**
   * Every feature lookback and target alignment enforces `first_usable_origin_utc <= forecast_origin_utc`, verified by `../tests/test_data_leakage.py` and recorded per origin in the `data_cutoff_utc` column.
3. **Intellectual Property of Custom Code:**
   * All custom source code developed by the team (`../src`, `apps/`, `tests/`, `scripts/`) uses only permissive open-source dependencies (MIT, BSD-3-Clause, Apache 2.0, PSF) and carries no copyleft (GPL/AGPL) or proprietary third-party restrictions.
