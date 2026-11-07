# INSTRUCTIONS.md — Command Reference for Training, Starting, and Testing

This guide lists all commands required to set up the environment, train the Level 1 forecasting models, generate and score submissions, launch the Level 2 Situational Awareness Dashboard, and run the automated test suite.

All commands assume your current working directory is the repository root (`Turku_Energia_SinceAI/`).

---

## 1. Environment Setup

### Create & Activate Virtual Environment (Python 3.11+)
If `./venv` does not exist yet:
```bash
python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt
```

If `./venv` already exists, verify or update dependencies with:
```bash
./venv/bin/pip install -r requirements.txt
```

---

## 2. Training & Generating Submissions (Level 1 Pipeline)

### Run the Full End-to-End Training, Validation & Submission Pipeline
This command executes the leak-free `FeatureStore` pipeline, runs expanding 7-day walk-forward training of the 6-model LightGBM suite + TreeSHAP explainability, evaluates the validation period (`7 Sep – 4 Oct 2026`), generates the official test submission (`5 Oct – 1 Nov 2026`), and validates all outputs:
```bash
./venv/bin/python -m src.evaluation.pipeline
```

**Artifacts produced by the pipeline:**
- `forecast_backtest.csv` — Final official test submission (`26,920` rows = `673` origins $\times$ `40` steps).
- `results/validation/forecast_backtest.csv` — Validation period backtest (`26,880` rows = `672` origins $\times$ `40` steps).
- `results/validation/score/score_summary.csv` & `score_by_horizon.csv` — Validation metrics compared against the official baseline.
- `results/feature_importance.csv` — Global LightGBM feature importances for the Level 2 UI.

---

## 3. Starting the Level 2 Situational Awareness Dashboard

### Launch the Streamlit Application
Start the interactive Level 2 Situational Desk & Historical Time Machine UI:
```bash
./venv/bin/streamlit run apps/dashboard.py
```

Optional custom port / headless flags:
```bash
./venv/bin/streamlit run apps/dashboard.py --server.port 8501
```

Once started, open `http://localhost:8501` in your browser to access:
- **Tab 1 — Live Situational Desk & Actionable Recommendations**: Spike probability alerts ($P(\text{spike}) \ge 30\%$), P10/P50/P90 price fan chart against day-ahead spot, mFRR direction probabilities, and concrete trading/dispatch actions.
- **Tab 2 — Historical Time Machine & Forecast Revision Analysis**: Step through any historical validation or test forecast origin $t$, compare against actuals, and inspect revision deltas between consecutive origins ($t - 1\text{h} \to t$).
- **Tab 3 — SHAP Forecast Drivers & Grid Fundamentals**: Per-step TreeSHAP top drivers and strictly filtered (`first_usable_origin_utc <= t`) Fingrid / ENTSO-E series.
- **Tab 4 — Validation Performance vs. Official Baseline**: Horizon-by-horizon MAE, Pinball loss, Spike Brier, Direction Brier, and D14 points breakdown.

---

## 4. Testing & Validation

### Run All Checks (Recommended Single Command)
Execute the full automated `pytest` suite (Rules #1, #2, #3, metrics, and end-to-end pipeline) followed by the official submission validator:
```bash
./scripts/run_checks.sh
```

You can also pass any extra `pytest` flags directly to `./scripts/run_checks.sh`:
```bash
./scripts/run_checks.sh -k "leakage or schema"
```

### Run Individual Pytest Modules
```bash
# Run all tests
./venv/bin/python -m pytest -v

# Rule #1: Data availability & anti-leakage (first_usable_origin_utc <= origin)
./venv/bin/python -m pytest tests/test_data_leakage.py -v

# Rule #2: Data privacy & Nord Pool licensing (.gitignore & git index checks)
./venv/bin/python -m pytest tests/test_licensing_privacy.py -v

# Rule #3: Submission schema & mathematical constraints (p10 <= p50 <= p90, simplex, spike sum)
./venv/bin/python -m pytest tests/test_submission_schema.py -v

# Evaluation metrics & D14 points formula
./venv/bin/python -m pytest tests/test_metrics.py -v

# End-to-end mini walk-forward pipeline, SHAP drivers & baseline benchmark superiority
./venv/bin/python -m pytest tests/test_pipeline_e2e.py -v
```

### Run Official Submission Validator (`validate_submission.py`)
Validate the final test submission (`forecast_backtest.csv`):
```bash
./venv/bin/python validate_submission.py \
    --backtest forecast_backtest.csv \
    --origins data/forecast_origins.csv
```

Validate the validation-period backtest (`results/validation/forecast_backtest.csv`):
```bash
./venv/bin/python validate_submission.py \
    --backtest results/validation/forecast_backtest.csv \
    --origins baseline_output/validation/forecast_origins.csv
```

### Run Official Scoring Script (`score.py`)
**Mode 1 — Score validation predictions against `data/actuals_validation.csv`:**
```bash
./venv/bin/python score.py \
    --submission results/validation/forecast_backtest.csv \
    --actuals data/actuals_validation.csv \
    --spike-margin 50 \
    --output-dir results/validation/score \
    --baseline-summary baseline_output/validation/score/score_summary.csv
```

**Mode 2 — Compute D14 automatic points (`points_summary.csv`) against the baseline:**
```bash
./venv/bin/python score.py \
    --points results/points_comparison \
    --baseline results/points_comparison/baseline/score_summary.csv \
    --output results/points_comparison/points_summary.csv
```

---

## 5. Exploratory Notebook (`notebooks/baseline.ipynb`)

To inspect or run the official starter baseline notebook interactively:
```bash
./venv/bin/jupyter lab notebooks/baseline.ipynb
```
