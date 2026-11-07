#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root regardless of working directory
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -x "${REPO_ROOT}/venv/bin/python" ]]; then
    PYTHON="${REPO_ROOT}/venv/bin/python"
else
    PYTHON="$(command -v python3)"
fi

if ! "${PYTHON}" -c "import pytest" >/dev/null 2>&1; then
    echo "ERROR: pytest is not installed in ${PYTHON}. Run: ${PYTHON} -m pip install 'pytest>=8.0'" >&2
    exit 1
fi

echo "======================================================================"
echo "1. Running Automated Pytest Suite (Rules #1, #2, #3, Metrics, E2E)"
echo "======================================================================"
"${PYTHON}" -m pytest \
    tests/test_submission_schema.py \
    tests/test_data_leakage.py \
    tests/test_licensing_privacy.py \
    tests/test_metrics.py \
    tests/test_pipeline_e2e.py \
    -v "$@"

echo ""
echo "======================================================================"
echo "2. Running Official Submission Validator (validate_submission.py)"
echo "======================================================================"
"${PYTHON}" validate_submission.py \
    --backtest forecast_backtest.csv \
    --origins data/forecast_origins.csv

echo ""
echo "======================================================================"
echo "ALL CHECKS PASSED SUCCESSFULLY"
echo "======================================================================"
