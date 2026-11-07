# PowerShell equivalent of scripts/run_checks.sh
$ErrorActionPreference = "Stop"

# Resolve repository root regardless of working directory
$RepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $RepoRoot

try {
    $VenvPythonWin = Join-Path $RepoRoot "venv\Scripts\python.exe"
    $VenvPythonUnix = Join-Path $RepoRoot "venv/bin/python"

    if (Test-Path $VenvPythonWin) {
        $Python = $VenvPythonWin
    } elseif (Test-Path $VenvPythonUnix) {
        $Python = $VenvPythonUnix
    } else {
        $PythonCmd = Get-Command python -ErrorAction SilentlyContinue
        if (-not $PythonCmd) {
            $PythonCmd = Get-Command python3 -ErrorAction SilentlyContinue
        }
        if (-not $PythonCmd) {
            Write-Error "ERROR: Python interpreter not found."
            exit 1
        }
        $Python = $PythonCmd.Source
    }

    & $Python -c "import pytest" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "ERROR: pytest is not installed in $Python. Run: $Python -m pip install 'pytest>=8.0'"
        exit 1
    }

    Write-Host "======================================================================"
    Write-Host "1. Running Automated Pytest Suite (Rules #1, #2, #3, Metrics, E2E)"
    Write-Host "======================================================================"
    & $Python -m pytest `
        tests/test_submission_schema.py `
        tests/test_data_leakage.py `
        tests/test_licensing_privacy.py `
        tests/test_metrics.py `
        tests/test_pipeline_e2e.py `
        -v @args
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    Write-Host ""
    Write-Host "======================================================================"
    Write-Host "2. Running Official Submission Validator (validate_submission.py)"
    Write-Host "======================================================================"
    & $Python validate_submission.py `
        --backtest forecast_backtest.csv `
        --origins data/forecast_origins.csv
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    Write-Host ""
    Write-Host "======================================================================"
    Write-Host "ALL CHECKS PASSED SUCCESSFULLY"
    Write-Host "======================================================================"
} finally {
    Pop-Location
}
