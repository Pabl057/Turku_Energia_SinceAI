"""Rule #2 Data Privacy & Nord Pool Licensing tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

from validate_submission import ALL_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_gitignore_blocks_data_csv_and_parquet() -> None:
    """Verify .gitignore contains required data ignore rules and git check-ignore blocks data files."""
    gitignore_path = PROJECT_ROOT / ".gitignore"
    assert gitignore_path.exists(), ".gitignore is missing"
    lines = [line.strip() for line in gitignore_path.read_text(encoding="utf-8").splitlines()]

    for required_pattern in ("/data/", "/data/*.csv", "/data/*.parquet"):
        assert required_pattern in lines, f"Missing required .gitignore pattern: {required_pattern}"

    protected_paths = [
        "data/spot_nordpool.csv",
        "data/inputs.parquet",
        "data/inputs.csv",
        "data/actuals_validation.csv",
    ]
    for rel_path in protected_paths:
        res = subprocess.run(
            ["git", "check-ignore", "-q", rel_path],
            cwd=PROJECT_ROOT,
            check=False,
        )
        assert res.returncode == 0, f"{rel_path} is NOT ignored by git!"


def test_git_index_does_not_track_licensed_or_raw_data_files() -> None:
    """Verify git ls-files and git status report zero tracked or staged files under data/ or spot_nordpool.csv."""
    ls_data = subprocess.run(
        ["git", "ls-files", "data/"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert ls_data.stdout.strip() == "", f"Tracked files found under data/: {ls_data.stdout}"

    status_data = subprocess.run(
        ["git", "status", "--porcelain", "data/"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert status_data.stdout.strip() == "", f"Staged/unignored files under data/: {status_data.stdout}"

    ls_all = subprocess.run(
        ["git", "ls-files"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    tracked_files = ls_all.stdout.splitlines()
    assert not any("spot_nordpool" in f for f in tracked_files), (
        "Licensed Nord Pool file tracked in git index!"
    )


def test_submission_and_source_files_do_not_expose_raw_nordpool_series() -> None:
    """Verify forecast_backtest.csv contains only the 13 official columns and no raw spot column."""
    sub_path = PROJECT_ROOT / "forecast_backtest.csv"
    if sub_path.exists():
        cols = pd.read_csv(sub_path, nrows=0).columns.tolist()
        assert cols == ALL_COLUMNS
        assert "spot_price" not in cols
        assert "fi_spot_price" not in cols
