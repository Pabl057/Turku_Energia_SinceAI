"""Evaluation and validation pipeline wrapping score.py and validate_submission.py."""

from src.evaluation.pipeline import (
    run_full_pipeline,
    run_scorer,
    run_validator,
    write_submission,
)

__all__ = [
    "run_full_pipeline",
    "run_scorer",
    "run_validator",
    "write_submission",
]
