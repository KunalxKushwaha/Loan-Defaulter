"""
utils.py – Shared helper functions for the backend.
"""

import os
import pandas as pd


def read_dataset(path: str) -> pd.DataFrame:
    """Read a CSV and return a DataFrame, raising clear errors."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def validate_input(form_data: dict, valid_options: dict) -> list[str]:
    """
    Check that all required keys are present and categorical values
    are within the allowed set.  Returns a list of error strings.
    """
    errors = []
    for col, opts in valid_options.items():
        val = form_data.get(col)
        if val is None:
            errors.append(f"Missing field: {col}")
        elif val not in opts:
            errors.append(f"Invalid value '{val}' for {col}. Allowed: {opts}")
    return errors


def format_currency(amount: float) -> str:
    return f"${amount:,.2f}"


def format_percent(value: float) -> str:
    return f"{value:.2f}%"
