"""
Backend predictor module for Loan Defaulter Prediction.
Handles model loading, input preprocessing, and predictions.
"""

import os
import joblib
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

# ── Feature definitions (must match notebook) ────────────────────────────────
CATEGORICAL_COLS = [
    "person_gender",
    "person_education",
    "person_home_ownership",
    "loan_intent",
    "previous_loan_defaults_on_file",
]

NUMERICAL_COLS = [
    "person_age",
    "person_income",
    "person_emp_exp",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
]

ALL_FEATURES = NUMERICAL_COLS + CATEGORICAL_COLS

# Valid categorical options ────────────────────────────────────────────────────
VALID_OPTIONS = {
    "person_gender":                   ["male", "female"],
    "person_education":                ["High School", "Associate", "Bachelor", "Master", "Doctorate"],
    "person_home_ownership":           ["RENT", "OWN", "MORTGAGE", "OTHER"],
    "loan_intent":                     ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
    "previous_loan_defaults_on_file":  ["Yes", "No"],
}


def load_model():
    """Load the trained pipeline from disk. Returns the pipeline or None."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def build_input_dataframe(form_data: dict) -> pd.DataFrame:
    """
    Convert a flat dict of form values into a single-row DataFrame
    with columns in the same order the training pipeline expects.
    """
    row = {col: form_data[col] for col in ALL_FEATURES}
    return pd.DataFrame([row])


def predict(form_data: dict, threshold: float = 0.5):
    """
    Run inference.

    Parameters
    ----------
    form_data : dict  – raw values from the Streamlit form
    threshold  : float – decision threshold (default 0.5)

    Returns
    -------
    dict with keys:
        prediction  : int   (0 = No Default, 1 = Default)
        probability : float (probability of default)
        label       : str
        confidence  : float (probability of the predicted class)
        risk_level  : str   ('Low' | 'Medium' | 'High')
    """
    model = load_model()
    if model is None:
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Please train the model first (run the notebook and copy model.pkl to models/)."
        )

    df   = build_input_dataframe(form_data)
    prob = model.predict_proba(df)[0][1]           # P(default)
    pred = int(prob >= threshold)

    if prob < 0.30:
        risk = "Low"
    elif prob < 0.60:
        risk = "Medium"
    else:
        risk = "High"

    return {
        "prediction":  pred,
        "probability": round(float(prob), 4),
        "label":       "Loan Default" if pred == 1 else "No Default",
        "confidence":  round(float(prob if pred == 1 else 1 - prob), 4),
        "risk_level":  risk,
    }


def model_exists() -> bool:
    return os.path.exists(MODEL_PATH)


def get_valid_options() -> dict:
    return VALID_OPTIONS
