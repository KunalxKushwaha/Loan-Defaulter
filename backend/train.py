"""
train.py  –  Trains the XGBoost loan-defaulter pipeline and saves model.pkl.

Run from the project root:
    python backend/train.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
MODEL_PATH  = os.path.join(MODEL_DIR, "model.pkl")

# ── Feature lists ─────────────────────────────────────────────────────────────
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

TARGET = "loan_status"


def find_dataset():
    """Look for loan_data.csv in the dataset folder."""
    candidates = [
        os.path.join(DATASET_DIR, "loan_data.csv"),
        os.path.join(DATASET_DIR, "loan_data.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # fallback: search dataset dir for any csv
    for f in os.listdir(DATASET_DIR):
        if f.endswith(".csv"):
            return os.path.join(DATASET_DIR, f)
    return None


def train():
    print("=" * 60)
    print("  Loan Defaulter – Model Training")
    print("=" * 60)

    # 1. Load data
    csv_path = find_dataset()
    if csv_path is None:
        sys.exit(
            "[ERROR] No CSV found in dataset/. "
            "Place loan_data.csv there and retry."
        )
    print(f"\n[+] Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"    Shape: {df.shape}")

    # 2. Prepare features / target
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[+] Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

    # 3. Build preprocessing + XGBoost pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ],
        remainder="passthrough",
    )

    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    xgb_clf = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("xgb_model",    xgb_clf),
    ])

    # 4. Hyper-parameter search
    params = {
        "xgb_model__n_estimators":    [100, 200, 300],
        "xgb_model__max_depth":       [3, 5, 7],
        "xgb_model__learning_rate":   [0.05, 0.1, 0.2],
        "xgb_model__subsample":       [0.8, 1.0],
        "xgb_model__colsample_bytree":[0.8, 1.0],
        "xgb_model__gamma":           [0, 0.1],
    }

    print("\n[+] Running RandomizedSearchCV (this may take a few minutes)…")
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=params,
        n_iter=10,
        cv=3,
        scoring="f1",
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    print(f"\n[+] Best params : {search.best_params_}")
    print(f"    Best F1      : {search.best_score_:.4f}")

    # 5. Evaluate on test set
    y_pred = best_model.predict(X_test)
    print(f"\n[+] Test Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 6. Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"\n[✓] Model saved → {MODEL_PATH}")


if __name__ == "__main__":
    train()
