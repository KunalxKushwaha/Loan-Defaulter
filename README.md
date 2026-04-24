# 🏦 Loan Default Predictor

ML-powered loan default risk assessment — XGBoost + Streamlit full-stack project.

---

## 📁 Project Structure

```
loan_defaulter/
├── backend/
│   ├── __init__.py          # Package exports
│   ├── predictor.py         # Model loading & inference
│   ├── train.py             # Training script (XGBoost + hyperparameter tuning)
│   └── utils.py             # Shared helpers
│
├── dataset/
│   └── loan_data.csv        
│
├── frontend/
│   └── app.py               # Streamlit UI
│
├── models/
│   └── model.pkl            ← trained model saved here (auto-generated)
│
├── notebooks/
│   └── Loan_approval.ipynb  
│
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your dataset
Copy `loan_data.csv` into the `dataset/` folder.

### 3. Train the model
```bash
python backend/train.py
```
This will:
- Load and preprocess the dataset
- Run RandomizedSearchCV to tune XGBoost hyperparameters
- Save the best pipeline as `models/model.pkl`

### 4. Launch the app
```bash
streamlit run frontend/app.py
```

Open your browser at **http://localhost:8501**

---

## 🧠 Model Details

| Component           | Details                                      |
|---------------------|----------------------------------------------|
| Algorithm           | XGBoost Classifier                           |
| Preprocessing       | sklearn Pipeline (StandardScaler + OHE)      |
| Tuning              | RandomizedSearchCV, 10 iterations, 3-fold CV |
| Scoring metric      | F1-score (handles class imbalance)           |
| Class imbalance     | `scale_pos_weight` auto-calculated           |
| Output              | Probability + binary prediction              |

### Features used

**Numerical (8):** `person_age`, `person_income`, `person_emp_exp`, `loan_amnt`,
`loan_int_rate`, `loan_percent_income`, `cb_person_cred_hist_length`, `credit_score`

**Categorical (5):** `person_gender`, `person_education`, `person_home_ownership`,
`loan_intent`, `previous_loan_defaults_on_file`

---

## 🎛️ App Features

- **Interactive form** — all 13 features with validation
- **Adjustable decision threshold** — tune precision vs recall
- **Risk gauge chart** — visual probability meter
- **Application summary table** — all inputs at a glance
- **Risk factor analysis** — explains key drivers (credit score, DTI, prior defaults…)
- **Actionable recommendation** — approve / review / reject guidance

---

## 📝 Notes

- `loan_percent_income` is **auto-calculated** from loan amount ÷ income — no manual input needed.
- If the model file is missing, a warning banner appears and training instructions are shown.
- The app works with any sklearn-compatible pipeline saved with `joblib.dump`.

# Made with 💖 by Kunal Kushwaha