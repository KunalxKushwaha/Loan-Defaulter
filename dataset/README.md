# Dataset

Place `loan_data.csv` in this folder.

Expected columns:
- `person_age` (int)
- `person_gender` (str: male/female)
- `person_education` (str: High School/Associate/Bachelor/Master/Doctorate)
- `person_home_ownership` (str: RENT/OWN/MORTGAGE/OTHER)
- `person_income` (float)
- `person_emp_exp` (int, years)
- `loan_intent` (str: PERSONAL/EDUCATION/MEDICAL/VENTURE/HOMEIMPROVEMENT/DEBTCONSOLIDATION)
- `loan_amnt` (float)
- `loan_int_rate` (float, %)
- `loan_percent_income` (float)
- `cb_person_cred_hist_length` (int, years)
- `credit_score` (int)
- `previous_loan_defaults_on_file` (str: Yes/No)
- `loan_status` (int: 0 = No Default, 1 = Default)  ← target column
