"""
Generates a synthetic loan-applicant dataset with the same columns used by
the classic "Loan Prediction" problem (Gender, Married, Dependents, Education,
Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount,
Loan_Amount_Term, Credit_History, Property_Area, Loan_Status).

Replace this file's output (data/loan_data.csv) with your real dataset any
time -- just keep the same column names and train.py will keep working.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 1200

gender = np.random.choice(["Male", "Female"], N, p=[0.8, 0.2])
married = np.random.choice(["Yes", "No"], N, p=[0.65, 0.35])
dependents = np.random.choice(["0", "1", "2", "3+"], N, p=[0.55, 0.18, 0.17, 0.10])
education = np.random.choice(["Graduate", "Not Graduate"], N, p=[0.78, 0.22])
self_employed = np.random.choice(["Yes", "No"], N, p=[0.14, 0.86])
property_area = np.random.choice(["Urban", "Semiurban", "Rural"], N, p=[0.38, 0.38, 0.24])

applicant_income = np.random.gamma(shape=3.0, scale=1800, size=N).astype(int) + 1500
coapplicant_income = np.where(
    married == "Yes",
    np.random.gamma(shape=2.0, scale=900, size=N).astype(int),
    0,
)
loan_amount = (
    (applicant_income + coapplicant_income) * np.random.uniform(0.05, 0.18, N)
).astype(int)
loan_amount = np.clip(loan_amount, 10, 700)

loan_term = np.random.choice([360, 240, 180, 120, 60], N, p=[0.75, 0.10, 0.08, 0.05, 0.02])
credit_history = np.random.choice([1.0, 0.0], N, p=[0.84, 0.16])

# Ground-truth approval driven by a logistic-ish rule + noise, so the models
# have real signal to learn (mirrors how credit history / income dominate in
# the real dataset).
total_income = applicant_income + coapplicant_income
income_to_loan = total_income / (loan_amount * 1000 + 1)

score = (
    2.5 * credit_history
    + 0.8 * (education == "Graduate").astype(int)
    + 0.6 * np.tanh(income_to_loan / 2)
    + 0.3 * (property_area == "Semiurban").astype(int)
    - 0.4 * (self_employed == "Yes").astype(int)
    + np.random.normal(0, 0.6, N)
)
prob_approved = 1 / (1 + np.exp(-(score - 1.2)))
loan_status = np.where(np.random.rand(N) < prob_approved, "Y", "N")

df = pd.DataFrame(
    {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
        "Loan_Status": loan_status,
    }
)

# sprinkle a few missing values, like the real-world dataset has
for col in ["Gender", "Married", "Dependents", "Self_Employed", "LoanAmount", "Loan_Amount_Term", "Credit_History"]:
    mask = np.random.rand(N) < 0.03
    df.loc[mask, col] = np.nan

df.to_csv("data/loan_data.csv", index=False)
print(f"Wrote data/loan_data.csv with {len(df)} rows")
print(df["Loan_Status"].value_counts(normalize=True))
