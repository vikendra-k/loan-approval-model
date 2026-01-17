import joblib
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "loan_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

THRESHOLD = 0.7

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def predict_loan(X_new):
    X_scaled = scaler.transform(X_new)
    prob = model.predict_proba(X_scaled)[:, 1]
    decision = (prob >= THRESHOLD).astype(int)
    return decision, prob


if __name__ == "__main__":
    # Example applicant (FAKE DATA)
    # Must match feature order used during training
    X_sample = np.array([[
        1001,      # Applicant_ID
        5000,      # Applicant_Income
        2000,      # Coapplicant_Income
        0,         # Dependents
        120,       # Loan_Amount
        360,       # Loan_Term
        1,         # Credit_History
        1,         # Married
        0,         # Self_Employed
        1,         # Education
        1,         # Gender
        0,         # Employer_Category
        1,         # Property_Area
        0, 0, 0, 0, 0, 0   # remaining dummy/encoded fields (adjust if needed)
    ]])

    decision, prob = predict_loan(X_sample)

    print("Approval Probability:", prob[0])
    print("Loan Approved:", "YES" if decision[0] == 1 else "NO")

