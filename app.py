import streamlit as st
import pandas as pd
import joblib
import os

from src.config import FEATURES, THRESHOLD

# ------------------ Page config ------------------
st.set_page_config(
    page_title="Loan Approval Predictor",
    layout="centered"
)

st.title("🏦 Loan Approval Prediction System")
st.caption(
    "⚠️ This tool is for educational purposes only and does not represent a real banking decision system."
)

st.write("Enter applicant details to predict loan approval probability.")

# ------------------ Load model & scaler ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "loan_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_artifacts()

# ------------------ Input fields ------------------
applicant_income = st.number_input("Applicant Income", min_value=0.0, step=500.0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, step=500.0)
dependents = st.number_input("Dependents", min_value=0, max_value=5, step=1)
loan_amount = st.number_input("Loan Amount", min_value=0.0, step=10.0)
loan_term = st.number_input("Loan Term (months)", min_value=1, step=12)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, step=10)

# ------------------ Prediction ------------------
if st.button("Predict Loan Approval"):

    # 🔒 EXACT FEATURE NAMES (MATCH CSV + TRAINING)
    user_input = {
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Dependents": dependents,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Credit_Score": credit_score,
    }

    input_df = pd.DataFrame([user_input])[FEATURES]

    input_scaled = scaler.transform(input_df)
    prob = model.predict_proba(input_scaled)[0][1]

    st.subheader("Result")
    st.write(f"Approval Probability: **{prob:.2f}**")

    if prob >= THRESHOLD:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
