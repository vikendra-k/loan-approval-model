

import streamlit as st
import numpy as np
import joblib
import os

st.caption(
    "⚠️ This tool is for educational purposes only and does not represent a real banking decision system."
)

# Load trained model and scaler
MODEL_PATH = "models/loan_model.pkl"
SCALER_PATH = "models/scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

st.title("🏦 Loan Approval Prediction System")
st.write("Enter applicant details to predict loan approval probability.")

# ---- Input fields ----
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Term (months)", min_value=1)
credit_history = st.selectbox("Credit History", [0, 1])
dependents = st.number_input("Dependents", min_value=0)

if st.button("Predict Loan Approval"):
    # ⚠️ Feature order must match training data
    X_input = np.array([[
        0,  # Applicant_ID (dummy)
        applicant_income,
        coapplicant_income,
        dependents,
        loan_amount,
        loan_term,
        credit_history,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]])

    X_scaled = scaler.transform(X_input)
    prob = model.predict_proba(X_scaled)[0][1]

    st.subheader("Result")
    st.write(f"Approval Probability: **{prob:.2f}**")

    if prob >= 0.7:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
