# Loan Approval Prediction System

🚀 **Live App:**  
👉 https://loan-approval-model-vihvjcncds6x4bi2oljbnn.streamlit.app/

ℹ️ *Note: The app may take a few seconds to load if it was inactive.*

An end-to-end machine learning project that predicts whether a loan
application should be approved based on applicant financial, personal,
and credit information.

## Problem Statement
Banks face financial losses when risky applicants are approved and
business losses when reliable applicants are rejected.  
This project builds a cost-sensitive ML model to assist loan approval
decisions.

## Dataset
The dataset contains applicant-level information such as:
- Income details
- Credit history
- Loan amount and term
- Employment and demographic attributes

Target variable:
- `Loan_Approved` (Yes / No)

## ML Pipeline
1. Data preprocessing (missing values, encoding)
2. Train–test split with stratification
3. Feature scaling using StandardScaler
4. Cost-sensitive Logistic Regression
5. Model evaluation (accuracy, confusion matrix, classification report)
6. Probability threshold tuning
7. Model and scaler persistence

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit

## How to Run

### Train the model
```bash
python main.py
