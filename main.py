import pandas as pd

df = pd.read_csv("data/loan_approval_data.csv")

print("Dataset loaded successfully")
print(df.shape)
print(df.head())

from src.data_preprocessing import load_and_preprocess_data, split_data

DATA_PATH = "data/loan_approval_data.csv"

# Load + preprocess
X, y = load_and_preprocess_data(DATA_PATH)

# Train-test split
X_train, X_test, y_train, y_test = split_data(X, y)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

from src.data_preprocessing import load_and_preprocess_data, split_data
from src.model_training import scale_features

DATA_PATH = "data/loan_approval_data.csv"

X, y = load_and_preprocess_data(DATA_PATH)

X_train, X_test, y_train, y_test = split_data(X, y)

X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

print("Scaling done")
print("Mean (train):", X_train_scaled.mean())
print("Std (train):", X_train_scaled.std())

from src.data_preprocessing import load_and_preprocess_data, split_data
from src.model_training import scale_features, train_logistic_regression

DATA_PATH = "data/loan_approval_data.csv"

X, y = load_and_preprocess_data(DATA_PATH)

X_train, X_test, y_train, y_test = split_data(X, y)

X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

model = train_logistic_regression(X_train_scaled, y_train)

print("Model trained successfully")

from src.data_preprocessing import load_and_preprocess_data, split_data
from src.model_training import scale_features, train_logistic_regression
from src.evaluate import evaluate_model

DATA_PATH = "data/loan_approval_data.csv"

X, y = load_and_preprocess_data(DATA_PATH)

X_train, X_test, y_train, y_test = split_data(X, y)

X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

model = train_logistic_regression(X_train_scaled, y_train)

accuracy, cm, report = evaluate_model(model, X_test_scaled, y_test)

print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

from src.evaluate import evaluate_with_threshold

for t in [0.4, 0.5, 0.6, 0.7]:
    cm = evaluate_with_threshold(model, X_test_scaled, y_test, threshold=t)
    print(f"\nThreshold = {t}")
    print(cm)

import joblib
import os

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/loan_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model and scaler saved")
