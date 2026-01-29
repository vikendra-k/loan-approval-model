from src.data_preprocessing import load_and_preprocess_data, split_data
from src.model_training import scale_features, train_logistic_regression
from src.evaluate import evaluate_model
import joblib
import os

DATA_PATH = "data/loan_approval_data.csv"

# 1. Load + preprocess
X, y = load_and_preprocess_data(DATA_PATH)

# 2. Split
X_train, X_test, y_train, y_test = split_data(X, y)

# 3. Scale
X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

# 4. Train
model = train_logistic_regression(X_train_scaled, y_train)

# 5. Evaluate
accuracy, cm, report = evaluate_model(model, X_test_scaled, y_test)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", report)

# 6. Save artifacts
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/loan_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model and scaler saved successfully")

