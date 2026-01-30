import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from src.config import FEATURES, TARGET

# ================= PATH SETUP =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "loan_approval_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "loan_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "..", "models", "scaler.pkl")

print("Loading data from:", DATA_PATH)

# ================= LOAD DATA =================
df = pd.read_csv(DATA_PATH)

print("\nColumns in dataset:")
print(df.columns.tolist())

# ================= CLEAN TARGET =================
df[TARGET] = (
    df[TARGET]
    .astype(str)
    .str.strip()
    .str.lower()
)

# YES -> 1, everything else -> 0
df[TARGET] = (df[TARGET] == "yes").astype(int)

# ================= FEATURE SELECTION =================
X = df[FEATURES]
y = df[TARGET]

# ================= HANDLE MISSING VALUES (KEY FIX) =================
X = X.fillna(X.median())

# ================= TRAIN TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ================= SCALING =================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================= MODEL =================
base_model = LogisticRegression(
    max_iter=1000,
    solver="lbfgs",
    class_weight="balanced"
)

model = CalibratedClassifierCV(
    base_model,
    method="sigmoid",
    cv=5
)

model.fit(X_train_scaled, y_train)

# ================= SANITY CHECK =================
probs = model.predict_proba(X_train_scaled)[:, 1]
print("\nProbability sanity check:")
print("Min probability:", probs.min())
print("Max probability:", probs.max())

# ================= SAVE ARTIFACTS =================
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("\n✅ Model and scaler saved successfully")
