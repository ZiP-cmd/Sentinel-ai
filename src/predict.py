
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "sentinel_fraud_model.joblib"
)

DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"

# Load our saved model.
artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
features = artifact["features"]

# Load historical transaction data.
df = pd.read_csv(DATA_PATH)
df = df.sort_values(
    "Time", kind="stable"
).reset_index(drop=True)

# Select the first transaction from our validation set.
validation_start = int(len(df) * 0.60)

transaction = df.iloc[
    [validation_start]
][features]

# Predict fraud risk.
fraud_probability = model.predict_proba(
    transaction
)[0, 1]

print("\n=== SENTINEL AI FRAUD PREDICTION ===")
print(f"Transaction amount: ${transaction['Amount'].iloc[0]:.2f}")
print(f"Model score: {fraud_probability:.4f}")

# For demonstration, use a 0.50 threshold.
threshold = 0.50

if fraud_probability >= threshold:
    print("Decision: FLAG FOR REVIEW")
else:
    print("Decision: NOT FLAGGED")

print("\nPrediction complete.")
