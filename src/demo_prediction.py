
from pathlib import Path

import pandas as pd

from prediction_service import (
    load_model,
    predict_transaction,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "creditcard.csv"

# Load the trained model once.
artifact = load_model()

# Load historical transactions.
df = pd.read_csv(DATA_PATH)
df = df.sort_values(
    "Time", kind="stable"
).reset_index(drop=True)

# Choose a transaction from the validation period.
validation_start = int(len(df) * 0.60)

transaction = df.iloc[validation_start].drop("Class").to_dict()

result = predict_transaction(
    transaction,
    artifact,
    threshold=0.50,
)

print("\n=== SENTINEL AI PREDICTION SERVICE ===")
print(f"Transaction amount: ${transaction['Amount']:.2f}")
print(f"Fraud score: {result['fraud_score']:.4f}")
print(f"Threshold: {result['threshold']:.2f}")
print(f"Flagged: {result['flagged']}")

print("\nPrediction service demonstration complete.")
