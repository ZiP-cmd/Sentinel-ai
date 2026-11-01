
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "creditcard.csv"
MODEL_PATH = ROOT / "models" / "sentinel_fraud_model.joblib"

# Read and chronologically order the dataset.
df = pd.read_csv(DATA_PATH)

df = df.sort_values(
    "Time", kind="stable"
).reset_index(drop=True)

# Reproduce the established split boundaries.
train_end = int(len(df) * 0.60)
validation_end = int(len(df) * 0.80)

train = df.iloc[:train_end]
validation = df.iloc[train_end:validation_end]
test = df.iloc[validation_end:]

print("\n=== SENTINEL AI DATA SPLIT VERIFICATION ===")

print(f"Total transactions: {len(df):,}")
print(f"Training transactions: {len(train):,}")
print(f"Validation transactions: {len(validation):,}")
print(f"Reserved test transactions: {len(test):,}")

# Confirm the expected partition sizes.
assert len(df) == 284807
assert len(train) == 170884
assert len(validation) == 56961
assert len(test) == 56962

# Confirm each partition is chronologically ordered.
assert train["Time"].is_monotonic_increasing
assert validation["Time"].is_monotonic_increasing
assert test["Time"].is_monotonic_increasing

# Confirm chronological boundaries.
assert train["Time"].max() <= validation["Time"].min()
assert validation["Time"].max() <= test["Time"].min()

# Confirm the saved model's feature schema matches.
artifact = joblib.load(MODEL_PATH)

expected_features = [
    column for column in df.columns
    if column != "Class"
]

assert artifact["features"] == expected_features

# Verify there are no missing input values.
assert not df[expected_features].isna().any().any()

print("\nChronological boundaries: VERIFIED")
print("Model input features: VERIFIED")
print("Missing input values: NONE")

print("\nData split verification PASSED.")
print("Test-set predictions have NOT been generated.")
