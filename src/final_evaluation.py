
from pathlib import Path
import hashlib

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score

from alert_budget import evaluate_alert_budget
from model_config import EVALUATION_ALERT_BUDGET


ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "creditcard.csv"
MODEL_PATH = ROOT / "models" / "sentinel_fraud_model.joblib"

EXPECTED_SHA256 = (
    "4ABC7B3024B3301BF9525D086259905D69EC7383778BF08A85514AB7A86F9842"
)

print("\n=== SENTINEL AI FINAL HELD-OUT EVALUATION ===")

# 1. Verify that the saved model has not changed.
actual_hash = hashlib.sha256(
    MODEL_PATH.read_bytes()
).hexdigest().upper()

if actual_hash != EXPECTED_SHA256:
    raise RuntimeError(
        "Frozen model hash mismatch. "
        "Do not proceed until the artifact is verified."
    )

print("Frozen model SHA-256: VERIFIED")

# 2. Load the previously selected model.
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
features = artifact["features"]

# 3. Recreate the chronological data ordering.
df = pd.read_csv(DATA_PATH)

df = df.sort_values(
    "Time", kind="stable"
).reset_index(drop=True)

assert len(df) == 284807

train_end = int(len(df) * 0.60)
validation_end = int(len(df) * 0.80)

assert train_end == 170884
assert validation_end == 227845

# 4. Select ONLY the reserved test partition.
test = df.iloc[validation_end:].copy()

X_test = test[features]
y_test = test["Class"].to_numpy()

assert len(test) == 56962

# 5. Generate test scores once.
scores = model.predict_proba(X_test)[:, 1]

# 6. Calculate ranking quality.
average_precision = average_precision_score(
    y_test,
    scores,
)

# 7. Evaluate the preselected top-100 alert policy.
results = evaluate_alert_budget(
    y_true=y_test,
    scores=scores,
    budget=EVALUATION_ALERT_BUDGET,
)

print(f"\nTest transactions: {len(test):,}")
print(f"Actual fraud cases: {int(y_test.sum())}")

print("\n=== RANKING PERFORMANCE ===")
print(f"Average Precision: {average_precision:.4f}")

print("\n=== FROZEN INVESTIGATION POLICY ===")
print(f"Review budget: {results['budget']}")
print(f"Fraud caught (TP): {results['tp']}")
print(f"False alarms (FP): {results['fp']}")
print(f"Fraud missed (FN): {results['fn']}")
print(f"Correct legitimate (TN): {results['tn']}")
print(f"Precision: {results['precision']:.4f}")
print(f"Recall: {results['recall']:.4f}")

print("\nFinal held-out evaluation complete.")
print("Do not retune the model using these results.")
