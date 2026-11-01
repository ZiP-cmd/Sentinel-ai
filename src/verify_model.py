
from pathlib import Path

import joblib

from model_config import MODEL_PARAMETERS

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT / "models" / "sentinel_fraud_model.joblib"
)

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
features = artifact["features"]

print("\n=== SENTINEL AI MODEL VERIFICATION ===")

print("\nSaved pipeline:")
print(model)

print("\nNumber of input features:")
print(len(features))

print("\nExpected model parameters:")
print(MODEL_PARAMETERS)

classifier = model.named_steps["classifier"]
scaler = model.named_steps["scaler"]

assert type(classifier).__name__ == "LogisticRegression"
assert type(scaler).__name__ == "StandardScaler"

for parameter, expected in MODEL_PARAMETERS.items():
    actual = classifier.get_params()[parameter]

    assert actual == expected, (
        f"Mismatch: {parameter}. "
        f"Expected {expected}, got {actual}"
    )

assert len(features) == 30, (
    f"Expected 30 input features, got {len(features)}"
)

print("\nModel architecture verification PASSED.")
