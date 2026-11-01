
"""Generate a synthetic model for CI testing only."""

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "models" / "sentinel_fraud_model.joblib"

FEATURES = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)


def main():
    rng = np.random.default_rng(42)

    # Synthetic examples; no real transaction records.
    X = rng.normal(size=(40, len(FEATURES)))
    y = np.array([0, 1] * 20)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])

    model.fit(X, y)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "model_name": "Synthetic CI Test Model",
        },
        OUTPUT,
    )

    print("Synthetic CI model created successfully.")
    print(f"Features: {len(FEATURES)}")


if __name__ == "__main__":
    main()
