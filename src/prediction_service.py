
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = (
    PROJECT_ROOT / "models" / "sentinel_fraud_model.joblib"
)


def load_model():
    """Load the saved Sentinel AI model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Sentinel model not found. "
            "Run python src/save_model.py first."
        )

    return joblib.load(MODEL_PATH)


def predict_transaction(transaction, artifact, threshold=0.50):
    """Score one transaction using the saved model."""

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    model = artifact["model"]
    features = artifact["features"]

    # Ensure transaction features are correctly ordered.
    data = pd.DataFrame([transaction])

    missing = [f for f in features if f not in data.columns]

    if missing:
        raise ValueError(
            f"Missing transaction features: {missing}"
        )

    data = data[features]

    score = float(model.predict_proba(data)[0, 1])

    return {
        "fraud_score": round(score, 4),
        "threshold": threshold,
        "flagged": score >= threshold,
    }
