
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "creditcard.csv"
FROZEN_MODEL_PATH = ROOT / "models" / "sentinel_fraud_model.joblib"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset missing: data/creditcard.csv"
        )

    df = pd.read_csv(DATA_PATH)
    df = df.sort_values(
        "Time", kind="stable"
    ).reset_index(drop=True)

    train_end = int(len(df) * 0.60)
    train = df.iloc[:train_end]

    features = [
        column for column in df.columns
        if column != "Class"
    ]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])

    model.fit(
        train[features],
        train["Class"],
    )

    # Verify the model can be serialized and reloaded
    # without modifying the existing frozen artifact.
    with TemporaryDirectory() as directory:
        path = Path(directory) / "retrained_model.joblib"

        joblib.dump(
            {"model": model, "features": features},
            path,
        )

        restored = joblib.load(path)

        assert restored["features"] == features

        sample = train[features].iloc[[0]]
        score = restored["model"].predict_proba(
            sample
        )[0, 1]

        assert 0 <= score <= 1

    print("Retraining verification passed.")
    print(f"Features: {len(features)}")
    print(f"Example model score: {score:.6f}")
    print(
        "Original model artifact left unchanged:",
        FROZEN_MODEL_PATH,
    )


if __name__ == "__main__":
    main()
