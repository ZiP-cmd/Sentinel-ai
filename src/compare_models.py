
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"

# Use the same chronological training and validation splits.
df = pd.read_csv(DATA_PATH)
df = df.sort_values("Time", kind="stable").reset_index(drop=True)

train_end = int(len(df) * 0.60)
validation_end = int(len(df) * 0.80)

train_df = df.iloc[:train_end]
validation_df = df.iloc[train_end:validation_end]

features = [col for col in df.columns if col != "Class"]

X_train = train_df[features]
y_train = train_df["Class"]

X_val = validation_df[features]
y_val = validation_df["Class"]

# Compare two logistic regression configurations.
model_configs = {
    "Standard": None,
    "Class Weighted": "balanced",
}

thresholds = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70]

for name, class_weight in model_configs.items():

    print(f"\n=== {name.upper()} MODEL ===")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight=class_weight,
            max_iter=1000,
            random_state=42,
        )),
    ])

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)[:, 1]

    ap = average_precision_score(y_val, probabilities)

    print(f"Average Precision: {ap:.4f}")

    print(
        f"{'Threshold':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'False Pos':<12}"
        f"{'Missed Fraud':<12}"
        f"{'Alerts':<10}"
    )

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_val, predictions, labels=[0, 1]
        ).ravel()

        precision = precision_score(
            y_val, predictions, zero_division=0
        )

        recall = recall_score(
            y_val, predictions, zero_division=0
        )

        print(
            f"{threshold:<12.2f}"
            f"{precision:<12.4f}"
            f"{recall:<12.4f}"
            f"{fp:<12}"
            f"{fn:<12}"
            f"{tp + fp:<10}"
        )

print("\nComparison complete.")
print("Only training and validation data were used.")
