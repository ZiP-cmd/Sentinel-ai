
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

# Locate our dataset.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"

# Load and chronologically order transactions.
df = pd.read_csv(DATA_PATH)
df = df.sort_values("Time", kind="stable").reset_index(drop=True)

# Use the same 60/20/20 split as our earlier audit.
train_end = int(len(df) * 0.60)
validation_end = int(len(df) * 0.80)

train_df = df.iloc[:train_end].copy()
validation_df = df.iloc[train_end:validation_end].copy()

# Separate features from labels.
feature_columns = [col for col in df.columns if col != "Class"]

X_train = train_df[feature_columns]
y_train = train_df["Class"]

X_val = validation_df[feature_columns]
y_val = validation_df["Class"]

# A pipeline ensures scaling is learned from training data.
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42,
    )),
])

print("Training Sentinel AI logistic regression baseline...")

# Fit the model using training data only.
model.fit(X_train, y_train)

print("Training complete!")

# Predict probabilities and labels on validation data.
fraud_probabilities = model.predict_proba(X_val)[:, 1]


print("\n=== THRESHOLD COMPARISON ===")

thresholds = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70]

print(
    f"{'Threshold':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'False Pos':<12}"
    f"{'Missed Fraud':<12}"
)

for threshold in thresholds:
    predictions = (
        fraud_probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )

    print(
        f"{threshold:<12.2f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{fp:<12}"
        f"{fn:<12}"
    )

print("\n=== RANKING PERFORMANCE ===")
print(
    "Average precision:",
    round(
        average_precision_score(
            y_val,
            fraud_probabilities
        ),
        4
    )
)

print("\nTest dataset remains untouched.")
