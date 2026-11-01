
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "creditcard.csv")
df = df.sort_values("Time", kind="stable").reset_index(drop=True)

train_end = int(len(df) * 0.60)
val_end = int(len(df) * 0.80)

train = df.iloc[:train_end]
val = df.iloc[train_end:val_end]

features = [c for c in df.columns if c != "Class"]

X_train = train[features]
y_train = train["Class"]

X_val = val[features]
y_val = val["Class"].to_numpy()

models = {
    "Standard LR": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000, random_state=42
        ))
    ]),
    "Weighted LR": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ]),
    "Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=100,
        max_leaf_nodes=15,
        learning_rate=0.05,
        l2_regularization=1.0,
        random_state=42
    ),
}

# Three consecutive validation windows.
windows = np.array_split(np.arange(len(val)), 3)

for name, model in models.items():
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_val)[:, 1]

    print(f"\n=== {name.upper()} ===")

    for i, indices in enumerate(windows, start=1):
        labels = y_val[indices]
        window_scores = scores[indices]

        fraud_total = int(labels.sum())
        budget = min(20, len(indices))

        ranked = np.argsort(
            -window_scores, kind="stable"
        )[:budget]

        found = int(labels[ranked].sum())
        missed = fraud_total - found
        precision = found / budget

        ap = (
            average_precision_score(labels, window_scores)
            if fraud_total > 0 else float("nan")
        )

        print(f"\nWindow {i}")
        print(f"Transactions: {len(indices):,}")
        print(f"Actual fraud: {fraud_total}")
        print(f"Fraud found in top {budget}: {found}")
        print(f"Fraud missed: {missed}")
        print(f"Precision at {budget}: {precision:.4f}")
        print(f"Average precision: {ap:.4f}")

print("\nTest dataset remains untouched.")
