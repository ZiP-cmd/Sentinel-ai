
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"

# Reuse our established chronological split.
df = pd.read_csv(DATA_PATH)
df = df.sort_values(
    "Time", kind="stable"
).reset_index(drop=True)

train_end = int(len(df) * 0.60)
val_end = int(len(df) * 0.80)

train = df.iloc[:train_end]
validation = df.iloc[train_end:val_end]

features = [c for c in df.columns if c != "Class"]

X_train = train[features]
y_train = train["Class"]

X_val = validation[features]
y_val = validation["Class"].to_numpy()

print("Training gradient boosting model...")

model = HistGradientBoostingClassifier(
    max_iter=100,
    max_leaf_nodes=15,
    learning_rate=0.05,
    l2_regularization=1.0,
    random_state=42,
)

model.fit(X_train, y_train)

print("Training complete!")

scores = model.predict_proba(X_val)[:, 1]
ap = average_precision_score(y_val, scores)

print("\n=== GRADIENT BOOSTING RESULTS ===")
print(f"Average precision: {ap:.4f}")

ranked = np.argsort(-scores, kind="stable")
total_fraud = int(y_val.sum())

print(
    f"{'Alerts':<10}"
    f"{'Fraud Found':<14}"
    f"{'False Alarms':<15}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
)

for budget in [30, 60, 100, 200]:
    selected = ranked[:budget]
    fraud_found = int(y_val[selected].sum())

    false_alarms = budget - fraud_found
    precision = fraud_found / budget
    recall = fraud_found / total_fraud

    print(
        f"{budget:<10}"
        f"{fraud_found:<14}"
        f"{false_alarms:<15}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
    )

print("\nOnly validation data was evaluated.")
print("The test dataset remains untouched.")
