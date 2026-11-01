
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"

# Load data with the same chronological split.
df = pd.read_csv(DATA_PATH)
df = df.sort_values("Time", kind="stable").reset_index(drop=True)

train_end = int(len(df) * 0.60)
validation_end = int(len(df) * 0.80)

train = df.iloc[:train_end]
validation = df.iloc[train_end:validation_end]

features = [col for col in df.columns if col != "Class"]

X_train = train[features]
y_train = train["Class"]

X_val = validation[features]
y_val = validation["Class"].to_numpy()

total_fraud = int(y_val.sum())

# These budgets represent alerts per validation period,
# not alerts per day.
alert_budgets = [30, 60, 100, 200]

configs = {
    "Standard": None,
    "Class Weighted": "balanced",
}

print("\n=== EQUAL ALERT BUDGET COMPARISON ===")

for name, weight in configs.items():
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight=weight,
            max_iter=1000,
            random_state=42,
        )),
    ])

    model.fit(X_train, y_train)
    scores = model.predict_proba(X_val)[:, 1]

    ap = average_precision_score(y_val, scores)

    # Rank transactions from highest risk to lowest risk.
    ranked_indices = np.argsort(-scores, kind="stable")

    print(f"\n=== {name.upper()} ===")
    print(f"Average precision: {ap:.4f}")

    print(
        f"{'Alerts':<10}"
        f"{'Fraud Found':<14}"
        f"{'False Alarms':<15}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
    )

    for budget in alert_budgets:
        selected_indices = ranked_indices[:budget]

        fraud_found = int(y_val[selected_indices].sum())
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
