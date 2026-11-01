
from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "sentinel_fraud_model.joblib"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load the transaction dataset.
df = pd.read_csv(DATA_PATH)

# Preserve chronological ordering.
df = df.sort_values(
    "Time", kind="stable"
).reset_index(drop=True)

# Use our established 60% training split.
train_end = int(len(df) * 0.60)
train = df.iloc[:train_end]

# Separate features from fraud labels.
features = [col for col in df.columns if col != "Class"]

X_train = train[features]
y_train = train["Class"]

# Build our selected model.
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )),
])

print("Training Sentinel AI...")

model.fit(X_train, y_train)

# Save the trained model and feature names.
artifact = {
    "model": model,
    "features": features,
    "model_name": "Sentinel Weighted Logistic Regression",
}

joblib.dump(artifact, MODEL_PATH)

print("Model trained and saved successfully!")
print(f"Saved to: {MODEL_PATH}")
