
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"


def main():
    # Load the original dataset without modifying it.
    df = pd.read_csv(DATA_PATH)

    required_columns = (
        ["Time", "Amount", "Class"]
        + [f"V{i}" for i in range(1, 29)]
    )

    if not all(col in df.columns for col in required_columns):
        raise ValueError("Dataset is missing required columns.")

    if df.empty or df[required_columns].isna().any().any():
        raise ValueError("Dataset is empty or has missing values.")

    if not df["Class"].isin([0, 1]).all():
        raise ValueError("Unexpected fraud labels detected.")

    # Sort chronologically.
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)

    # Preserve the original 1,081 duplicates for now.
    # We will investigate cross-split duplicate overlap below.
    train_end = int(len(df) * 0.60)
    validation_end = int(len(df) * 0.80)

    splits = {
        "train": df.iloc[:train_end].copy(),
        "validation": df.iloc[train_end:validation_end].copy(),
        "test": df.iloc[validation_end:].copy(),
    }

    for name, data in splits.items():
        label_counts = data["Class"].value_counts()

        print(f"\n=== {name.upper()} DATA ===")
        print(f"Rows: {len(data):,}")
        print(f"Legitimate: {label_counts.get(0, 0):,}")
        print(f"Fraudulent: {label_counts.get(1, 0):,}")
        print(f"Fraud rate: {data['Class'].mean() * 100:.4f}%")
        print(
            f"Time range: "
            f"{data['Time'].min():.0f} to "
            f"{data['Time'].max():.0f}"
        )

    # Evaluate exact row overlap between split datasets.
    columns = list(df.columns)

    train_rows = pd.MultiIndex.from_frame(
        splits["train"][columns]
    )
    validation_rows = pd.MultiIndex.from_frame(
        splits["validation"][columns]
    )
    test_rows = pd.MultiIndex.from_frame(
        splits["test"][columns]
    )

    print("\n=== CROSS-SPLIT DUPLICATE AUDIT ===")
    print(
        "Validation rows also seen in training:",
        validation_rows.isin(train_rows).sum()
    )
    print(
        "Test rows also seen in training:",
        test_rows.isin(train_rows).sum()
    )
    print(
        "Test rows also seen in validation:",
        test_rows.isin(validation_rows).sum()
    )

    print("\nData splitting audit complete.")
    print("No training or test files have been modified.")


if __name__ == "__main__":
    main()
