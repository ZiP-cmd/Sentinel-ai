
from pathlib import Path
import pandas as pd

# Locate the dataset relative to this Python file.
project_root = Path(__file__).resolve().parent.parent
dataset_path = project_root / "data" / "creditcard.csv"

# Load the CSV into a pandas DataFrame.
df = pd.read_csv(dataset_path)

# Examine the size and structure of our dataset.
print("\n=== DATASET SHAPE ===")
print(df.shape)

print("\n=== FIRST FIVE TRANSACTIONS ===")
print(df.head())

print("\n=== COLUMN NAMES ===")
print(df.columns.tolist())

# Check that our expected fraud labels exist.
print("\n=== TRANSACTION LABEL COUNTS ===")
print(df["Class"].value_counts().sort_index())

print("\n=== TRANSACTION LABEL PERCENTAGES ===")
print(
    df["Class"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(4)
)

# Inspect missing values and duplicates.
print("\n=== TOTAL MISSING VALUES ===")
print(df.isna().sum().sum())

print("\n=== EXACT DUPLICATE ROWS ===")
print(df.duplicated().sum())

# Analyze transaction amounts.
print("\n=== TRANSACTION AMOUNT STATISTICS ===")
print(df.groupby("Class")["Amount"].describe())
