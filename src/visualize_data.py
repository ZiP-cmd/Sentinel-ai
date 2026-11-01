
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Visualization 1: Class distribution
class_counts = df["Class"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(
    ["Legitimate", "Fraud"],
    class_counts.values,
    color=["steelblue", "indianred"],
)

ax.set_title("Sentinel AI - Transaction Class Distribution")
ax.set_ylabel("Number of Transactions")

for index, count in enumerate(class_counts.values):
    ax.text(index, count, f"{count:,}", ha="center", va="bottom")

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "class_distribution.png", dpi=150)
plt.close(fig)

# Visualization 2: Transaction amount distributions
# A log scale makes low and high amounts easier to compare.
fig, ax = plt.subplots(figsize=(9, 5))

for class_value, label in [(0, "Legitimate"), (1, "Fraud")]:
    amounts = df.loc[df["Class"] == class_value, "Amount"]

    # log1p handles zero-dollar transactions safely.
    transformed_amounts = amounts.map(lambda x: __import__("numpy").log1p(x))

    ax.hist(
        transformed_amounts,
        bins=40,
        density=True,
        alpha=0.55,
        label=label,
    )

ax.set_title("Sentinel AI - Transaction Amount Distributions")
ax.set_xlabel("log(1 + Transaction Amount)")
ax.set_ylabel("Probability Density")
ax.legend()

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "amount_distribution.png", dpi=150)
plt.close(fig)

print("Visualizations created successfully!")
print(f"Saved to: {OUTPUT_DIR}")
