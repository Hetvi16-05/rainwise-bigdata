import pandas as pd

# Load datasets
state_features = pd.read_csv("data/processed/state_daily_features.csv")
flood = pd.read_csv("data/processed/flood_labels_clean.csv")

# Convert date columns
state_features["date"] = pd.to_datetime(state_features["date"])
flood["date"] = pd.to_datetime(flood["date"])

# Clean state names
state_features["state"] = state_features["state"].str.lower().str.strip()
flood["Location"] = flood["Location"].str.lower().str.strip()

# Merge
merged = state_features.merge(
    flood,
    left_on=["date", "state"],
    right_on=["date", "Location"],
    how="left"
)

# Fill missing flood values with 0
merged["flood"] = merged["flood"].fillna(0)

# Drop extra column
merged = merged.drop(columns=["Location"])

# Save final training dataset
merged.to_csv("data/processed/training_dataset.csv", index=False)

print("Final training dataset created ✅")