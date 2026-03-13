import pandas as pd
import os

print("Loading state daily dataset...")

df = pd.read_csv("data/processed/state_daily_features.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["state", "date"]).reset_index(drop=True)

# =====================================
# Create Lag & Rolling Features
# =====================================
df["rain_lag1"] = df.groupby("state")["precipitation_mm"].shift(1)
df["rain_lag2"] = df.groupby("state")["precipitation_mm"].shift(2)

df["rain_3day"] = (
    df.groupby("state")["precipitation_mm"]
    .rolling(window=3)
    .sum()
    .reset_index(level=0, drop=True)
)

df["rain_7day"] = (
    df.groupby("state")["precipitation_mm"]
    .rolling(window=7)
    .sum()
    .reset_index(level=0, drop=True)
)

df.fillna(0, inplace=True)

df["month"] = df["date"].dt.month
df["monsoon_flag"] = df["month"].isin([6,7,8,9]).astype(int)

# =====================================
# Load Flood Labels
# =====================================
print("Processing flood labels...")

flood_df = pd.read_csv("data/processed/flood_labels_clean.csv")
flood_df["date"] = pd.to_datetime(flood_df["date"])

# Get unique state names from rainfall dataset
states = df["state"].unique()

# Expand flood rows into (date, state) pairs
expanded_rows = []

for _, row in flood_df.iterrows():
    for state in states:
        if state.lower() in row["Location"].lower():
            expanded_rows.append({
                "date": row["date"],
                "state": state,
                "flood": 1
            })

expanded_flood_df = pd.DataFrame(expanded_rows)

# Merge
df = df.merge(
    expanded_flood_df,
    on=["date", "state"],
    how="left"
)

df["flood"] = df["flood"].fillna(0).astype(int)

print("\nDataset summary:")
print("Total rows:", len(df))
print("Total flood events:", df["flood"].sum())

# =====================================
# Save
# =====================================
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/training_dataset_india_perfected.csv", index=False)

print("\nDataset saved successfully.")