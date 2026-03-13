import pandas as pd

INPUT_FILE = "data/processed/training_dataset_india.csv"
OUTPUT_FILE = "data/processed/training_dataset_india_enhanced.csv"

print("Loading dataset...")
df = pd.read_csv(INPUT_FILE)

# Convert date properly
df["date"] = pd.to_datetime(df["date"])

# Sort properly (VERY IMPORTANT)
df = df.sort_values(["state", "date"])

print("Generating temporal rainfall features...")

# Group by state for rolling calculations
grouped = df.groupby("state")

# 3-day rolling sum
df["rain_3day"] = grouped["precipitation_mm"] \
    .rolling(window=3, min_periods=1) \
    .sum() \
    .reset_index(level=0, drop=True)

# 7-day rolling sum
df["rain_7day"] = grouped["precipitation_mm"] \
    .rolling(window=7, min_periods=1) \
    .sum() \
    .reset_index(level=0, drop=True)

# Lag features
df["rain_lag1"] = grouped["precipitation_mm"] \
    .shift(1)

df["rain_lag2"] = grouped["precipitation_mm"] \
    .shift(2)

# Fill initial lag NaNs with 0
df["rain_lag1"] = df["rain_lag1"].fillna(0)
df["rain_lag2"] = df["rain_lag2"].fillna(0)

print("Saving enhanced dataset...")

df.to_csv(OUTPUT_FILE, index=False)

print("===================================")
print("✅ Temporal Features Added")
print("New Columns:", df.columns.tolist())
print("Total Rows:", len(df))
print("===================================")