import pandas as pd

input_path = "data/processed/gujarat_rainfall_history.csv"
output_path = "data/processed/gujarat_features.csv"

print("Loading data...")

df = pd.read_csv(input_path)

print("Rows:", len(df))

# -------------------------
# Fix date
# -------------------------

df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

# sort for rolling
df = df.sort_values(["city", "date"])

print("Creating rolling features...")

# -------------------------
# Rolling features per city
# -------------------------

df["rain_3day"] = (
    df.groupby("city")["rain_mm"]
    .rolling(3)
    .sum()
    .reset_index(level=0, drop=True)
)

df["rain_7day"] = (
    df.groupby("city")["rain_mm"]
    .rolling(7)
    .sum()
    .reset_index(level=0, drop=True)
)

df["rain_lag1"] = (
    df.groupby("city")["rain_mm"]
    .shift(1)
)

df["rain_lag2"] = (
    df.groupby("city")["rain_mm"]
    .shift(2)
)

# -------------------------
# Flood rule
# -------------------------

print("Creating flood label...")

def flood_rule(row):

    if row["rain_7day"] > 120:
        return 1

    if row["rain_3day"] > 70:
        return 1

    if row["rain_mm"] > 50:
        return 1

    return 0


df["flood"] = df.apply(flood_rule, axis=1)

# -------------------------
# Remove NaN from rolling
# -------------------------

df = df.dropna()

print("Final rows:", len(df))

# -------------------------
# Save
# -------------------------

df.to_csv(output_path, index=False)

print("Saved to:", output_path)