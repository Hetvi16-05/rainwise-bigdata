import pandas as pd

input_path = "data/processed/gujarat_rainfall_history.csv"
output_path = "data/processed/gujarat_features_new.csv"

df = pd.read_csv(input_path)

df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

df = df.sort_values(["city", "date"])


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

df["rain_lag1"] = df.groupby("city")["rain_mm"].shift(1)
df["rain_lag2"] = df.groupby("city")["rain_mm"].shift(2)


def flood_rule(row):

    if row["rain_7day"] > 80:
        return 1
    if row["rain_3day"] > 40:
        return 1
    if row["rain_mm"] > 25:
        return 1
    return 0


df["flood"] = df.apply(flood_rule, axis=1)

df = df.dropna()

df.to_csv(output_path, index=False)

print("Saved:", output_path)