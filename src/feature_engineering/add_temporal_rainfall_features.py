import pandas as pd

inp = "data/processed/gujarat_rainfall_history.csv"
out = "data/processed/gujarat_rainfall_history.csv"

df = pd.read_csv(inp)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date")

df["rain_3day"] = df["precipitation_mm"].rolling(3, min_periods=1).sum()
df["rain_7day"] = df["precipitation_mm"].rolling(7, min_periods=1).sum()

df.to_csv(out, index=False)

print("Saved:", out)
