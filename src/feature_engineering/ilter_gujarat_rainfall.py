import pandas as pd

inp = "data/processed/state_daily_features.csv"
out = "data/processed/gujarat_rainfall_history.csv"

df = pd.read_csv(inp)

df = df[df["state"] == "Gujarat"]

df.to_csv(out, index=False)

print("Saved:", out)
