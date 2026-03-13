import pandas as pd
import numpy as np

INPUT_CSV = "data/processed/rainfall/gujarat_rainfall_features.csv"
OUTPUT_CSV = "data/processed/rainfall/gujarat_rainfall_advanced_features.csv"

df = pd.read_csv(INPUT_CSV)

# ---------------------------------------------------
# 1️⃣ Rainfall Anomaly (Monthly)
# ---------------------------------------------------

monthly_mean = df.groupby("Month")["Mean_Rainfall_mm"].transform("mean")
df["Rainfall_Anomaly_mm"] = df["Mean_Rainfall_mm"] - monthly_mean

# ---------------------------------------------------
# 2️⃣ Standardized Rainfall Index (Z-score)
# ---------------------------------------------------

monthly_std = df.groupby("Month")["Mean_Rainfall_mm"].transform("std")
df["Rainfall_Zscore"] = df["Rainfall_Anomaly_mm"] / monthly_std

# ---------------------------------------------------
# 3️⃣ Extreme Monsoon Year Flag (>90 percentile)
# ---------------------------------------------------

monsoon_90 = np.percentile(df["Monsoon_Total_mm"].unique(), 90)

df["Extreme_Monsoon_Year"] = df["Monsoon_Total_mm"] > monsoon_90

print("Monsoon 90th Percentile Threshold:", round(monsoon_90, 2))

# Save updated dataset
df.to_csv(OUTPUT_CSV, index=False)

print("Advanced rainfall feature dataset created ✅")
print("Saved at:", OUTPUT_CSV)