import pandas as pd
import numpy as np

INPUT_CSV = "data/processed/rainfall/gujarat_monthly_mean_rainfall.csv"
OUTPUT_CSV = "data/processed/rainfall/gujarat_rainfall_features.csv"

# Load data
df = pd.read_csv(INPUT_CSV)

# -------------------------------
# 1️⃣ Extreme Rainfall Index (95th percentile)
# -------------------------------

threshold_95 = np.percentile(df["Mean_Rainfall_mm"], 95)

df["Extreme_Rainfall"] = df["Mean_Rainfall_mm"] > threshold_95

print("95th Percentile Threshold:", round(threshold_95, 2))

# -------------------------------
# 2️⃣ Monsoon Rainfall (Jun–Sep)
# -------------------------------

monsoon = df[df["Month"].isin([6, 7, 8, 9])]
monsoon_yearly = monsoon.groupby("Year")["Mean_Rainfall_mm"].sum().reset_index()
monsoon_yearly.rename(columns={"Mean_Rainfall_mm": "Monsoon_Total_mm"}, inplace=True)

# -------------------------------
# 3️⃣ Annual Rainfall
# -------------------------------

annual = df.groupby("Year")["Mean_Rainfall_mm"].sum().reset_index()
annual.rename(columns={"Mean_Rainfall_mm": "Annual_Total_mm"}, inplace=True)

# -------------------------------
# Merge all features
# -------------------------------

features = df.merge(monsoon_yearly, on="Year")
features = features.merge(annual, on="Year")

# Save
features.to_csv(OUTPUT_CSV, index=False)

print("Rainfall feature dataset created ✅")
print("Saved at:", OUTPUT_CSV)