import pandas as pd
import numpy as np
import os

# ----------------------
# PATHS
# ----------------------
INPUT_FILE = "data/processed/training_dataset_gujarat_advanced_labeled.csv"
OUTPUT_FILE = "data/processed/training_cleaned.csv"

print("📥 Loading dataset...")
df = pd.read_csv(INPUT_FILE, low_memory=False)

print("Original Shape:", df.shape)

# =========================================================
# 1. DROP USELESS / DUPLICATE COLUMNS
# =========================================================
print("\n🧹 Dropping unnecessary columns...")

drop_cols = [
    "date",        # corrupted
    "state",
    "lat_x", "lon_x",
    "lat_y", "lon_y",
    "river_distance",
    "precip_mm"
]

df = df.drop(columns=drop_cols, errors="ignore")

# =========================================================
# 2. HANDLE OUTLIERS (VERY IMPORTANT)
# =========================================================
print("\n📊 Handling outliers...")

# cap extreme rainfall
df["rain3_mm"] = df["rain3_mm"].clip(upper=5000)
df["rain7_mm"] = df["rain7_mm"].clip(upper=10000)

# distance cap (huge unrealistic values)
df["distance_to_river_m"] = df["distance_to_river_m"].clip(upper=500000)

# elevation cap
df["elevation_m"] = df["elevation_m"].clip(lower=0, upper=1000)

# =========================================================
# 3. REMOVE INVALID ROWS
# =========================================================
print("\n🚫 Removing invalid rows...")

df = df[df["distance_to_river_m"] > 0]
df = df[df["elevation_m"] >= 0]

# =========================================================
# 4. LOG TRANSFORM (REDUCE SKEW)
# =========================================================
print("\n📉 Applying log transform...")

df["log_rain3"] = np.log1p(df["rain3_mm"])
df["log_rain7"] = np.log1p(df["rain7_mm"])

# =========================================================
# 5. FINAL CHECK
# =========================================================
print("\n📊 Cleaned Shape:", df.shape)

print("\nFlood distribution:")
print(df["flood"].value_counts())

# =========================================================
# SAVE
# =========================================================
os.makedirs("data/processed", exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print("\n💾 Saved cleaned dataset →", OUTPUT_FILE)
print("✅ CLEANING COMPLETE")
