import pandas as pd
import numpy as np

print("📥 Loading datasets...")

train = pd.read_csv("data/processed/training_cleaned.csv")
nasa = pd.read_csv("data/processed/nasa_rainfall_gujarat.csv", low_memory=False)

# fix NASA date
nasa["date"] = pd.to_datetime(nasa["date"], errors="coerce")
nasa = nasa.dropna(subset=["date"])

# =========================================================
# 1. CREATE CITY REFERENCE TABLE
# =========================================================
print("📍 Creating city reference...")

city_ref = nasa.groupby("city")[["lat", "lon"]].mean().reset_index()

print("Cities:", len(city_ref))

# =========================================================
# 2. FUNCTION: FIND NEAREST CITY
# =========================================================
def find_nearest_city(lat, lon, city_df):
    distances = (
        (city_df["lat"] - lat) ** 2 +
        (city_df["lon"] - lon) ** 2
    )
    return city_df.iloc[distances.idxmin()]["city"]

# =========================================================
# 3. APPLY MAPPING (LIMIT FOR SPEED)
# =========================================================
print("🔗 Mapping cities... (this may take time)")

# OPTIONAL: sample for testing first
# train = train.sample(10000)

train["city"] = train.apply(
    lambda row: find_nearest_city(row["lat"], row["lon"], city_ref),
    axis=1
)

# =========================================================
# SAVE INTERMEDIATE
# =========================================================
train.to_csv("data/processed/train_with_city.csv", index=False)

print("✅ City mapping done & saved")
