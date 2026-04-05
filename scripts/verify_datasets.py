import pandas as pd

NASA_FILE = "data/processed/nasa_rainfall_gujarat.csv"
TRAIN_FILE = "data/processed/training_dataset_gujarat_advanced_labeled.csv"

print("📥 Loading datasets...\n")

nasa = pd.read_csv(NASA_FILE, low_memory=False)
train = pd.read_csv(TRAIN_FILE, low_memory=False)

# =========================================================
# SAFE DATE FIX
# =========================================================
print("🛠 Fixing date formats safely...\n")

nasa["date"] = pd.to_datetime(nasa["date"], errors="coerce")

# DO NOT FIX TRAIN DATE (it's broken → ignore for now)

# =========================================================
# NASA CHECK
# =========================================================
print("=================================================")
print("📊 NASA DATA")
print("=================================================")

print("Shape:", nasa.shape)

print("\nMissing date:", nasa["date"].isna().sum())

print("\nValid date range:",
      nasa["date"].dropna().min(),
      "→",
      nasa["date"].dropna().max())

print("\nRainfall stats:\n", nasa["rainfall"].describe())

# =========================================================
# TRAIN CHECK
# =========================================================
print("\n\n=================================================")
print("📊 TRAIN DATA")
print("=================================================")

print("Shape:", train.shape)

print("\nColumns:\n", list(train.columns))

print("\nFlood distribution:\n", train["flood"].value_counts())

print("\nRain3 stats:\n", train["rain3_mm"].describe())

print("\nElevation stats:\n", train["elevation_m"].describe())

# =========================================================
# CHECK LOCATION POSSIBILITY
# =========================================================
print("\n\n=================================================")
print("📍 LOCATION CHECK")
print("=================================================")

if "lat" in train.columns and "lon" in train.columns:
    print("✅ Train has lat/lon → spatial modeling possible")
else:
    print("❌ No location columns")

# =========================================================
# FINAL DECISION
# =========================================================
print("\n\n=================================================")
print("🚀 FINAL DECISION")
print("=================================================")

print("""
✔ Use TRAIN dataset as PRIMARY
✔ Ignore TRAIN date (corrupted)
✔ Ignore NASA for now (date mismatch)
✔ Build model using spatial + rainfall features
""")