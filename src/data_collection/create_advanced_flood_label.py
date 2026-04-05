import pandas as pd
import os
import numpy as np

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data/processed/training_dataset_gujarat_labeled.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data/processed/training_dataset_gujarat_advanced_labeled.csv"
)

print("📥 Loading dataset...")
df = pd.read_csv(INPUT_FILE)

print("Columns:", df.columns.tolist())

# ----------------------
# SAFETY CHECK
# ----------------------
required = ["rain3_mm", "rain7_mm", "elevation_m", "distance_to_river_m"]

for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

# ----------------------
# CLEAN DATA
# ----------------------
print("🧹 Cleaning data...")

df = df.replace([float("inf"), -float("inf")], None)
df = df.fillna(df.median(numeric_only=True))
df = df.fillna(0)

# ----------------------
# ADVANCED NON-LINEAR SCORE
# ----------------------
print("🌊 Creating advanced flood label...")

score = (
    df["rain3_mm"] * 0.3 +
    df["rain7_mm"] * 0.2 +
    (1 / (df["distance_to_river_m"] + 1)) * 50000 +
    (1 / (df["elevation_m"] + 1)) * 2000
)

# normalize score → probability
prob = (score - score.min()) / (score.max() - score.min())

# add randomness (real-world uncertainty)
random_noise = np.random.rand(len(df))

df["flood"] = (prob > random_noise).astype(int)

# ----------------------
# CHECK DISTRIBUTION
# ----------------------
print("\nFlood Distribution:")
print(df["flood"].value_counts())

# ----------------------
# SAVE
# ----------------------
df.to_csv(OUTPUT_FILE, index=False)

print("\n✅ Saved:", OUTPUT_FILE)