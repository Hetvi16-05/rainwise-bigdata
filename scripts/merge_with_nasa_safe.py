import pandas as pd

print("📥 Loading data...")

train = pd.read_csv("data/processed/train_with_city.csv")
nasa = pd.read_csv("data/processed/nasa_city_level.csv")

# =========================================================
# SAFE MERGE (NO EXPLOSION)
# =========================================================
print("🔗 Merging safely...")

merged = pd.merge(
    train,
    nasa,
    on="city",
    how="left"
)

print("Shape:", merged.shape)

merged.to_csv("data/processed/final_merged.csv", index=False)

print("✅ Merge complete (safe)")
