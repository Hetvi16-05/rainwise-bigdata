import pandas as pd

print("📥 Loading data...")

train = pd.read_csv("data/processed/train_with_city.csv")
nasa = pd.read_csv("data/processed/nasa_rainfall_gujarat.csv", low_memory=False)

# fix date
nasa["date"] = pd.to_datetime(nasa["date"], errors="coerce")
nasa = nasa.dropna(subset=["date"])

# rename rainfall
nasa = nasa.rename(columns={"rainfall": "nasa_rain"})

# =========================================================
# MERGE (ONLY CITY)
# =========================================================
print("🔗 Merging...")

merged = pd.merge(
    train,
    nasa[["city", "nasa_rain"]],
    on="city",
    how="left"
)

print("Shape:", merged.shape)

merged.to_csv("data/processed/final_merged.csv", index=False)

print("✅ Merge complete")
