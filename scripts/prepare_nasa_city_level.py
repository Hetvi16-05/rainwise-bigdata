import pandas as pd

print("📥 Loading NASA data...")

nasa = pd.read_csv("data/processed/nasa_rainfall_gujarat.csv", low_memory=False)

# fix date
nasa["date"] = pd.to_datetime(nasa["date"], errors="coerce")
nasa = nasa.dropna(subset=["date"])

# =========================================================
# AGGREGATE (VERY IMPORTANT)
# =========================================================
print("📊 Aggregating NASA rainfall per city...")

city_rain = nasa.groupby("city")["rainfall"].agg([
    "mean",
    "max",
    "std"
]).reset_index()

city_rain.columns = [
    "city",
    "nasa_avg_rain",
    "nasa_max_rain",
    "nasa_std_rain"
]

print(city_rain.head())

# save
city_rain.to_csv("data/processed/nasa_city_level.csv", index=False)

print("✅ Saved: nasa_city_level.csv")
