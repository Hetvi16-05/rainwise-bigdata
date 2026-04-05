import pandas as pd
import numpy as np

print("📥 Loading data...")

df = pd.read_csv("data/processed/rain_city_geo_features.csv")
cities = pd.read_csv("data/config/gujarat_cities.csv")

df.columns = df.columns.str.lower()
cities.columns = cities.columns.str.lower()

# ----------------------
# GET UNIQUE GRID POINTS (IMPORTANT)
# ----------------------
print("📍 Extracting unique grid points...")

grid = df[["lat", "lon"]].drop_duplicates().reset_index(drop=True)

# ----------------------
# MAP EACH CITY TO NEAREST GRID (ONCE ONLY)
# ----------------------
print("🧠 Mapping cities to nearest grid...")

city_grid_map = {}

for _, city_row in cities.iterrows():
    city = city_row["city"]
    lat = city_row["lat"]
    lon = city_row["lon"]

    grid["dist"] = (grid["lat"] - lat)**2 + (grid["lon"] - lon)**2
    nearest = grid.loc[grid["dist"].idxmin()]

    city_grid_map[city] = (nearest["lat"], nearest["lon"])

print("✅ Mapping complete")

# ----------------------
# BUILD DATASET
# ----------------------
print("🏗 Building dataset...")

final_list = []

for city, (lat, lon) in city_grid_map.items():
    print(f"Processing {city}...")

    city_data = df[
        (df["lat"] == lat) &
        (df["lon"] == lon)
    ].copy()

    city_data["city"] = city
    final_list.append(city_data)

final_df = pd.concat(final_list, ignore_index=True)

# ----------------------
# CREATE FLOOD LABEL
# ----------------------
print("🌊 Creating flood labels...")

final_df["flood"] = (
    (final_df["rain_7day"] > 150) &
    (final_df["distance_to_river_m"] < 5000)
).astype(int)

# ----------------------
# SELECT FEATURES
# ----------------------
final_df = final_df[[
    "city",
    "date",
    "rain_3day",
    "rain_7day",
    "elevation_m",
    "distance_to_river_m",
    "flood"
]]

# ----------------------
# SAVE
# ----------------------
output = "data/processed/final_city_dataset.csv"
final_df.to_csv(output, index=False)

print("✅ Saved:", output)
print("📊 Shape:", final_df.shape)
print(final_df["flood"].value_counts())