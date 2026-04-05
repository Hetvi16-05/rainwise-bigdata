import pandas as pd
import numpy as np

print("📥 Loading city data...")

cities = pd.read_csv("data/config/gujarat_cities.csv")
cities.columns = cities.columns.str.lower()

# ----------------------
# STEP 1: FIND NEAREST GRID USING SMALL SAMPLE
# ----------------------
print("📍 Finding nearest grid points...")

sample_df = pd.read_csv(
    "data/processed/rain_city_geo_features.csv",
    nrows=50000  # small sample only
)

sample_df.columns = sample_df.columns.str.lower()
grid = sample_df[["lat", "lon"]].drop_duplicates()

city_grid_map = {}

for _, row in cities.iterrows():
    city = row["city"]
    lat = row["lat"]
    lon = row["lon"]

    grid["dist"] = (grid["lat"] - lat)**2 + (grid["lon"] - lon)**2
    nearest = grid.loc[grid["dist"].idxmin()]

    city_grid_map[city] = (nearest["lat"], nearest["lon"])

print("✅ Mapping ready")

# ----------------------
# STEP 2: PROCESS IN CHUNKS
# ----------------------
print("🏗 Building dataset safely...")

chunk_size = 50000
output_file = "data/processed/final_city_dataset.csv"

first_write = True

for chunk in pd.read_csv(
    "data/processed/rain_city_geo_features.csv",
    chunksize=chunk_size
):

    chunk.columns = chunk.columns.str.lower()

    result_rows = []

    for city, (lat, lon) in city_grid_map.items():
        city_data = chunk[
            (chunk["lat"] == lat) &
            (chunk["lon"] == lon)
        ].copy()

        if not city_data.empty:
            city_data["city"] = city
            result_rows.append(city_data)

    if result_rows:
        out_df = pd.concat(result_rows)

        # create flood label
        out_df["flood"] = (
            (out_df["rain_7day"] > 150) &
            (out_df["distance_to_river_m"] < 5000)
        ).astype(int)

        # select columns
        out_df = out_df[[
            "city",
            "date",
            "rain_3day",
            "rain_7day",
            "elevation_m",
            "distance_to_river_m",
            "flood"
        ]]

        # append to file
        if first_write:
            out_df.to_csv(output_file, index=False)
            first_write = False
        else:
            out_df.to_csv(output_file, mode="a", header=False, index=False)

print("✅ DONE!")