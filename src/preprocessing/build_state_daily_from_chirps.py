import os
import pandas as pd
import geopandas as gpd
from rasterstats import zonal_stats
from datetime import datetime

# ---------------- PATHS ----------------

CHIRPS_DIR = "data/raw/rainfall/chirps_india_daily"
STATE_SHP = "data/raw/boundary/gadm41_IND_1.shp"
OUTPUT_FILE = "data/processed/state_daily_features.csv"

print("Loading India states shapefile...")

states = gpd.read_file(STATE_SHP)
states = states.to_crs("EPSG:4326")

records = []

print("Processing CHIRPS rasters...")

files = sorted(os.listdir(CHIRPS_DIR))

for file in files:

    if not file.endswith(".tif"):
        continue

    if file.startswith("."):
        continue

    filepath = os.path.join(CHIRPS_DIR, file)

    # -------- FIXED DATE PARSE --------
    try:
        # chirps_india_2025_09_17.tif
        name = file.replace(".tif", "")
        parts = name.split("_")

        year = parts[2]
        month = parts[3]
        day = parts[4]

        date = datetime.strptime(
            f"{year}-{month}-{day}",
            "%Y-%m-%d"
        )

    except Exception as e:
        print("Skipping bad file:", file)
        continue

    print("Processing:", file)

    stats = zonal_stats(
        states,
        filepath,
        stats="mean",
        nodata=-9999
    )

    for idx, stat in enumerate(stats):

        rain = stat["mean"]

        if rain is None:
            rain = 0

        records.append({
            "date": date,
            "state": states.iloc[idx]["NAME_1"],
            "precipitation_mm": rain
        })


df = pd.DataFrame(records)

os.makedirs("data/processed", exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print("\n✅ Historical state_daily_features created")
print("Total rows:", len(df))