import os
import pandas as pd
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
from datetime import datetime

# Paths
CHIRPS_DIR = "data/raw/rainfall/chirps_india_daily"
STATE_SHP = "data/raw/boundary/gadm41_IND_1.shp"
OUTPUT_FILE = "data/processed/state_daily_features.csv"

print("Loading India states shapefile...")
states = gpd.read_file(STATE_SHP)
states = states.to_crs("EPSG:4326")

records = []

print("Processing CHIRPS rasters...")

for file in sorted(os.listdir(CHIRPS_DIR)):
    if not file.endswith(".tif"):
        continue

    filepath = os.path.join(CHIRPS_DIR, file)

    # Extract date from filename
    date_str = file.split(".")[2:5]
    date = datetime.strptime(".".join(date_str), "%Y.%m.%d")

    print("Processing:", file)

    stats = zonal_stats(
        states,
        filepath,
        stats="mean",
        nodata=-9999
    )

    for idx, stat in enumerate(stats):
        records.append({
            "date": date,
            "state": states.iloc[idx]["NAME_1"],
            "precipitation_mm": stat["mean"]
        })

df = pd.DataFrame(records)

df.to_csv(OUTPUT_FILE, index=False)

print("✅ Historical state_daily_features created")
print("Total rows:", len(df))