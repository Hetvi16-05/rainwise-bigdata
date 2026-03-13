import geopandas as gpd
from rasterstats import zonal_stats
import numpy as np
import os

WATERSHED_PATH = "data/processed/hydrology/basins_clean.geojson"
DISTANCE_PATH = "data/processed/hydrology/distance_to_river.tif"
OUTPUT_PATH = "data/processed/features/watershed_distance_percent.csv"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

gdf = gpd.read_file(WATERSHED_PATH)

def percent_near_river(values):
    values = np.array(values)
    total = len(values)
    near = np.sum(values < 0.01)  # ~1 km
    return near / total if total > 0 else 0

stats = zonal_stats(
    gdf,
    DISTANCE_PATH,
    add_stats={"percent_near": percent_near_river}
)

gdf["percent_near_river"] = [s["percent_near"] for s in stats]

gdf[["percent_near_river"]].to_csv(OUTPUT_PATH, index=False)

print("✅ Percent-near-river feature computed!")