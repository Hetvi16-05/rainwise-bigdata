import geopandas as gpd
import rasterio
from rasterio import features
import numpy as np
import os

WATERSHED_PATH = "data/processed/hydrology/gujarat/gujarat_watersheds.geojson"
DEM_PATH = "data/raw/elevation/merged_dem/merged_dem.tif"
OUTPUT_PATH = "data/processed/features/watershed_elevation.csv"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

print("Loading watersheds...")
gdf = gpd.read_file(WATERSHED_PATH)

print("Loading DEM...")
with rasterio.open(DEM_PATH) as src:
    dem = src.read(1)
    transform = src.transform
    nodata = src.nodata

print("Computing mean elevation per watershed...")

mean_elevations = []

for geom in gdf.geometry:
    mask = features.geometry_mask(
        [geom],
        transform=transform,
        invert=True,
        out_shape=dem.shape
    )

    values = dem[mask]

    if nodata is not None:
        values = values[values != nodata]

    if len(values) > 0:
        mean_elevations.append(float(np.mean(values)))
    else:
        mean_elevations.append(np.nan)

gdf["mean_elevation"] = mean_elevations
gdf[["mean_elevation"]].to_csv(OUTPUT_PATH, index=False)

print("✅ Elevation extraction completed!")