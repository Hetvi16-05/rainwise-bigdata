import geopandas as gpd
import rasterio
from rasterio import features
import numpy as np
import os

WATERSHED_PATH = "data/processed/hydrology/gujarat/gujarat_watersheds.geojson"
SLOPE_PATH = "data/processed/elevation/gujarat_slope.tif"
OUTPUT_PATH = "data/processed/features/watershed_slope.csv"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

print("Loading watersheds...")
gdf = gpd.read_file(WATERSHED_PATH)

print("Loading slope raster...")
with rasterio.open(SLOPE_PATH) as src:
    slope = src.read(1)
    transform = src.transform
    nodata = src.nodata

print("Computing mean slope per watershed...")

mean_slopes = []

for geom in gdf.geometry:
    mask = features.geometry_mask(
        [geom],
        transform=transform,
        invert=True,
        out_shape=slope.shape
    )

    values = slope[mask]

    if nodata is not None:
        values = values[values != nodata]

    if len(values) > 0:
        mean_slopes.append(float(np.mean(values)))
    else:
        mean_slopes.append(np.nan)

gdf["mean_slope"] = mean_slopes
gdf[["mean_slope"]].to_csv(OUTPUT_PATH, index=False)

print("✅ Slope extraction completed!")