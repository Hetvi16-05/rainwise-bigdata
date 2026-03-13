import geopandas as gpd
import rasterio
from rasterio import features
import numpy as np
import os

WATERSHED_PATH = "data/processed/hydrology/gujarat/gujarat_watersheds.geojson"
RAINFALL_PATH = "data/processed/rainfall/chirps_gujarat.tif"
OUTPUT_PATH = "data/processed/features/watershed_rainfall.csv"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

print("Loading watersheds...")
gdf = gpd.read_file(WATERSHED_PATH)

print("Loading rainfall raster...")
with rasterio.open(RAINFALL_PATH) as src:
    rain = src.read(1)
    transform = src.transform
    nodata = src.nodata

print("Computing mean rainfall per watershed...")

mean_rain = []

for geom in gdf.geometry:
    mask = features.geometry_mask(
        [geom],
        transform=transform,
        invert=True,
        out_shape=rain.shape
    )

    values = rain[mask]

    if nodata is not None:
        values = values[values != nodata]

    if len(values) > 0:
        mean_rain.append(float(np.mean(values)))
    else:
        mean_rain.append(np.nan)

gdf["mean_annual_rainfall_mm"] = mean_rain
gdf[["mean_annual_rainfall_mm"]].to_csv(OUTPUT_PATH, index=False)

print("✅ Climatology rainfall extraction completed!")