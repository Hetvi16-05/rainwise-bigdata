import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import pandas as pd

WATERSHED_PATH = "data/processed/hydrology/watersheds_clean.geojson"
RAINFALL_RASTER = "data/processed/rainfall/chirps_gujarat.tif"
OUTPUT_PATH = "data/processed/features/watershed_rainfall.csv"

print("Loading watersheds...")
watersheds = gpd.read_file(WATERSHED_PATH)

# ---- FIX SWAPPED COORDINATES ----
from shapely.ops import transform
def swap_xy(geom):
    return transform(lambda x, y: (y, x), geom)

watersheds["geometry"] = watersheds["geometry"].apply(swap_xy)

print("Opening raster...")
src = rasterio.open(RAINFALL_RASTER)

# Ensure CRS match
watersheds = watersheds.to_crs(src.crs)

means = []

print("Computing rainfall per watershed...")

for geom in watersheds.geometry:
    try:
        out_image, out_transform = mask(src, [geom], crop=True)
        data = out_image[0]
        
        # Remove nodata
        data = data[data != src.nodata]
        
        if len(data) == 0:
            means.append(0)
        else:
            means.append(np.mean(data))
    except:
        means.append(0)

watersheds["avg_rainfall_mm"] = means

result = pd.DataFrame({
    "watershed_id": watersheds.index,
    "avg_rainfall_mm": watersheds["avg_rainfall_mm"]
})

result.to_csv(OUTPUT_PATH, index=False)

print("✅ Watershed rainfall aggregation completed!")