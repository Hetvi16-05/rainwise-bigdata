import os
import rasterio
import geopandas as gpd
from rasterio.mask import mask

INPUT_DIR = "data/processed/rainfall/chirps_india_monthly"
OUTPUT_DIR = "data/processed/rainfall/chirps_gujarat_monthly"
SHAPEFILE = "data/raw/boundary/gadm41_IND_1.shp"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load India states shapefile
gdf = gpd.read_file(SHAPEFILE)

# Select only Gujarat
gujarat = gdf[gdf["NAME_1"] == "Gujarat"]

geometry = gujarat.geometry.values

for filename in sorted(os.listdir(INPUT_DIR)):
    if filename.endswith(".tif"):
        input_path = os.path.join(INPUT_DIR, filename)

        with rasterio.open(input_path) as src:
            out_image, out_transform = mask(src, geometry, crop=True)
            out_meta = src.meta.copy()

            out_meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

            output_path = os.path.join(OUTPUT_DIR, filename)

            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)

        print(f"Clipped: {filename}")

print("All monthly rasters clipped to Gujarat ✅")