import pandas as pd
import rasterio
import os

# Paths
RAINFALL_FILE = "data/raw/rainfall/india_grid_rainfall.csv"
DEM_FILE = "data/raw/elevation/merged_dem/merged_dem.tif"
OUTPUT_FILE = "data/processed/elevation_features.csv"

def extract_elevation():
    # Read rainfall data
    df = pd.read_csv(RAINFALL_FILE)

    # Open DEM raster
    with rasterio.open(DEM_FILE) as src:
        elevations = []

        for _, row in df.iterrows():
            lon = row["longitude"]
            lat = row["latitude"]

            # Convert lat/lon to raster row/col
            row_idx, col_idx = src.index(lon, lat)

            try:
                elevation = src.read(1)[row_idx, col_idx]
            except:
                elevation = None

            elevations.append(elevation)

    df["elevation_m"] = elevations

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("✅ Elevation extraction completed successfully.")

if __name__ == "__main__":
    extract_elevation()