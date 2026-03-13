import pandas as pd
import rasterio
import os

INPUT_FILE = "data/processed/final_features_with_slope.csv"
LULC_RASTER = "data/raw/lulc/gujarat_lulc_merged.tif"
OUTPUT_FILE = "data/processed/final_features_complete.csv"

def extract_lulc():
    df = pd.read_csv(INPUT_FILE)

    with rasterio.open(LULC_RASTER) as src:
        lulc_values = []

        for _, row in df.iterrows():
            lon = row["longitude"]
            lat = row["latitude"]

            try:
                row_idx, col_idx = src.index(lon, lat)
                value = src.read(1)[row_idx, col_idx]
                lulc = None if value == 0 else value
            except:
                lulc = None

            lulc_values.append(lulc)

    df["land_cover_class"] = lulc_values

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("✅ LULC extraction completed successfully.")

if __name__ == "__main__":
    extract_lulc()