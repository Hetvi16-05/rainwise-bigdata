import pandas as pd
import rasterio
import os

INPUT_FILE = "data/processed/final_features.csv"
SLOPE_RASTER = "data/processed/elevation/slope.tif"
OUTPUT_FILE = "data/processed/final_features_with_slope.csv"

def extract_slope():
    df = pd.read_csv(INPUT_FILE)

    with rasterio.open(SLOPE_RASTER) as src:
        slope_values = []

        for _, row in df.iterrows():
            lon = row["longitude"]
            lat = row["latitude"]

            try:
                row_idx, col_idx = src.index(lon, lat)
                value = src.read(1)[row_idx, col_idx]
                slope = None if value == -9999 else value
            except:
                slope = None

            slope_values.append(slope)

    df["slope_degree"] = slope_values

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("✅ Slope extraction completed successfully.")

if __name__ == "__main__":
    extract_slope()