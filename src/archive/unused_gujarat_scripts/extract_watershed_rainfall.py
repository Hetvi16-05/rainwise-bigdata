import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os

WATERSHED_PATH = "data/processed/hydrology/gujarat/gujarat_watersheds.geojson"
RAINFALL_PATH = "data/raw/rainfall/india_grid_rainfall.csv"
OUTPUT_PATH = "data/processed/features/watershed_rainfall.csv"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

print("Loading rainfall data...")
rain_df = pd.read_csv(RAINFALL_PATH)

# Convert to GeoDataFrame
geometry = [Point(xy) for xy in zip(rain_df.longitude, rain_df.latitude)]
rain_gdf = gpd.GeoDataFrame(rain_df, geometry=geometry, crs="EPSG:4326")

print("Loading watersheds...")
watersheds = gpd.read_file(WATERSHED_PATH)

print("Filtering rainfall points inside Gujarat extent...")
rain_gdf = gpd.sjoin(rain_gdf, watersheds, predicate="within")

print("Computing mean rainfall per watershed...")

rainfall_mean = (
    rain_gdf
    .groupby("watershed_code")["precipitation_mm"]
    .mean()
    .reset_index()
)

rainfall_mean.rename(columns={"precipitation_mm": "mean_rainfall"}, inplace=True)

rainfall_mean.to_csv(OUTPUT_PATH, index=False)

print("✅ Rainfall extraction completed!")