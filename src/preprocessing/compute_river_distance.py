import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os

RAIN_FILE = "data/processed/elevation_features.csv"
RIVER_FILE = "data/raw/rivers/india_rivers.geojson"
OUTPUT_FILE = "data/processed/final_features.csv"

def compute_distance():
    # Load rainfall + elevation data
    df = pd.read_csv(RAIN_FILE)

    # Convert rainfall points to GeoDataFrame
    geometry = [Point(xy) for xy in zip(df.longitude, df.latitude)]
    gdf_points = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # Load river network
    gdf_rivers = gpd.read_file(RIVER_FILE)

    # Reproject to metric system (meters)
    gdf_points = gdf_points.to_crs(epsg=3857)
    gdf_rivers = gdf_rivers.to_crs(epsg=3857)

    # Compute nearest distance
    distances = []

    for point in gdf_points.geometry:
        min_dist = gdf_rivers.distance(point).min()
        distances.append(min_dist)

    gdf_points["distance_to_river_m"] = distances

    # Back to lat/lon
    gdf_points = gdf_points.to_crs(epsg=4326)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    gdf_points.drop(columns="geometry").to_csv(OUTPUT_FILE, index=False)

    print("✅ Distance-to-river computed successfully.")

if __name__ == "__main__":
    compute_distance()