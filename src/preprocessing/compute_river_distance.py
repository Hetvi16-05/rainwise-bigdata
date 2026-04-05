import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os

RAIN_FILE = "data/processed/elevation_features.csv"

RIVER_FILE = "data/gis/rivers/ne_10m_rivers_lake_centerlines.shp"

OUTPUT_FILE = "data/processed/final_features.csv"


def compute_distance():

    df = pd.read_csv(RAIN_FILE)

    geometry = [Point(xy) for xy in zip(df.longitude, df.latitude)]

    gdf_points = gpd.GeoDataFrame(
        df,
        geometry=geometry,
        crs="EPSG:4326"
    )

    print("Loading rivers...")
    gdf_rivers = gpd.read_file(RIVER_FILE)

    print("Projecting CRS...")
    gdf_points = gdf_points.to_crs(epsg=3857)
    gdf_rivers = gdf_rivers.to_crs(epsg=3857)

    print("Computing distance to river...")

    distances = []

    for point in gdf_points.geometry:
        min_dist = gdf_rivers.distance(point).min()
        distances.append(min_dist)

    gdf_points["distance_to_river_m"] = distances

    gdf_points = gdf_points.to_crs(epsg=4326)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    gdf_points.drop(columns="geometry").to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("✅ Distance-to-river computed successfully")


if __name__ == "__main__":
    compute_distance()