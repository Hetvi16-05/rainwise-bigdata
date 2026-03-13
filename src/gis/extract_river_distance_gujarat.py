import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

grid_file = "data/processed/gujarat_grid_025.csv"
river_file = "data/raw/rivers/india_rivers.geojson"

out_file = "data/processed/gujarat_river_distance.csv"

grid = pd.read_csv(grid_file)

points = gpd.GeoDataFrame(
    grid,
    geometry=gpd.points_from_xy(grid.lon, grid.lat),
    crs="EPSG:4326"
)

rivers = gpd.read_file(river_file)

# ✅ convert to projected CRS (meters)
points = points.to_crs(3857)
rivers = rivers.to_crs(3857)

distances = []

for p in points.geometry:

    d = rivers.distance(p).min()
    distances.append(d)

points["river_distance"] = distances

points.drop(columns="geometry").to_csv(
    out_file,
    index=False
)

print("Saved:", out_file)