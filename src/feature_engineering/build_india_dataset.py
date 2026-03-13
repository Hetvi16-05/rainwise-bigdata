import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Load features
features = pd.read_csv("data/processed/final_features_complete.csv")

# Convert timestamp to date
features["date"] = pd.to_datetime(features["run_timestamp"]).dt.date

# Create geometry from lat/lon
geometry = [Point(xy) for xy in zip(features["longitude"], features["latitude"])]
gdf = gpd.GeoDataFrame(features, geometry=geometry, crs="EPSG:4326")

# Load India states shapefile
states = gpd.read_file("data/raw/static/boundaries/ne_10m_admin_1_states_provinces.shp")

# Keep only India
states = states[states["admin"] == "India"]

# Spatial join: assign each point to state
gdf = gpd.sjoin(gdf, states[["name", "geometry"]], how="left", predicate="within")

# Rename state column
gdf.rename(columns={"name": "state"}, inplace=True)

# Drop points outside India (if any)
gdf = gdf.dropna(subset=["state"])

# Aggregate to state-level daily
state_daily = gdf.groupby(["date", "state"]).agg({
    "precipitation_mm": "mean",
    "elevation_m": "mean",
    "distance_to_river_m": "mean",
    "slope_degree": "mean"
}).reset_index()

# Save
state_daily.to_csv("data/processed/state_daily_features.csv", index=False)

print("State-level features created ✅")