import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Load features
features = pd.read_csv("data/processed/final_features_complete.csv")

# Add missing columns if not present
if "precipitation_mm" not in features.columns:
    features["precipitation_mm"] = 0

if "distance_to_river_m" not in features.columns:
    features["distance_to_river_m"] = 0

if "slope_degree" not in features.columns:
    features["slope_degree"] = 0

if "elevation_m" not in features.columns:
    features["elevation_m"] = 0

# Convert timestamp to date
if "run_timestamp" in features.columns:
    features["date"] = pd.to_datetime(features["run_timestamp"]).dt.date
else:
    features["date"] = pd.to_datetime("2000-01-01")

# Create geometry
geometry = [
    Point(xy)
    for xy in zip(
        features["longitude"],
        features["latitude"]
    )
]

gdf = gpd.GeoDataFrame(
    features,
    geometry=geometry,
    crs="EPSG:4326"
)

# Load India states
states = gpd.read_file(
    "data/raw/static/boundaries/ne_10m_admin_1_states_provinces.shp"
)

states = states[states["admin"] == "India"]

# Spatial join
gdf = gpd.sjoin(
    gdf,
    states[["name", "geometry"]],
    how="left",
    predicate="within"
)

gdf.rename(
    columns={"name": "state"},
    inplace=True
)

gdf = gdf.dropna(subset=["state"])

# Aggregate
state_daily = gdf.groupby(
    ["date", "state"]
).agg({
    "precipitation_mm": "mean",
    "elevation_m": "mean",
    "distance_to_river_m": "mean",
    "slope_degree": "mean"
}).reset_index()

state_daily.to_csv(
    "data/processed/state_daily_features.csv",
    index=False
)

print("State-level features created ✅")