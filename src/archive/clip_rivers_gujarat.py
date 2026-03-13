import geopandas as gpd
import os

rivers = gpd.read_file(
    "data/processed/hydrology/rivers_india.geojson"
)

gujarat_ws = gpd.read_file(
    "data/processed/hydrology/gujarat/gujarat_watersheds.geojson"
)

# Keep only LineString
rivers = rivers[rivers.geometry.type == "LineString"]

# Spatial filter (fast bounding box clip)
rivers_gujarat = gpd.clip(rivers, gujarat_ws)

print("Rivers in Gujarat:", len(rivers_gujarat))

rivers_gujarat.to_file(
    "data/processed/hydrology/gujarat/gujarat_rivers.geojson",
    driver="GeoJSON"
)