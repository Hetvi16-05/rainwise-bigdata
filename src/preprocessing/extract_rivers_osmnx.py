import osmnx as ox
import geopandas as gpd
import os

OUTPUT_PATH = "data/processed/hydrology/rivers_clean.geojson"

print("Downloading India river network from OSM...")

# Get India boundary
india = ox.geocode_to_gdf("India")

# Extract waterways
tags = {"waterway": ["river", "stream", "canal"]}

rivers = ox.features_from_polygon(india.geometry.iloc[0], tags)

rivers = rivers.to_crs("EPSG:4326")

# Keep only LineStrings
rivers = rivers[rivers.geometry.type.isin(["LineString", "MultiLineString"])]

os.makedirs("data/processed/hydrology", exist_ok=True)

rivers.to_file(OUTPUT_PATH, driver="GeoJSON")

print("Total river features:", len(rivers))
print("Saved to:", OUTPUT_PATH)