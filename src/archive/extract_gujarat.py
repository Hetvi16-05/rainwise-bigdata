import geopandas as gpd
import os

watersheds = gpd.read_file(
    "data/processed/hydrology/fixed/watersheds_fixed.geojson"
)

print("Total watersheds:", len(watersheds))
print("CRS:", watersheds.crs)

# Gujarat approximate bounding box
# Longitude: 68 to 75
# Latitude: 20 to 24.5

gujarat_ws = watersheds.cx[68:75, 20:24.5]

print("Gujarat watersheds:", len(gujarat_ws))

os.makedirs("data/processed/hydrology/gujarat", exist_ok=True)

gujarat_ws.to_file(
    "data/processed/hydrology/gujarat/gujarat_watersheds.geojson",
    driver="GeoJSON"
)

print("Saved Gujarat watersheds.")