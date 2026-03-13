import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
import os

print("Loading watershed file...")

watersheds = gpd.read_file(
    "data/processed/hydrology/watersheds_clean.geojson"
)

print("Original bounds:", watersheds.total_bounds)

def swap_xy(geom):
    if geom.geom_type == "Polygon":
        return Polygon([(y, x) for x, y in geom.exterior.coords])
    elif geom.geom_type == "MultiPolygon":
        return MultiPolygon(
            [Polygon([(y, x) for x, y in poly.exterior.coords])
             for poly in geom.geoms]
        )
    return geom

print("Swapping latitude/longitude...")

watersheds["geometry"] = watersheds["geometry"].apply(swap_xy)

print("New bounds:", watersheds.total_bounds)

os.makedirs("data/processed/hydrology/fixed", exist_ok=True)

watersheds.to_file(
    "data/processed/hydrology/fixed/watersheds_fixed.geojson",
    driver="GeoJSON"
)

print("Saved corrected watershed file.")