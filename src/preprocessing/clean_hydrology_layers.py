import geopandas as gpd
import os

RAW_BASE = "data/raw/static/hydrology"
PROCESSED_BASE = "data/processed/hydrology"

os.makedirs(PROCESSED_BASE, exist_ok=True)

layers = {
    "basins": "basins/Central Water Commission Basin.geojson",
    "sub_basins": "sub_basins/Central Water Commission Sub Basin.geojson",
    "watersheds": "watersheds/Watershed Boundary.geojson"
}

def clean_layer(name, path):
    print(f"\nProcessing {name}...")

    gdf = gpd.read_file(os.path.join(RAW_BASE, path))

    print("Original CRS:", gdf.crs)

    # Ensure WGS84
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    # Remove invalid geometries
    gdf = gdf[gdf.geometry.is_valid]

    # Standardize column names
    gdf.columns = [col.lower().replace(" ", "_") for col in gdf.columns]

    output_path = os.path.join(PROCESSED_BASE, f"{name}_clean.geojson")
    gdf.to_file(output_path, driver="GeoJSON")

    print(f"Saved cleaned file to {output_path}")
    print("Feature count:", len(gdf))


for name, path in layers.items():
    clean_layer(name, path)

print("\nAll hydrology layers cleaned successfully.")