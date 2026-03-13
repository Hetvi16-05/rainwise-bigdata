import geopandas as gpd
import os

print("Loading datasets...")

watersheds = gpd.read_file(
    "data/processed/hydrology/watersheds_clean.geojson"
)

rivers = gpd.read_file(
    "data/processed/hydrology/rivers_india.geojson"
)

print("Watersheds:", len(watersheds))
print("Rivers (raw):", len(rivers))

# ------------------------------------------------------------------
# STEP 1 — Clean river geometries properly
# ------------------------------------------------------------------

# Keep only LineString geometries
rivers = rivers[rivers.geometry.type == "LineString"]

print("Rivers (LineString only):", len(rivers))

# ------------------------------------------------------------------
# STEP 2 — Project to metric CRS
# ------------------------------------------------------------------

watersheds_m = watersheds.to_crs("EPSG:3857")
rivers_m = rivers.to_crs("EPSG:3857")

# ------------------------------------------------------------------
# STEP 3 — Perform intersection
# ------------------------------------------------------------------

print("Performing geometric intersection...")
intersection = gpd.overlay(rivers_m, watersheds_m, how="intersection")

print("Intersected segments:", len(intersection))

if len(intersection) == 0:
    print("⚠️ No intersections found. Something still wrong.")
    exit()

# ------------------------------------------------------------------
# STEP 4 — Compute river length per watershed
# ------------------------------------------------------------------

intersection["river_length_m"] = intersection.geometry.length

river_length = (
    intersection.groupby("index_right")["river_length_m"]
    .sum()
    .reset_index()
)

watersheds_m = watersheds_m.reset_index()
watersheds_m["watershed_area_m2"] = watersheds_m.geometry.area

watersheds_m = watersheds_m.merge(
    river_length,
    left_on="index",
    right_on="index_right",
    how="left"
)

watersheds_m["river_length_m"] = watersheds_m["river_length_m"].fillna(0)

watersheds_m["river_density"] = (
    watersheds_m["river_length_m"] / watersheds_m["watershed_area_m2"]
)

# ------------------------------------------------------------------
# STEP 5 — Save
# ------------------------------------------------------------------

output = watersheds_m.to_crs("EPSG:4326")

os.makedirs("data/processed/hydrology/features", exist_ok=True)

output.to_file(
    "data/processed/hydrology/features/watersheds_with_river_density.geojson",
    driver="GeoJSON"
)

output.drop(columns="geometry").to_csv(
    "data/processed/hydrology/features/watersheds_river_density.csv",
    index=False
)

print("\nRiver density computed successfully.")
print("Total watersheds:", len(output))