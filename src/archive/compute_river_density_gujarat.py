import geopandas as gpd
import os

print("Loading Gujarat datasets...")

# Load Gujarat watersheds
watersheds = gpd.read_file(
    "data/processed/hydrology/gujarat/gujarat_watersheds.geojson"
)

# Load Gujarat rivers
rivers = gpd.read_file(
    "data/processed/hydrology/gujarat/gujarat_rivers.geojson"
)

print("Watersheds:", len(watersheds))
print("Rivers:", len(rivers))

# ------------------------------------------------------------------
# STEP 1 — Project to metric CRS for accurate length/area
# ------------------------------------------------------------------

watersheds_m = watersheds.to_crs("EPSG:3857")
rivers_m = rivers.to_crs("EPSG:3857")

# ------------------------------------------------------------------
# STEP 2 — Perform geometric intersection
# ------------------------------------------------------------------

print("Performing intersection...")
intersection = gpd.overlay(rivers_m, watersheds_m, how="intersection")

print("Intersected segments:", len(intersection))

if len(intersection) == 0:
    print("⚠️ No intersections found.")
    exit()

# ------------------------------------------------------------------
# STEP 3 — Compute river length inside each watershed
# ------------------------------------------------------------------

intersection["river_length_m"] = intersection.geometry.length

# Print columns to identify watershed index column
print("Intersection columns:", intersection.columns)

# Detect watershed index column automatically
watershed_index_col = None
for col in intersection.columns:
    if "index" in col.lower():
        watershed_index_col = col
        break

# If no index column found, create one
if watershed_index_col is None:
    intersection = intersection.reset_index()
    watershed_index_col = "index"

print("Using watershed index column:", watershed_index_col)

# Aggregate river length per watershed
river_length = (
    intersection.groupby(watershed_index_col)["river_length_m"]
    .sum()
    .reset_index()
)

# ------------------------------------------------------------------
# STEP 4 — Merge with watershed areas
# ------------------------------------------------------------------

watersheds_m = watersheds_m.reset_index()
watersheds_m["watershed_area_m2"] = watersheds_m.geometry.area

watersheds_m = watersheds_m.merge(
    river_length,
    left_on="index",
    right_on=watershed_index_col,
    how="left"
)

watersheds_m["river_length_m"] = watersheds_m["river_length_m"].fillna(0)

# River density (meters per square meter)
watersheds_m["river_density"] = (
    watersheds_m["river_length_m"] / watersheds_m["watershed_area_m2"]
)

# ------------------------------------------------------------------
# STEP 5 — Save outputs
# ------------------------------------------------------------------

output = watersheds_m.to_crs("EPSG:4326")

os.makedirs("data/processed/hydrology/gujarat/features", exist_ok=True)

output.to_file(
    "data/processed/hydrology/gujarat/features/gujarat_watersheds_river_density.geojson",
    driver="GeoJSON"
)

output.drop(columns="geometry").to_csv(
    "data/processed/hydrology/gujarat/features/gujarat_river_density.csv",
    index=False
)

print("\nRiver density computed successfully.")
print("Watersheds processed:", len(output))
print("Files saved in: data/processed/hydrology/gujarat/features/")