import geopandas as gpd
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

shp_path = os.path.join(
    BASE_DIR,
    "data",
    "gis",
    "rivers",
    "ne_10m_rivers_lake_centerlines.shp"
)

output_path = os.path.join(
    BASE_DIR,
    "river_database.csv"
)

print("Loading shapefile...")

gdf = gpd.read_file(shp_path)

print("Total rivers:", len(gdf))


# India bounding box
MIN_LAT = 6
MAX_LAT = 37
MIN_LON = 68
MAX_LON = 97


rows = []

for _, row in gdf.iterrows():

    geom = row.geometry

    if geom is None:
        continue

    try:
        x, y = geom.coords[0]
    except:
        continue

    lat = y
    lon = x

    if (
        MIN_LAT <= lat <= MAX_LAT
        and MIN_LON <= lon <= MAX_LON
    ):

        name = row.get("name", "Unknown")

        rows.append(
            {
                "river": name,
                "station": name,
                "lat": lat,
                "lon": lon,
                "danger": 20,
                "warning": 18,
            }
        )


df = pd.DataFrame(rows)

df = df.drop_duplicates(subset=["lat", "lon"])

df.to_csv(output_path, index=False)

print("✅ India river database created")
print("Rows:", len(df))
print("Saved to:", output_path)