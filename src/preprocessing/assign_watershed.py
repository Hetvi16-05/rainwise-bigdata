import geopandas as gpd
import pandas as pd

INPUT_FILE = "data/processed/final_features_complete.csv"
WATERSHED_FILE = "data/raw/static/hydrology/watersheds/Watershed Boundary.geojson"
OUTPUT_FILE = "data/processed/final_features_with_watershed.csv"


def assign_watershed():
    df = pd.read_csv(INPUT_FILE)

    gdf_points = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326"
    )

    watersheds = gpd.read_file(WATERSHED_FILE)

    # Ensure CRS match
    watersheds = watersheds.to_crs("EPSG:4326")

    # First spatial join
    joined = gpd.sjoin(
        gdf_points,
        watersheds[["Watershed_full_Code", "geometry"]],
        how="left",
        predicate="intersects"
    )

    # Rename column immediately
    joined.rename(columns={"Watershed_full_Code": "watershed_id"}, inplace=True)

    # Drop spatial join helper column if exists
    if "index_right" in joined.columns:
        joined = joined.drop(columns=["index_right"])

    # Handle missing watershed_id using nearest in projected CRS
    missing = joined["watershed_id"].isna()

    if missing.any():
        print("Assigning nearest watershed for missing points...")

        # Project both layers to metric CRS
        points_proj = joined[missing].to_crs("EPSG:3857")
        watersheds_proj = watersheds.to_crs("EPSG:3857")

        nearest = gpd.sjoin_nearest(
            points_proj,
            watersheds_proj[["Watershed_full_Code", "geometry"]],
            how="left"
        )

        # Assign values back
        joined.loc[missing, "watershed_id"] = nearest["Watershed_full_Code"].values

    # Drop geometry
    joined = joined.drop(columns=["geometry"])

    joined.to_csv(OUTPUT_FILE, index=False)

    print("✅ Watershed integration completed successfully.")


if __name__ == "__main__":
    assign_watershed()