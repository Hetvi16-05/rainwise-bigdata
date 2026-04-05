import pandas as pd
import rasterio

grid_file = "data/processed/gujarat_grid_025.csv"
dem_file = "data/raw/elevation/merged_dem/merged_dem.tif"

out_file = "data/processed/gujarat_elevation.csv"

grid = pd.read_csv(grid_file)

elevations = []

with rasterio.open(dem_file) as dem:

    for i, row in grid.iterrows():

        lat = row["lat"]
        lon = row["lon"]

        try:
            val = list(dem.sample([(lon, lat)]))[0][0]
        except:
            val = None

        elevations.append(val)

grid["elevation"] = elevations

print("Done, rows:", len(grid))

grid.to_csv(out_file, index=False)

print("Saved:", out_file)