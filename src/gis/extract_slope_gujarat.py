import pandas as pd
import rasterio
import numpy as np

grid_file = "data/processed/gujarat_grid_025.csv"
dem_file = "data/raw/elevation/merged_dem/merged_dem.tif"

out_file = "data/processed/gujarat_slope.csv"

grid = pd.read_csv(grid_file)

slopes = []

with rasterio.open(dem_file) as dem:

    data = dem.read(1)
    transform = dem.transform

    xres = transform[0]
    yres = -transform[4]

    gy, gx = np.gradient(data, yres, xres)
    slope = np.sqrt(gx**2 + gy**2)

    for i, row in grid.iterrows():

        lon = row["lon"]
        lat = row["lat"]

        try:
            rowcol = dem.index(lon, lat)
            val = slope[rowcol[0], rowcol[1]]
        except:
            val = None

        slopes.append(val)

grid["slope"] = slopes

grid.to_csv(out_file, index=False)

print("Saved:", out_file)