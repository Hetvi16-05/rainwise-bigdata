import rasterio
import numpy as np
import os

DEM = "data/raw/elevation/merged_dem/merged_dem.tif"
OUT = "data/processed/elevation/slope.tif"


def compute_slope():

    os.makedirs("data/processed/elevation", exist_ok=True)

    with rasterio.open(DEM) as src:

        dem = src.read(1).astype(float)

        x, y = np.gradient(dem)

        slope = np.sqrt(x*x + y*y)

        meta = src.meta

    with rasterio.open(
        OUT,
        "w",
        driver="GTiff",
        height=slope.shape[0],
        width=slope.shape[1],
        count=1,
        dtype=slope.dtype,
        crs=meta["crs"],
        transform=meta["transform"],
    ) as dst:
        dst.write(slope, 1)

    print("✅ slope.tif created")


if __name__ == "__main__":
    compute_slope()