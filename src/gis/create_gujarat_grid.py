import pandas as pd
import numpy as np
import os

CITIES = "data/config/gujarat_cities.csv"
OUT = "data/processed/gujarat_grid_025.csv"

STEP = 0.25


def create_grid():

    cities = pd.read_csv(CITIES)

    print("Columns:", cities.columns)

    # auto detect column names
    lat_col = None
    lon_col = None

    for c in cities.columns:
        if "lat" in c.lower():
            lat_col = c
        if "lon" in c.lower():
            lon_col = c

    if lat_col is None or lon_col is None:
        raise Exception("Latitude/Longitude column not found")

    rows = []

    for _, r in cities.iterrows():

        lat = r[lat_col]
        lon = r[lon_col]

        lat_vals = np.arange(lat - 0.5, lat + 0.5, STEP)
        lon_vals = np.arange(lon - 0.5, lon + 0.5, STEP)

        for la in lat_vals:
            for lo in lon_vals:
                rows.append([la, lo])

    df = pd.DataFrame(rows, columns=["lat", "lon"])

    os.makedirs("data/processed", exist_ok=True)

    df.to_csv(OUT, index=False)

    print("Points:", len(df))
    print("Saved:", OUT)


if __name__ == "__main__":
    create_grid()