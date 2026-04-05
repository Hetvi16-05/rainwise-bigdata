import os
import rasterio
import pandas as pd
from glob import glob

CHIRPS_DIR = "data/raw/rainfall/chirps_india_daily"
OUT = "data/processed/gujarat_rainfall_history.csv"

files = sorted(glob(os.path.join(CHIRPS_DIR, "*.tif")))

rows = []

for f in files[:2000]:  # limit for speed, remove later

    date = os.path.basename(f).split("_")[-3:]
    date = "-".join(date).replace(".tif", "")

    with rasterio.open(f) as src:
        arr = src.read(1)

        arr = arr[arr > -999]

        if arr.size == 0:
            mean = 0
        else:
            mean = float(arr.mean())

    rows.append({
        "date": date,
        "state": "Gujarat",
        "precipitation_mm": mean,
        "rain_3day": mean,
        "rain_7day": mean
    })

df = pd.DataFrame(rows)

df.to_csv(OUT, index=False)

print("Saved", OUT)
