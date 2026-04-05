import os
import rasterio
import numpy as np
import pandas as pd
import re

INPUT_DIR = "data/processed/rainfall/chirps_gujarat_monthly"
OUTPUT_CSV = "data/processed/rainfall/gujarat_monthly_mean_rainfall.csv"

records = []

pattern = re.compile(r"(\d{4})_(\d{2})")

for filename in sorted(os.listdir(INPUT_DIR)):
    if filename.endswith(".tif"):
        match = pattern.search(filename)
        if match:
            year, month = match.groups()
            path = os.path.join(INPUT_DIR, filename)

            with rasterio.open(path) as src:
                data = src.read(1).astype(np.float32)

                # Remove invalid values
                data[data < 0] = np.nan      # rainfall cannot be negative
                data[data > 10000] = np.nan  # remove abnormal values

                mean_rainfall = np.nanmean(data)

                records.append({
                    "Year": int(year),
                    "Month": int(month),
                    "Mean_Rainfall_mm": round(float(mean_rainfall), 2)
                })

df = pd.DataFrame(records)
df = df.sort_values(["Year", "Month"])
df.to_csv(OUTPUT_CSV, index=False)

print("Corrected monthly rainfall CSV created ✅")