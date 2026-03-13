import os
import re
import rasterio
import numpy as np
from collections import defaultdict

INPUT_DIR = "data/processed/rainfall/chirps_india_daily_full"
OUTPUT_DIR = "data/processed/rainfall/chirps_india_monthly"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Regex to extract date
date_pattern = re.compile(r"(\d{4})\.(\d{2})\.(\d{2})")

# Group files by (year, month)
monthly_files = defaultdict(list)

for filename in os.listdir(INPUT_DIR):
    match = date_pattern.search(filename)
    if match:
        year, month, _ = match.groups()
        key = (year, month)
        monthly_files[key].append(os.path.join(INPUT_DIR, filename))

print(f"Total months detected: {len(monthly_files)}")

for (year, month), files in sorted(monthly_files.items()):
    print(f"Processing {year}-{month} ({len(files)} days)")

    monthly_sum = None
    meta = None

    for file in files:
        with rasterio.open(file) as src:
            data = src.read(1)
            if monthly_sum is None:
                monthly_sum = np.zeros_like(data, dtype=np.float32)
                meta = src.meta.copy()
            
            monthly_sum += data

    meta.update(dtype=rasterio.float32)

    output_path = os.path.join(
        OUTPUT_DIR,
        f"chirps_monthly_{year}_{month}.tif"
    )

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(monthly_sum.astype(rasterio.float32), 1)

    print(f"Saved: {output_path}")

print("Monthly aggregation complete ✅")