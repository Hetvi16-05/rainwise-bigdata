import os
import requests
import gzip
import shutil
from datetime import datetime, timedelta
import subprocess

# Separate raw + output folders (safe for parallel runs)
raw_dir = "data/raw/rainfall/chirps_raw_recent"
india_dir = "data/raw/rainfall/chirps_india_daily_recent"

os.makedirs(raw_dir, exist_ok=True)
os.makedirs(india_dir, exist_ok=True)

# India bounding box
west, south, east, north = 68, 6, 97, 37

# Date range
start_date = datetime(2025, 1, 1)
end_date = datetime(2025, 12, 31)

current_date = start_date

while current_date <= end_date:
    year = current_date.strftime("%Y")
    month = current_date.strftime("%m")
    day = current_date.strftime("%d")

    filename = f"chirps-v2.0.{year}.{month}.{day}.tif"
    gz_filename = filename + ".gz"

    url = f"https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/p05/{year}/{gz_filename}"

    raw_gz_path = os.path.join(raw_dir, gz_filename)
    raw_tif_path = os.path.join(raw_dir, filename)
    india_path = os.path.join(india_dir, filename)

    # Skip if already processed
    if os.path.exists(india_path):
        print("Already exists:", filename)
        current_date += timedelta(days=1)
        continue

    try:
        # Download
        r = requests.get(url, timeout=30)

        if r.status_code != 200:
            print("HTTP error:", filename)
            current_date += timedelta(days=1)
            continue

        if len(r.content) == 0:
            print("Empty content:", filename)
            current_date += timedelta(days=1)
            continue

        # Save .gz file
        with open(raw_gz_path, "wb") as f:
            f.write(r.content)

        # Unzip safely
        try:
            with gzip.open(raw_gz_path, 'rb') as f_in:
                with open(raw_tif_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        except Exception as e:
            print("Unzip failed:", filename, e)
            if os.path.exists(raw_gz_path):
                os.remove(raw_gz_path)
            current_date += timedelta(days=1)
            continue

        # Clip to India (optimized GDAL)
        subprocess.run([
            "gdalwarp",
            "-multi",
            "-wo", "NUM_THREADS=ALL_CPUS",
            "-te", str(west), str(south), str(east), str(north),
            raw_tif_path,
            india_path
        ], check=True)

        # Cleanup temporary files
        if os.path.exists(raw_gz_path):
            os.remove(raw_gz_path)
        if os.path.exists(raw_tif_path):
            os.remove(raw_tif_path)

        print("Processed:", filename)

    except Exception as e:
        print("Error:", filename, e)

    current_date += timedelta(days=1)

print("All downloads complete ✅")