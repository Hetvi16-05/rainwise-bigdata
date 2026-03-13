import os
import requests
import gzip
import shutil
from datetime import datetime, timedelta
import subprocess
from src.utils.config import INDIA_BBOX

# ==============================
# Directories
# ==============================
RAW_DIR = "data/raw/rainfall/chirps_raw"
INDIA_DIR = "data/raw/rainfall/chirps_india_daily"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(INDIA_DIR, exist_ok=True)

# ==============================
# India Bounding Box (from config)
# ==============================
west = INDIA_BBOX["min_lon"]
south = INDIA_BBOX["min_lat"]
east = INDIA_BBOX["max_lon"]
north = INDIA_BBOX["max_lat"]

# ==============================
# Date Range
# ==============================
START_DATE = datetime(2016, 1, 1)
END_DATE = datetime(2019, 12, 31)

current_date = START_DATE

print("Starting India CHIRPS download... 🇮🇳")

while current_date <= END_DATE:
    year = current_date.strftime("%Y")
    month = current_date.strftime("%m")
    day = current_date.strftime("%d")

    filename = f"chirps-v2.0.{year}.{month}.{day}.tif"
    gz_filename = filename + ".gz"

    url = f"https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/p05/{year}/{gz_filename}"

    raw_gz_path = os.path.join(RAW_DIR, gz_filename)
    raw_tif_path = os.path.join(RAW_DIR, filename)
    india_path = os.path.join(INDIA_DIR, filename)

    if os.path.exists(india_path):
        print("Already exists:", filename)
        current_date += timedelta(days=1)
        continue

    try:
        print("Downloading:", filename)

        r = requests.get(url, timeout=60)

        if r.status_code != 200 or len(r.content) == 0:
            print("Download failed:", filename)
            current_date += timedelta(days=1)
            continue

        # Save .gz
        with open(raw_gz_path, "wb") as f:
            f.write(r.content)

        # Unzip
        with gzip.open(raw_gz_path, "rb") as f_in:
            with open(raw_tif_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Clip to India
        subprocess.run([
            "gdalwarp",
            "-multi",
            "-wo", "NUM_THREADS=ALL_CPUS",
            "-te",
            str(west), str(south), str(east), str(north),
            raw_tif_path,
            india_path
        ], check=True)

        # Cleanup
        os.remove(raw_gz_path)
        os.remove(raw_tif_path)

        print("Processed:", filename)

    except Exception as e:
        print("Error:", filename, e)

    current_date += timedelta(days=1)

print("All India CHIRPS downloads complete ✅")