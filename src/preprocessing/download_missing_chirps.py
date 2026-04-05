import os
import sys
import requests
from datetime import datetime

OUT_DIR = "data/raw/rainfall/chirps_india_daily"
TMP_DIR = "data/temp/chirps_tmp"

BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/p05"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)


def download_one(date_str):

    dt = datetime.strptime(date_str, "%Y-%m-%d")

    y = dt.year
    m = dt.month
    d = dt.day

    out_path = f"{OUT_DIR}/chirps_india_{y}_{m:02d}_{d:02d}.tif"

    if os.path.exists(out_path):
        return

    url = f"{BASE_URL}/{y}/chirps-v2.0.{y}.{m:02d}.{d:02d}.tif.gz"

    gz_path = f"{TMP_DIR}/{y}_{m:02d}_{d:02d}.tif.gz"
    tif_path = f"{TMP_DIR}/{y}_{m:02d}_{d:02d}.tif"

    print("Downloading", date_str)

    r = requests.get(url, timeout=60)

    if r.status_code != 200:
        print("Not found:", date_str)
        return

    with open(gz_path, "wb") as f:
        f.write(r.content)

    os.system(f"gunzip -f {gz_path}")

    os.system(
    f"gdalwarp -overwrite "
    f"-cutline data/external/boundary/gadm41_IND_1.shp "
    f"-crop_to_cutline "
    f"{tif_path} {out_path}"
)

    os.remove(tif_path)


if __name__ == "__main__":
    date = sys.argv[1]
    download_one(date)