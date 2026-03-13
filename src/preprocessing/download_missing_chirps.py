import os
import requests

MISSING_FILE = "missing_dates.txt"
DOWNLOAD_DIR = "data/raw/rainfall/chirps_india_daily_missing"

BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

with open(MISSING_FILE) as f:
    dates = [line.strip() for line in f if line.strip()]

print(f"Downloading {len(dates)} missing files...")

for date in dates:
    y, m, d = date.split("-")

    filename = f"chirps-v2.0.{y}.{m}.{d}.tif"
    url = f"{BASE_URL}/{y}/{filename}"   # ✅ Corrected URL

    outpath = os.path.join(DOWNLOAD_DIR, filename)

    if os.path.exists(outpath):
        print(f"{filename} already exists — skipping")
        continue

    print(f"Downloading {filename} ...")

    try:
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(outpath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("   ✔ Success")
        else:
            print(f"   ❌ Not available (status {r.status_code})")
    except Exception as e:
        print(f"   ⚠ Error: {e}")