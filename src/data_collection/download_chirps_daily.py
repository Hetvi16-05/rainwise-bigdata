import os
import requests
from datetime import datetime, timedelta

# ---- SETTINGS ----
START_DATE = datetime(2015, 1, 1)
END_DATE = datetime(2024, 12, 31)
OUTPUT_DIR = "data/raw/rainfall/chirps_daily"

BASE_URL = "http://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/p05"
os.makedirs(OUTPUT_DIR, exist_ok=True)

current = START_DATE

while current <= END_DATE:
    year = current.strftime("%Y")
    filename = f"chirps-v2.0.{current.strftime('%Y.%m.%d')}.tif"
    url = f"{BASE_URL}/{year}/{filename}"
    output_path = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(output_path):
        print(f"Downloading {filename}")
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(r.content)
            else:
                print("Missing:", filename)
        except:
            print("Error:", filename)

    current += timedelta(days=1)

print("✅ Download complete")