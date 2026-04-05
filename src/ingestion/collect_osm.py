import requests
import os

URL = "https://download.geofabrik.de/asia/india-latest.osm.pbf"
OUTPUT_FILE = "data/raw/osm/india-latest.osm.pbf"

def download_osm():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    print("⬇️ Downloading India OSM data... (This may take a few minutes)")

    response = requests.get(URL, stream=True)

    if response.status_code == 200:
        with open(OUTPUT_FILE, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print("✅ Download completed successfully.")
    else:
        print("❌ Failed to download OSM data.")

if __name__ == "__main__":
    download_osm()