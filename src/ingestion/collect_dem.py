import os
import requests

# Small demo bounding box (expand later)
LAT_MIN = 20
LAT_MAX = 23
LON_MIN = 72
LON_MAX = 75

BASE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi"

OUTPUT_DIR = "data/raw/elevation/srtm_tiles"

def download_tile(lat, lon):
    lat_prefix = "N" if lat >= 0 else "S"
    lon_prefix = "E" if lon >= 0 else "W"

    tile_name = f"{lat_prefix}{abs(lat):02d}{lon_prefix}{abs(lon):03d}.hgt.gz"
    url = f"{BASE_URL}/{tile_name[:3]}/{tile_name}"

    output_path = os.path.join(OUTPUT_DIR, tile_name)

    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(8192):
                    f.write(chunk)
            print(f"✅ Downloaded {tile_name}")
        else:
            print(f"❌ Tile not found: {tile_name}")
    except Exception as e:
        print(f"Error downloading {tile_name}: {e}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for lat in range(LAT_MIN, LAT_MAX + 1):
        for lon in range(LON_MIN, LON_MAX + 1):
            download_tile(lat, lon)

    print("🏔 Elevation tiles download complete.")

if __name__ == "__main__":
    main()