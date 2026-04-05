import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------

CITY_FILE = "data/config/gujarat_cities.csv"
OUT_FILE = "data/raw/realtime/rainfall/realtime_rainfall_log.csv"

Path("data/raw/realtime/rainfall").mkdir(parents=True, exist_ok=True)


# -----------------------------
# Load cities
# -----------------------------

cities = pd.read_csv(CITY_FILE)


# -----------------------------
# Fetch rainfall
# -----------------------------

rows = []

for _, row in cities.iterrows():

    city = row["city"]
    lat = row["lat"]
    lon = row["lon"]

    print("Fetching rainfall:", city)

    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}"
        f"&longitude={lon}"
        f"&current=rain"
    )

    try:
        r = requests.get(url, timeout=10).json()

        rain = r["current"].get("rain", 0)

        rows.append(
            {
                "city": city,
                "lat": lat,
                "lon": lon,
                "rain_mm": rain,
                "timestamp": datetime.now(),
            }
        )

    except Exception as e:
        print("Error:", city, e)


df = pd.DataFrame(rows)


# -----------------------------
# Append
# -----------------------------

try:
    old = pd.read_csv(OUT_FILE)
    df = pd.concat([old, df])
except:
    pass


df.to_csv(OUT_FILE, index=False)

print("Saved:", OUT_FILE)