import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------

CITY_FILE = "data/config/gujarat_cities.csv"
OUT_FILE = "data/raw/realtime/weather/realtime_weather_log.csv"

Path("data/raw/realtime/weather").mkdir(parents=True, exist_ok=True)


# -----------------------------
# Load cities
# -----------------------------

cities = pd.read_csv(CITY_FILE)


# -----------------------------
# Fetch weather
# -----------------------------

rows = []

for _, row in cities.iterrows():

    city = row["city"]
    lat = row["lat"]
    lon = row["lon"]

    print("Fetching:", city)

    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}"
        f"&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,rain"
    )

    try:
        r = requests.get(url, timeout=10).json()

        current = r["current"]

        rows.append(
            {
                "city": city,
                "lat": lat,
                "lon": lon,
                "temperature_C": current.get("temperature_2m"),
                "humidity_percent": current.get("relative_humidity_2m"),
                "wind_speed_kmh": current.get("wind_speed_10m"),
                "rain_mm": current.get("rain"),
                "timestamp": datetime.now(),
            }
        )

    except Exception as e:
        print("Error:", city, e)


df = pd.DataFrame(rows)


# -----------------------------
# Append to log
# -----------------------------

try:
    old = pd.read_csv(OUT_FILE)
    df = pd.concat([old, df])
except:
    pass


df.to_csv(OUT_FILE, index=False)

print("Saved:", OUT_FILE)