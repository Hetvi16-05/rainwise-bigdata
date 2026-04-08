import requests
import pandas as pd
from datetime import datetime, timedelta
import os

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

CITIES_FILE = os.path.join(BASE_DIR, "data/config/gujarat_cities.csv")

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data/raw/realtime/weather/realtime_weather_log.csv"
)


def fetch_weather(lat, lon):
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}"
        f"&longitude={lon}"
        f"&current=temperature_2m,precipitation,relative_humidity_2m,wind_speed_10m,surface_pressure,cloud_cover"
    )
    return requests.get(url, timeout=10).json()


def safe_read_csv(file):
    if os.path.exists(file):
        return pd.read_csv(file, on_bad_lines="skip")
    return pd.DataFrame()


def safe_write_csv(df, file):
    temp_file = file + ".tmp"
    df.to_csv(temp_file, index=False)
    os.replace(temp_file, file)


def main():

    cities = pd.read_csv(CITIES_FILE)

    rows = []

    for _, row in cities.iterrows():
        try:
            data = fetch_weather(row["lat"], row["lon"])
            current = data.get("current", {})

            rows.append({
                "timestamp": datetime.now(),
                "city": row["city"],
                "lat": row["lat"],
                "lon": row["lon"],
                "temperature_C": current.get("temperature_2m"),
                "precipitation_mm": current.get("precipitation"),
                "humidity_percent": current.get("relative_humidity_2m"),
                "wind_speed_kmh": current.get("wind_speed_10m"),
                "surface_pressure": current.get("surface_pressure"),
                "cloud_cover_percent": current.get("cloud_cover")
            })

        except Exception as e:
            print("Error:", e)

    new_df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    old_df = safe_read_csv(OUTPUT_FILE)

    df = pd.concat([old_df, new_df], ignore_index=True)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    safe_write_csv(df, OUTPUT_FILE)

    print("✅ Weather updated:", len(new_df))


if __name__ == "__main__":
    main()