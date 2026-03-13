import requests
import pandas as pd
from datetime import datetime, timedelta
import os

from src.data_collection.run_realtime_pipeline import BASE_DIR

CITIES_FILE = os.path.join(BASE_DIR, "data/config/gujarat_cities.csv")

OUTPUT_FILE = os.path.join(BASE_DIR, "data/raw/realtime/weather/realtime_weather_log.csv")


def fetch_weather(lat, lon):

    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}"
        f"&longitude={lon}"
        f"&current=temperature_2m,precipitation,relative_humidity_2m,wind_speed_10m"
    )

    return requests.get(url).json()


def main():

    cities = pd.read_csv(CITIES_FILE)

    rows = []

    for _, row in cities.iterrows():

        city = row["city"]
        lat = row["lat"]
        lon = row["lon"]

        print("Fetching:", city)

        try:

            data = fetch_weather(lat, lon)

            current = data.get("current", {})

            rows.append({
                "timestamp": datetime.now(),
                "city": city,
                "lat": lat,
                "lon": lon,
                "temperature_C": current.get("temperature_2m"),
                "precipitation_mm": current.get("precipitation"),
                "humidity_percent": current.get("relative_humidity_2m"),
                "wind_speed_kmh": current.get("wind_speed_10m")
            })

        except:
            continue


    new_df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)


    if os.path.exists(OUTPUT_FILE):

        old_df = pd.read_csv(OUTPUT_FILE)

        df = pd.concat([old_df, new_df], ignore_index=True)

    else:

        df = new_df


    # keep last 7 days
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    cutoff = datetime.now() - timedelta(days=7)

    df = df[df["timestamp"] >= cutoff]


    df.to_csv(OUTPUT_FILE, index=False)

    print("✅ Weather updated for all cities")
    print("Rows:", len(new_df))


if __name__ == "__main__":
    main()