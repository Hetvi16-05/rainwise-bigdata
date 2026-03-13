import requests
import pandas as pd
from datetime import datetime, timedelta, UTC
import os

OUTPUT_FILE = "data/raw/realtime/rainfall/user_rainfall_ml_ready.csv"

def get_user_location():
    try:
        response = requests.get("http://ip-api.com/json/")
        data = response.json()

        return {
            "city": data.get("city"),
            "region": data.get("regionName"),
            "country": data.get("country"),
            "lat": data.get("lat"),
            "lon": data.get("lon")
        }
    except:
        return None

def fetch_weather(lat, lon):
    today = datetime.now(UTC).date()
    past_date = today - timedelta(days=7)

    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}"
        f"&longitude={lon}"
        f"&current=precipitation,temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl"
        f"&daily=precipitation_sum,temperature_2m_max,temperature_2m_min,wind_speed_10m_max"
        f"&start_date={past_date}"
        f"&end_date={today}"
        f"&timezone=GMT"
    )

    response = requests.get(url)
    return response.json()

def main():
    location = get_user_location()

    if not location:
        print("❌ Could not detect location.")
        return

    data = fetch_weather(location["lat"], location["lon"])

    records = []

    # ---- DAILY HISTORICAL ----
    daily = data.get("daily", {})
    for i in range(len(daily.get("time", []))):
        records.append({
            "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "city": location["city"],
            "region": location["region"],
            "country": location["country"],
            "latitude": location["lat"],
            "longitude": location["lon"],
            "elevation": data.get("elevation"),
            "date": daily["time"][i],
            "precipitation_mm": daily["precipitation_sum"][i],
            "temp_max_C": daily["temperature_2m_max"][i],
            "temp_min_C": daily["temperature_2m_min"][i],
            "wind_speed_max_kmh": daily["wind_speed_10m_max"][i],
            "data_type": "historical"
        })

    # ---- CURRENT REALTIME ----
    current = data.get("current", {})
    records.append({
        "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "city": location["city"],
        "region": location["region"],
        "country": location["country"],
        "latitude": location["lat"],
        "longitude": location["lon"],
        "elevation": data.get("elevation"),
        "date": current.get("time"),
        "precipitation_mm": current.get("precipitation"),
        "temp_max_C": current.get("temperature_2m"),
        "temp_min_C": None,
        "wind_speed_max_kmh": current.get("wind_speed_10m"),
        "data_type": "realtime"
    })

    df = pd.DataFrame(records)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("✅ ML-ready rainfall dataset created successfully.")

if __name__ == "__main__":
    main()
