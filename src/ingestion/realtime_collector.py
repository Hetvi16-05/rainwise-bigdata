import requests
import pandas as pd
from datetime import datetime
import os

OUTPUT_FILE = "data/raw/realtime/weather/realtime_weather_log.csv"

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
    except Exception as e:
        print("❌ Location fetch failed:", e)
        return None

def fetch_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=precipitation,temperature_2m"
    response = requests.get(url)
    return response.json()

def main():
    location = get_user_location()

    if not location:
        print("❌ Could not detect location")
        return

    weather_data = fetch_weather(location["lat"], location["lon"])
    current = weather_data.get("current", {})

    record = {
        "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "city": location["city"],
        "region": location["region"],
        "country": location["country"],
        "latitude": location["lat"],
        "longitude": location["lon"],
        "api_time": current.get("time"),
        "precipitation_mm": current.get("precipitation"),
        "temperature_C": current.get("temperature_2m")
    }

    df = pd.DataFrame([record])

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    if os.path.exists(OUTPUT_FILE):
        df.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(OUTPUT_FILE, mode='w', header=True, index=False)

    print("✅ Real-time data appended successfully")

if __name__ == "__main__":
    main()
