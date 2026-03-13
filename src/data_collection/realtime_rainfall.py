import os
import requests
import pandas as pd
from datetime import datetime, timedelta

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

CITIES_FILE = os.path.join(
    BASE_DIR,
    "data/config/gujarat_cities.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data/raw/realtime/rainfall/realtime_rainfall_log.csv"
)

def fetch_rainfall(lat, lon):

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}"
        f"&longitude={lon}"
        "&current=precipitation"
    )

    return requests.get(url).json()


def main():

    cities = pd.read_csv(CITIES_FILE)

    rows = []

    for _, row in cities.iterrows():

        city = row["city"]
        lat = row["lat"]
        lon = row["lon"]

        print("Rain:", city)

        try:

            data = fetch_rainfall(lat, lon)

            current = data.get("current", {})

            rows.append({
                "date": datetime.now(),
                "city": city,
                "lat": lat,
                "lon": lon,
                "precipitation_mm": current.get("precipitation"),
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


    df["date"] = pd.to_datetime(df["date"])

    cutoff = datetime.now() - timedelta(days=7)

    df = df[df["date"] >= cutoff]


    df.to_csv(OUTPUT_FILE, index=False)

    print("✅ Rainfall updated for all cities")
    print("Rows:", len(new_df))


if __name__ == "__main__":
    main()