import os
import requests
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

CITIES_FILE = os.path.join(BASE_DIR, "data/config/gujarat_cities.csv")

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
            data = fetch_rainfall(row["lat"], row["lon"])
            current = data.get("current", {})

            rows.append({
                "date": datetime.now(),
                "city": row["city"],
                "lat": row["lat"],
                "lon": row["lon"],
                "precipitation_mm": current.get("precipitation"),
            })

        except Exception as e:
            print("Error:", e)

    new_df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    old_df = safe_read_csv(OUTPUT_FILE)

    df = pd.concat([old_df, new_df], ignore_index=True)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    safe_write_csv(df, OUTPUT_FILE)

    print("✅ Rainfall updated:", len(new_df))


if __name__ == "__main__":
    main()