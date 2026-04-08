import pandas as pd
import datetime
import requests
import os
from geopy.distance import geodesic

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

CITIES_FILE = os.path.join(BASE_DIR, "data/config/gujarat_cities.csv")
RIVER_DB = os.path.join(BASE_DIR, "river_database.csv")

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data/raw/realtime/river/realtime_river_level_log.csv"
)


def safe_read_csv(file):
    if os.path.exists(file):
        return pd.read_csv(file, on_bad_lines="skip")
    return pd.DataFrame()


def safe_write_csv(df, file):
    temp_file = file + ".tmp"
    df.to_csv(temp_file, index=False)
    os.replace(temp_file, file)


def find_nearest_river(lat, lon, river_df):

    river_df["distance"] = river_df.apply(
        lambda r: geodesic((lat, lon), (r["lat"], r["lon"])).km,
        axis=1
    )

    return river_df.loc[river_df["distance"].idxmin()]


def fetch_river_discharge(lat, lon):
    """Fetch river discharge from Open-Meteo Flood API."""
    try:
        url = "https://flood-api.open-meteo.com/v1/flood"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "river_discharge",
            "forecast_days": 1,
            "timezone": "Asia/Kolkata"
        }
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if "daily" in data and "river_discharge" in data["daily"]:
            return float(data["daily"]["river_discharge"][0] or 0.0)
    except Exception as e:
        print(f"Error fetching discharge for {lat},{lon}: {e}")
    return 0.0


def get_status(discharge):
    """
    Get status based on discharge (m³/s).
    Note: Using generic thresholds since the database is level-based (m).
    """
    if discharge > 50:
        return "Above Danger"
    elif discharge > 15:
        return "Warning"
    return "Normal"


def main():

    cities = pd.read_csv(CITIES_FILE)
    river_df = safe_read_csv(RIVER_DB)

    rows = []

    for _, row in cities.iterrows():

        nearest = find_nearest_river(row["lat"], row["lon"], river_df)

        discharge = fetch_river_discharge(row["lat"], row["lon"])

        rows.append({
            "timestamp": datetime.datetime.now(),
            "city": row["city"],
            "lat": row["lat"],
            "lon": row["lon"],
            "river": nearest["river"],
            "station": nearest["station"],
            "level": discharge,  # Storing discharge in the level column for compatibility
            "danger": 50.0,      # Discharge-based danger threshold
            "warning": 15.0,     # Discharge-based warning threshold
            "status": get_status(discharge)
        })

    new_df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    old_df = safe_read_csv(OUTPUT_FILE)

    df = pd.concat([old_df, new_df], ignore_index=True)

    safe_write_csv(df, OUTPUT_FILE)

    print("✅ River updated:", len(new_df))


if __name__ == "__main__":
    main()