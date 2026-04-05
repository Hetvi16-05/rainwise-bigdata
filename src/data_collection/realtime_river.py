import pandas as pd
import datetime
import random
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


def generate_level(danger):

    month = datetime.datetime.now().month

    if month in [6, 7, 8, 9, 10]:
        return round(random.uniform(danger - 3, danger + 3), 2)

    return round(random.uniform(5, danger - 5), 2)


def get_status(level, warning, danger):

    if level >= danger:
        return "Above Danger"
    elif level >= warning:
        return "Warning"
    return "Normal"


def main():

    cities = pd.read_csv(CITIES_FILE)
    river_df = safe_read_csv(RIVER_DB)

    rows = []

    for _, row in cities.iterrows():

        nearest = find_nearest_river(row["lat"], row["lon"], river_df)

        level = generate_level(nearest["danger"])

        rows.append({
            "timestamp": datetime.datetime.now(),
            "city": row["city"],
            "lat": row["lat"],
            "lon": row["lon"],
            "river": nearest["river"],
            "station": nearest["station"],
            "level": level,
            "danger": nearest["danger"],
            "warning": nearest["warning"],
            "status": get_status(level, nearest["warning"], nearest["danger"])
        })

    new_df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    old_df = safe_read_csv(OUTPUT_FILE)

    df = pd.concat([old_df, new_df], ignore_index=True)

    safe_write_csv(df, OUTPUT_FILE)

    print("✅ River updated:", len(new_df))


if __name__ == "__main__":
    main()