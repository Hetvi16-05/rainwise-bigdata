import pandas as pd
import datetime
import random
import os
from geopy.distance import geodesic


CITIES_FILE = "data/config/gujarat_cities.csv"

RIVER_DB = "river_database.csv"

OUTPUT_FILE = "data/raw/realtime/river/realtime_river_level_log.csv"


def find_nearest_river(lat, lon):

    df = pd.read_csv(RIVER_DB)

    min_dist = 999999
    nearest = None

    for _, row in df.iterrows():

        dist = geodesic(
            (lat, lon),
            (row["lat"], row["lon"])
        ).km

        if dist < min_dist:
            min_dist = dist
            nearest = row

    return nearest


def generate_level(danger):

    month = datetime.datetime.now().month

    if month in [6, 7, 8, 9]:
        level = random.uniform(danger - 3, danger + 3)
    else:
        level = random.uniform(5, danger - 5)

    return round(level, 2)


def get_status(level, warning, danger):

    if level >= danger:
        return "Above Danger"

    elif level >= warning:
        return "Warning"

    else:
        return "Normal"


def main():

    cities = pd.read_csv(CITIES_FILE)

    rows = []

    for _, row in cities.iterrows():

        city = row["city"]
        lat = row["lat"]
        lon = row["lon"]

        river_row = find_nearest_river(lat, lon)

        river = river_row["river"]
        station = river_row["station"]
        danger = river_row["danger"]
        warning = river_row["warning"]

        level = generate_level(danger)

        status = get_status(level, warning, danger)

        rows.append({
            "time": datetime.datetime.now(),
            "city": city,
            "lat": lat,
            "lon": lon,
            "river": river,
            "station": station,
            "level": level,
            "danger": danger,
            "warning": warning,
            "status": status
        })


    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print("✅ River updated for all cities")
    print("Rows:", len(df))


if __name__ == "__main__":
    main()