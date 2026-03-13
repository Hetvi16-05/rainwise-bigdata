import pandas as pd
import os
from datetime import datetime


# ✅ project root
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)


WEATHER_FILE = os.path.join(
    BASE_DIR,
    "data/raw/realtime/weather/realtime_weather_log.csv"
)

RAINFALL_FILE = os.path.join(
    BASE_DIR,
    "data/raw/realtime/rainfall/realtime_rainfall_log.csv"
)

RIVER_FILE = os.path.join(
    BASE_DIR,
    "data/raw/realtime/river/realtime_river_level_log.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data/processed/realtime_dataset.csv"
)


def latest_per_city(df, time_col):

    df[time_col] = pd.to_datetime(df[time_col])

    df = df.sort_values(time_col)

    df = df.groupby("city", as_index=False).last()

    return df


def main():

    weather = pd.read_csv(WEATHER_FILE)
    rainfall = pd.read_csv(RAINFALL_FILE)
    river = pd.read_csv(RIVER_FILE)

    weather = latest_per_city(weather, "timestamp")
    rainfall = latest_per_city(rainfall, "date")
    river = latest_per_city(river, "time")

    df = weather.merge(
        rainfall,
        on=["city", "lat", "lon"],
        how="left",
        suffixes=("", "_rain"),
    )

    df = df.merge(
        river,
        on=["city", "lat", "lon"],
        how="left",
        suffixes=("", "_river"),
    )

    keep_cols = [
        "city",
        "lat",
        "lon",
        "temperature_C",
        "humidity_percent",
        "wind_speed_kmh",
        "precipitation_mm_rain",
        "river",
        "level",
        "warning",
        "danger",
        "status",
    ]

    df = df[keep_cols]

    df.rename(
        columns={
            "precipitation_mm_rain": "rain_mm"
        },
        inplace=True,
    )

    # flood rule
    df["flood_risk"] = 0

    df.loc[
        (df["rain_mm"] > 20) |
        (df["level"] > df["warning"]),
        "flood_risk"
    ] = 1

    # date time
    now = datetime.now()

    df["date"] = now.strftime("%Y-%m-%d")
    df["time"] = now.strftime("%H:%M:%S")

    cols = ["date", "time"] + [
        c for c in df.columns
        if c not in ["date", "time"]
    ]

    df = df[cols]

    os.makedirs(
        os.path.join(BASE_DIR, "data/processed"),
        exist_ok=True
    )

    # append mode
    if os.path.exists(OUTPUT_FILE):

        df.to_csv(
            OUTPUT_FILE,
            mode="a",
            header=False,
            index=False,
        )

    else:

        df.to_csv(
            OUTPUT_FILE,
            index=False,
        )

    print("Dataset appended:", OUTPUT_FILE)
    print("Rows added:", len(df))


if __name__ == "__main__":
    main()