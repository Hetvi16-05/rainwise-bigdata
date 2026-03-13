import pandas as pd
import os

WEATHER_FILE = "data/raw/realtime/weather/realtime_weather_log.csv"
RAINFALL_FILE = "data/raw/realtime/rainfall/realtime_rainfall_log.csv"
RIVER_FILE = "data/raw/realtime/river/realtime_river_level_log.csv"

OUTPUT_FILE = "data/processed/realtime_dataset.csv"


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


    # ------------------
    # clean columns
    # ------------------

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


    # ------------------
    # flood rule
    # ------------------

    df["flood_risk"] = 0

    df.loc[
        (df["rain_mm"] > 20) |
        (df["level"] > df["warning"]),
        "flood_risk"
    ] = 1


    os.makedirs("data/processed", exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print("Dataset updated:", OUTPUT_FILE)
    print("Rows:", len(df))


if __name__ == "__main__":
    main()