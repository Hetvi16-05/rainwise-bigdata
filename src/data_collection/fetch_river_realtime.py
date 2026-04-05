import pandas as pd
from datetime import datetime
from pathlib import Path

RAIN_FILE = "data/raw/realtime/rainfall/realtime_rainfall_log.csv"
OUT_FILE = "data/raw/realtime/river/realtime_river_log.csv"

Path("data/raw/realtime/river").mkdir(parents=True, exist_ok=True)

rain = pd.read_csv(RAIN_FILE)

rows = []

for city in rain["city"].unique():

    df = rain[rain["city"] == city].tail(7)

    if df.empty:
        continue

    lat = df["lat"].iloc[-1]
    lon = df["lon"].iloc[-1]

    # fill missing rain with 0
    rain_vals = df["precipitation_mm"].fillna(0)

    rain_today = rain_vals.iloc[-1]
    rain_3 = rain_vals.tail(3).sum()
    rain_7 = rain_vals.sum()

    base = 2

    level = (
        base
        + rain_today * 0.05
        + rain_3 * 0.02
        + rain_7 * 0.01
    )

    rows.append(
        {
            "city": city,
            "lat": lat,
            "lon": lon,
            "river_level_m": round(level, 2),
            "rain_today": rain_today,
            "rain_3day": rain_3,
            "rain_7day": rain_7,
            "timestamp": datetime.now(),
        }
    )

df = pd.DataFrame(rows)

df.to_csv(OUT_FILE, index=False)

print("Saved:", OUT_FILE)