import pandas as pd
from pathlib import Path

# -------------------------
# Files
# -------------------------

WEATHER = "data/raw/realtime/weather/realtime_weather_log.csv"
RAIN = "data/raw/realtime/rainfall/realtime_rainfall_log.csv"
RIVER = "data/raw/realtime/river/realtime_river_log.csv"

FEATURES = "data/processed/features/final_features.csv"

OUT = "data/processed/realtime_dataset.csv"

Path("data/processed").mkdir(parents=True, exist_ok=True)


# -------------------------
# function → latest row per city
# -------------------------

def latest(df):
    df = df.sort_values("timestamp")
    return df.groupby("city").tail(1)


# -------------------------
# load realtime files
# -------------------------

weather = latest(pd.read_csv(WEATHER))
rain = latest(pd.read_csv(RAIN))
river = latest(pd.read_csv(RIVER))


# -------------------------
# merge realtime data
# -------------------------

df = weather.merge(
    rain,
    on=["city", "lat", "lon"],
    how="left",
)

df = df.merge(
    river,
    on=["city", "lat", "lon"],
    how="left",
)


# -------------------------
# add static features
# -------------------------

try:

    feat = pd.read_csv(FEATURES)

    # match number of rows
    feat = feat.head(len(df))

    df = pd.concat([df.reset_index(drop=True), feat], axis=1)

except Exception as e:

    print("No features", e)


# -------------------------
# save dataset
# -------------------------

df.to_csv(OUT, index=False)

print("Saved:", OUT)