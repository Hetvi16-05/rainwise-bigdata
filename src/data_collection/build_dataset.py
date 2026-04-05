import pandas as pd
from pathlib import Path
import os

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

WEATHER = os.path.join(BASE_DIR, "data/raw/realtime/weather/realtime_weather_log.csv")
RAIN = os.path.join(BASE_DIR, "data/raw/realtime/rainfall/realtime_rainfall_log.csv")
RIVER = os.path.join(BASE_DIR, "data/raw/realtime/river/realtime_river_level_log.csv")

FEATURES = os.path.join(BASE_DIR, "data/processed/features/final_features.csv")

OUT = os.path.join(BASE_DIR, "data/processed/realtime_dataset.csv")

Path(os.path.dirname(OUT)).mkdir(parents=True, exist_ok=True)


# ----------------------
# safe read
# ----------------------
def safe_read_csv(file):
    if os.path.exists(file):
        return pd.read_csv(file, on_bad_lines="skip")
    print(f"⚠️ Missing file: {file}")
    return pd.DataFrame()


# ----------------------
# get latest per city
# ----------------------
def latest(df, time_col):

    if df.empty:
        return df

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    df = df.sort_values(time_col)

    return df.groupby("city").tail(1)


# ----------------------
# safe write
# ----------------------
def safe_write_csv(df, file):
    temp_file = file + ".tmp"
    df.to_csv(temp_file, index=False)
    os.replace(temp_file, file)


# ----------------------
# load data
# ----------------------
weather = safe_read_csv(WEATHER)
rain = safe_read_csv(RAIN)
river = safe_read_csv(RIVER)

# ----------------------
# extract latest rows
# ----------------------
weather = latest(weather, "timestamp")
rain = latest(rain, "date")
river = latest(river, "timestamp")

# ----------------------
# merge datasets
# ----------------------
df = weather.merge(
    rain,
    on=["city", "lat", "lon"],
    how="left"
)

df = df.merge(
    river,
    on=["city", "lat", "lon"],
    how="left"
)

# ----------------------
# add static features safely
# ----------------------
try:
    feat = safe_read_csv(FEATURES)

    # Better merge instead of head()
    if "city" in feat.columns:
        df = df.merge(feat, on="city", how="left")
    else:
        print("⚠️ Features file has no city column")

except Exception as e:
    print("⚠️ No features:", e)


# ----------------------
# clean duplicates
# ----------------------
df = df.drop_duplicates(subset=["city"], keep="last")

# ----------------------
# save safely
# ----------------------
safe_write_csv(df, OUT)

print("✅ Realtime dataset saved:", OUT)
print("Rows:", len(df))