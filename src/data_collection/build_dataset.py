"""
📦 Real-Time Dataset Builder — Enhanced with Satellite & Geospatial Features
============================================================================
Merges weather, satellite rainfall, river levels, and geospatial features
into a single dataset ready for model inference.

Output: data/processed/realtime_dataset.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import logging
import sys
from src.bigdata.hdfs_simulator import HDFSSimulator

logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)

# ----------------------
# DATA PATHS
# ----------------------
WEATHER = os.path.join(BASE_DIR, "data/raw/realtime/weather/realtime_weather_log.csv")
RAIN = os.path.join(BASE_DIR, "data/raw/realtime/rainfall/realtime_rainfall_log.csv")
SATELLITE = os.path.join(BASE_DIR, "data/raw/realtime/satellite/satellite_rainfall_log.csv")
RIVER = os.path.join(BASE_DIR, "data/raw/realtime/river/realtime_river_level_log.csv")

# GIS data
RIVER_DIST = os.path.join(BASE_DIR, "data/processed/gujarat_river_distance.csv")
ELEVATION = os.path.join(BASE_DIR, "data/processed/gujarat_elevation.csv")

OUT = os.path.join(BASE_DIR, "data/processed/realtime_dataset.csv")
Path(os.path.dirname(OUT)).mkdir(parents=True, exist_ok=True)


# ----------------------
# UTILITIES
# ----------------------
def safe_read_csv(file):
    """Read CSV safely, handle missing/corrupt files."""
    if os.path.exists(file):
        try:
            df = pd.read_csv(file, on_bad_lines="skip")
            # Remove unnamed columns
            df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
            return df
        except Exception as e:
            logger.warning(f"⚠️ Error reading {file}: {e}")
    else:
        logger.info(f"ℹ️ File not found: {os.path.basename(file)}")
    return pd.DataFrame()


def latest(df, time_col):
    """Get the latest row per city."""
    if df.empty:
        return df

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    df = df.sort_values(time_col)

    return df.groupby("city").tail(1).reset_index(drop=True)


def safe_write_csv(df, file):
    """Atomic write to prevent corruption."""
    temp_file = file + ".tmp"
    df.to_csv(temp_file, index=False)
    os.replace(temp_file, file)


def find_nearest(gis_df, lat, lon, value_col):
    """Find nearest GIS value for a lat/lon."""
    if gis_df.empty or value_col not in gis_df.columns:
        return None
    gis_df = gis_df.copy()
    gis_df["_dist"] = (gis_df["lat"] - lat)**2 + (gis_df["lon"] - lon)**2
    nearest = gis_df.loc[gis_df["_dist"].idxmin()]
    return float(nearest[value_col])


# ----------------------
# MAIN BUILD
# ----------------------
def build():
    logger.info("📦 Building real-time dataset...")

    # --- Load raw data ---
    weather = safe_read_csv(WEATHER)
    rain = safe_read_csv(RAIN)
    satellite = safe_read_csv(SATELLITE)
    river = safe_read_csv(RIVER)

    # --- Load GIS data ---
    river_dist_df = safe_read_csv(RIVER_DIST)
    elevation_df = safe_read_csv(ELEVATION)

    # Normalize column names
    for df in [river_dist_df, elevation_df]:
        if not df.empty:
            df.columns = df.columns.str.lower()

    # --- Extract latest per city ---
    if not weather.empty:
        # Clean weather: only keep rows with actual data
        required_cols = ["timestamp", "city", "lat", "lon"]
        if all(c in weather.columns for c in required_cols):
            weather = weather.dropna(subset=["city"])
            weather = latest(weather, "timestamp")
        else:
            logger.warning("⚠️ Weather log missing required columns")
            weather = pd.DataFrame()

    if not rain.empty and "date" in rain.columns:
        rain = rain.dropna(subset=["city"])
        rain = latest(rain, "date")

    if not satellite.empty and "timestamp" in satellite.columns:
        satellite = satellite.dropna(subset=["city"])
        satellite = latest(satellite, "timestamp")

    if not river.empty and "timestamp" in river.columns:
        river = river.dropna(subset=["city"])
        river = latest(river, "timestamp")

    # --- Merge weather + rain ---
    if not weather.empty and not rain.empty:
        df = weather.merge(rain, on=["city", "lat", "lon"], how="left", suffixes=("", "_rain"))
    elif not weather.empty:
        df = weather.copy()
    elif not rain.empty:
        df = rain.copy()
    else:
        # Read cities list as base
        cities_file = os.path.join(BASE_DIR, "data/config/gujarat_cities.csv")
        df = safe_read_csv(cities_file)

    # --- Merge satellite rainfall ---
    if not satellite.empty and "city" in satellite.columns and "city" in df.columns:
        sat_cols = ["city"]
        if "rainfall_mm" in satellite.columns:
            sat_cols.append("rainfall_mm")
        if "hourly_max_mm" in satellite.columns:
            sat_cols.append("hourly_max_mm")
        if "source" in satellite.columns:
            sat_cols.append("source")

        df = df.merge(
            satellite[sat_cols].drop_duplicates(subset=["city"]),
            on="city", how="left", suffixes=("", "_sat")
        )
        logger.info(f"  📡 Satellite data merged: {len(satellite)} cities")

    # --- Merge river data ---
    if not river.empty and "city" in river.columns and "city" in df.columns:
        river_cols = ["city", "river", "station", "level", "danger", "warning", "status"]
        river_cols = [c for c in river_cols if c in river.columns]
        df = df.merge(
            river[river_cols].drop_duplicates(subset=["city"]),
            on="city", how="left", suffixes=("", "_river")
        )
        logger.info(f"  🏞️ River data merged: {len(river)} cities")

    # --- Add GIS features (elevation + river distance) ---
    if "lat" in df.columns and "lon" in df.columns:
        if not elevation_df.empty:
            elevations = []
            for _, row in df.iterrows():
                elev = find_nearest(elevation_df, row["lat"], row["lon"], "elevation")
                elevations.append(elev)
            df["elevation_m"] = elevations
            logger.info("  ⛰️ Elevation data added")

        if not river_dist_df.empty:
            distances = []
            for _, row in df.iterrows():
                dist = find_nearest(river_dist_df, row["lat"], row["lon"], "river_distance")
                distances.append(dist)
            df["distance_to_river_m"] = distances
            logger.info("  🏞️ River distance data added")

    # --- Compute best rainfall estimate ---
    # Priority: satellite > pipeline rainfall > 0
    if "rainfall_mm" in df.columns:
        df["rain_mm"] = df["rainfall_mm"].fillna(0.0)
    elif "precipitation_mm" in df.columns:
        df["rain_mm"] = df["precipitation_mm"].fillna(0.0)
    else:
        df["rain_mm"] = 0.0

    # --- Handle missing values ---
    df["elevation_m"] = df.get("elevation_m", pd.Series(dtype=float)).fillna(50.0)
    df["distance_to_river_m"] = df.get("distance_to_river_m", pd.Series(dtype=float)).fillna(50000.0)

    # --- Save (APPEND MODE for Big Data Velocity) ---
    if os.path.exists(OUT):
        df.to_csv(OUT, mode='a', header=False, index=False)
        logger.info(f"💾 Appended {len(df)} records to {OUT}")
    else:
        df.to_csv(OUT, index=False)
        logger.info(f"💾 Created {OUT} with {len(df)} records")

    # ==================================================
    # 🌉 HDFS BRIDGE (OFFICIAL HADOOP STORAGE)
    # ==================================================
    hdfs_dest = "hdfs://raw/realtime/realtime_dataset.csv"
    logger.info(f"🌉 Syncing real-time data to HDFS: {hdfs_dest}")
    
    # Push to HDFS
    HDFSSimulator.put(OUT, hdfs_dest, append=True)
    
    # --- DELETE LOCAL COPY (HDFS ONLY REQUIREMENT) ---
    if os.path.exists(OUT):
        os.remove(OUT)
        logger.info(f"🗑️ Local staging file deleted: {OUT} (Data now resides in HDFS)")
    
    logger.info(f"✅ Realtime pipeline complete. Data stored in HDFS: {hdfs_dest}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    build()