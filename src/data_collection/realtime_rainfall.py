"""
🌧️ Real-Time Rainfall Collection — Satellite Powered
======================================================
Fetches satellite-derived rainfall using gpm_fetcher module.
Replaces the previous direct API call approach.

Output: data/raw/realtime/rainfall/realtime_rainfall_log.csv
"""

import os
import logging
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

CITIES_FILE = os.path.join(BASE_DIR, "data/config/gujarat_cities.csv")

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data/raw/realtime/rainfall/realtime_rainfall_log.csv"
)


def safe_read_csv(file):
    if os.path.exists(file):
        return pd.read_csv(file, on_bad_lines="skip")
    return pd.DataFrame()


def safe_write_csv(df, file):
    temp_file = file + ".tmp"
    df.to_csv(temp_file, index=False)
    os.replace(temp_file, file)


def main():
    # Import gpm_fetcher
    from src.data_collection.gpm_fetcher import fetch_multiple_locations, save_satellite_data

    cities = pd.read_csv(CITIES_FILE)

    logger.info(f"🌧️ Fetching satellite rainfall for {len(cities)} cities...")

    # Fetch satellite rainfall for ALL cities
    satellite_df = fetch_multiple_locations(cities)

    # Build rows compatible with existing pipeline format
    rows = []
    for _, row in satellite_df.iterrows():
        rows.append({
            "date": datetime.now(),
            "city": row["city"],
            "lat": row["lat"],
            "lon": row["lon"],
            "precipitation_mm": row.get("rainfall_mm", 0.0),
            "hourly_max_mm": row.get("hourly_max_mm", 0.0),
            "hours_with_rain": row.get("hours_with_rain", 0),
            "source": row.get("source", "unknown"),
        })

    new_df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Append to existing log
    old_df = safe_read_csv(OUTPUT_FILE)
    df = pd.concat([old_df, new_df], ignore_index=True)

    # Maintain history (no cutoff for Big Data)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    safe_write_csv(df, OUTPUT_FILE)

    # Also save satellite-specific log
    save_satellite_data(satellite_df)

    logger.info(f"✅ Rainfall updated (satellite): {len(new_df)} new rows")
    print(f"✅ Rainfall updated (satellite): {len(new_df)} new rows")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    import sys
    sys.path.insert(0, BASE_DIR)
    main()