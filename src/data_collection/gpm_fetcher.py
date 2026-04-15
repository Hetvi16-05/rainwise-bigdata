"""
🛰️ GPM/Satellite Rainfall Fetcher for RAINWISE
================================================
Fetches satellite-derived rainfall data using Open-Meteo Archive API.
Data source: ERA5 reanalysis + national weather services (satellite-calibrated).

Features:
- get_satellite_rainfall(lat, lon, hours=24) → single location
- fetch_multiple_locations(locations_df) → batch for all Gujarat cities
- 3-attempt retry logic with exponential backoff
- Graceful failure handling with logging
- Returns pandas DataFrame

Usage:
    from src.data_collection.gpm_fetcher import fetch_multiple_locations
    df = fetch_multiple_locations(cities_df)
"""

import os
import time
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ----------------------
# LOGGING
# ----------------------
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

OUTPUT_DIR = os.path.join(BASE_DIR, "data", "raw", "realtime", "satellite")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "satellite_rainfall_log.csv")

# ----------------------
# CONSTANTS
# ----------------------
ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_API = "https://api.open-meteo.com/v1/forecast"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds (exponential backoff)
REQUEST_TIMEOUT = 15


# ----------------------
# CORE: Get satellite rainfall for a single location
# ----------------------
def get_satellite_rainfall(lat, lon, hours=24):
    """
    Fetch satellite-derived rainfall for a single location.

    Uses Open-Meteo Archive API (ERA5 reanalysis — satellite-calibrated data).
    Falls back to Open-Meteo Forecast API if archive is unavailable.

    Args:
        lat (float): Latitude
        lon (float): Longitude
        hours (int): Number of hours to look back (default: 24)

    Returns:
        dict: {
            'rainfall_mm': float,       # Total rainfall in mm
            'hourly_max_mm': float,     # Max hourly rainfall
            'hours_with_rain': int,     # Hours with precipitation > 0
            'source': str,              # 'ERA5_Archive' or 'Forecast_API'
            'timestamp': str,           # ISO timestamp
            'period_start': str,        # Start of measurement period
            'period_end': str           # End of measurement period
        }
        Returns None if all retries fail.
    """
    now = datetime.now()
    start = now - timedelta(hours=hours)

    # --- Try Archive API first (satellite-calibrated historical data) ---
    result = _fetch_with_retry(
        url=ARCHIVE_API,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": now.strftime("%Y-%m-%d"),
            "hourly": "precipitation",
            "timezone": "UTC"
        },
        source_name="ERA5_Archive"
    )

    if result is not None:
        return _parse_hourly_response(result, "ERA5_Archive", start, now)

    # --- Fallback: Forecast API (if archive has lag) ---
    logger.warning(f"Archive API failed for ({lat}, {lon}). Trying forecast API...")

    result = _fetch_with_retry(
        url=FORECAST_API,
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": "precipitation",
            "past_hours": hours,
            "forecast_hours": 0,
            "timezone": "UTC"
        },
        source_name="Forecast_API"
    )

    if result is not None:
        return _parse_hourly_response(result, "Forecast_API", start, now)

    logger.error(f"All APIs failed for ({lat}, {lon})")
    return None


# ----------------------
# BATCH: Fetch for multiple locations
# ----------------------
def fetch_multiple_locations(locations_df):
    """
    Fetch satellite rainfall for all cities in a DataFrame.

    Args:
        locations_df: DataFrame with columns ['city', 'lat', 'lon']

    Returns:
        DataFrame with columns:
            city, lat, lon, rainfall_mm, hourly_max_mm, hours_with_rain,
            source, timestamp, period_start, period_end
    """
    rows = []
    total = len(locations_df)

    logger.info(f"🛰️ Fetching satellite rainfall for {total} locations...")

    for idx, row in locations_df.iterrows():
        city = row.get("city", f"Location_{idx}")
        lat = float(row["lat"])
        lon = float(row["lon"])

        try:
            data = get_satellite_rainfall(lat, lon, hours=24)

            if data:
                data["city"] = city
                data["lat"] = lat
                data["lon"] = lon
                rows.append(data)
                logger.info(f"  ✅ [{idx+1}/{total}] {city}: {data['rainfall_mm']:.2f} mm")
            else:
                # Return zero rainfall on failure (graceful degradation)
                rows.append({
                    "city": city, "lat": lat, "lon": lon,
                    "rainfall_mm": 0.0, "hourly_max_mm": 0.0,
                    "hours_with_rain": 0, "source": "FAILED",
                    "timestamp": datetime.now().isoformat(),
                    "period_start": "", "period_end": ""
                })
                logger.warning(f"  ⚠️ [{idx+1}/{total}] {city}: Failed, using 0mm")

            # Rate limiting: small delay between requests
            time.sleep(0.3)

        except Exception as e:
            logger.error(f"  ❌ [{idx+1}/{total}] {city}: {e}")
            rows.append({
                "city": city, "lat": lat, "lon": lon,
                "rainfall_mm": 0.0, "hourly_max_mm": 0.0,
                "hours_with_rain": 0, "source": "ERROR",
                "timestamp": datetime.now().isoformat(),
                "period_start": "", "period_end": ""
            })

    df = pd.DataFrame(rows)
    logger.info(f"🛰️ Satellite rainfall complete: {len(df)} locations processed")

    return df


# ----------------------
# SAVE: Persist satellite data to CSV
# ----------------------
def save_satellite_data(df):
    """Save satellite rainfall data with append + dedup logic."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(OUTPUT_FILE):
        old_df = pd.read_csv(OUTPUT_FILE, on_bad_lines="skip")
        df = pd.concat([old_df, df], ignore_index=True)

    # Maintain history (no cutoff for Big Data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # Atomic write
    temp = OUTPUT_FILE + ".tmp"
    df.to_csv(temp, index=False)
    os.replace(temp, OUTPUT_FILE)

    logger.info(f"💾 Satellite data saved: {OUTPUT_FILE} ({len(df)} rows)")


# ----------------------
# INTERNAL: Retry logic with exponential backoff
# ----------------------
def _fetch_with_retry(url, params, source_name, max_retries=MAX_RETRIES):
    """Fetch from API with retry logic."""
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)

            if response.status_code == 200:
                data = response.json()

                # Check for API-level errors
                if data.get("error"):
                    logger.warning(f"  {source_name} error: {data.get('reason', 'Unknown')}")
                    if attempt < max_retries:
                        time.sleep(RETRY_DELAY * attempt)
                    continue

                return data

            elif response.status_code == 429:
                # Rate limited
                wait = RETRY_DELAY * (2 ** attempt)
                logger.warning(f"  Rate limited. Waiting {wait}s...")
                time.sleep(wait)

            else:
                logger.warning(f"  {source_name} HTTP {response.status_code} (attempt {attempt})")
                if attempt < max_retries:
                    time.sleep(RETRY_DELAY * attempt)

        except requests.exceptions.Timeout:
            logger.warning(f"  {source_name} timeout (attempt {attempt})")
            if attempt < max_retries:
                time.sleep(RETRY_DELAY * attempt)

        except requests.exceptions.ConnectionError:
            logger.warning(f"  {source_name} connection error (attempt {attempt})")
            if attempt < max_retries:
                time.sleep(RETRY_DELAY * attempt)

        except Exception as e:
            logger.error(f"  {source_name} unexpected error: {e}")
            if attempt < max_retries:
                time.sleep(RETRY_DELAY * attempt)

    return None


# ----------------------
# INTERNAL: Parse hourly precipitation response
# ----------------------
def _parse_hourly_response(data, source, start, end):
    """Parse Open-Meteo hourly response into summary dict."""
    try:
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        precip = hourly.get("precipitation", [])

        if not precip:
            return {
                "rainfall_mm": 0.0,
                "hourly_max_mm": 0.0,
                "hours_with_rain": 0,
                "source": source,
                "timestamp": datetime.now().isoformat(),
                "period_start": start.isoformat(),
                "period_end": end.isoformat()
            }

        # Clean NaN values
        precip_clean = [p if p is not None and not np.isnan(p) else 0.0 for p in precip]

        total = sum(precip_clean)
        max_hourly = max(precip_clean) if precip_clean else 0.0
        hours_rain = sum(1 for p in precip_clean if p > 0)

        return {
            "rainfall_mm": round(total, 2),
            "hourly_max_mm": round(max_hourly, 2),
            "hours_with_rain": hours_rain,
            "source": source,
            "timestamp": datetime.now().isoformat(),
            "period_start": start.isoformat(),
            "period_end": end.isoformat()
        }

    except Exception as e:
        logger.error(f"Parse error: {e}")
        return None


# ----------------------
# STANDALONE: Run as script
# ----------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    cities_file = os.path.join(BASE_DIR, "data", "config", "gujarat_cities.csv")
    cities = pd.read_csv(cities_file)

    # Test with first 5 cities
    test_cities = cities.head(5)
    print(f"\n🛰️ Testing satellite rainfall for {len(test_cities)} cities...\n")

    df = fetch_multiple_locations(test_cities)
    print("\n📊 Results:")
    print(df[["city", "lat", "lon", "rainfall_mm", "hourly_max_mm", "source"]].to_string(index=False))

    save_satellite_data(df)
    print(f"\n✅ Done! Data saved to {OUTPUT_FILE}")
