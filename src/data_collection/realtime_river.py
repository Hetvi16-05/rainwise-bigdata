import pandas as pd
import datetime
import requests
import os
import time
import json
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


RIVER_CACHE_FILE = os.path.join(BASE_DIR, "data/config/river_name_cache.json")

def get_river_name_from_osm(lat, lon, city):
    """
    Use Overpass API to find the nearest river name.
    """
    # Load cache
    cache = {}
    if os.path.exists(RIVER_CACHE_FILE):
        try:
            with open(RIVER_CACHE_FILE, "r") as f:
                cache = json.load(f)
        except:
            pass
            
    if city in cache:
        return cache[city]

    try:
        # Rate limiting: wait 1.5 seconds before calling OSM (be more polite)
        time.sleep(1.5)
        
        # Try multiple Overpass mirrors
        overpass_servers = [
            "https://overpass.kumi.systems/api/interpreter",
            "https://overpass-api.de/api/interpreter",
            "https://lz4.overpass-api.de/api/interpreter"
        ]
        
        query = f"""
        [out:json][timeout:15];
        (
          way(around:5000, {lat}, {lon})[waterway~"river|stream|canal"];
        );
        out tags;
        """
        
        for server in overpass_servers:
            try:
                response = requests.post(server, data={'data': query}, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    elements = data.get('elements', [])
                    if elements:
                        # Try to find a named river
                        for el in elements:
                            name = el.get('tags', {}).get('name')
                            if name:
                                cache[city] = name
                                os.makedirs(os.path.dirname(RIVER_CACHE_FILE), exist_ok=True)
                                with open(RIVER_CACHE_FILE, "w") as f:
                                    json.dump(cache, f)
                                return name
                        
                        # If no name, use the type
                        w_type = elements[0].get('tags', {}).get('waterway', 'waterway').capitalize()
                        return f"Local {w_type}"
                elif response.status_code == 429:
                    print(f"Server {server} rate limited, trying next...")
                    continue
            except Exception as inner_e:
                print(f"Server {server} failed: {inner_e}")
                continue
            
    except Exception as e:
        print(f"OSM Total Error for {city}: {e}")
        
    return "Nearby River"

def find_nearest_river(lat, lon, river_df, city):
    """
    Find nearest river from local DB, or use OSM if distance is too high.
    """
    if river_df.empty:
        return {"river": get_river_name_from_osm(lat, lon, city), "station": "Global Station"}

    river_df["distance"] = river_df.apply(
        lambda r: geodesic((lat, lon), (r["lat"], r["lon"])).km,
        axis=1
    )
    
    nearest = river_df.loc[river_df["distance"].idxmin()]
    
    # If the nearest station is > 30km away, it's probably not accurate for this city.
    # Use OSM to get the local river name instead.
    if nearest["distance"] > 30:
        local_name = get_river_name_from_osm(lat, lon, city)
        return {"river": local_name, "station": "Auto-Calculated"}
        
    return nearest


def fetch_river_discharge(lat, lon):
    """Fetch river discharge from Open-Meteo Flood API."""
    try:
        url = "https://flood-api.open-meteo.com/v1/flood"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "river_discharge",
            "forecast_days": 1,
            "timezone": "Asia/Kolkata"
        }
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if "daily" in data and "river_discharge" in data["daily"]:
            return float(data["daily"]["river_discharge"][0] or 0.0)
    except Exception as e:
        print(f"Error fetching discharge for {lat},{lon}: {e}")
    return 0.0


def get_status(discharge):
    """
    Get status based on discharge (m³/s).
    Note: Using generic thresholds since the database is level-based (m).
    """
    if discharge > 50:
        return "Above Danger"
    elif discharge > 15:
        return "Warning"
    return "Normal"


def main():

    cities = pd.read_csv(CITIES_FILE)
    river_df = safe_read_csv(RIVER_DB)

    rows = []

    for _, row in cities.iterrows():
        # Pass city name for OSM caching
        nearest = find_nearest_river(row["lat"], row["lon"], river_df, row["city"])

        discharge = fetch_river_discharge(row["lat"], row["lon"])

        rows.append({
            "timestamp": datetime.datetime.now(),
            "city": row["city"],
            "lat": row["lat"],
            "lon": row["lon"],
            "river": nearest["river"],
            "station": nearest["station"],
            "level": discharge,  # Storing discharge in the level column for compatibility
            "danger": 50.0,      # Discharge-based danger threshold
            "warning": 15.0,     # Discharge-based warning threshold
            "status": get_status(discharge)
        })

    new_df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    old_df = safe_read_csv(OUTPUT_FILE)

    df = pd.concat([old_df, new_df], ignore_index=True)

    safe_write_csv(df, OUTPUT_FILE)

    print("✅ River updated:", len(new_df))


if __name__ == "__main__":
    main()