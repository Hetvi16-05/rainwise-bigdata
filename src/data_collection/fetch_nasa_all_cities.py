import requests
import pandas as pd
import time
import os

print("📥 Loading cities...")

cities = pd.read_csv("data/config/gujarat_cities.csv")

output_file = "data/processed/nasa_rainfall_gujarat.csv"

# -----------------------------
# Load existing data (if exists)
# -----------------------------
if os.path.exists(output_file):
    print("📂 Existing dataset found, loading...")
    existing_df = pd.read_csv(output_file)
else:
    print("🆕 No existing dataset found")
    existing_df = pd.DataFrame()

# ✅ FIX: Correct date parsing
if not existing_df.empty:
    existing_df['date'] = pd.to_datetime(existing_df['date'].astype(str), format='%Y%m%d')

# -----------------------------
# Fetch function
# -----------------------------
def fetch_rainfall(lat, lon, start, end):
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"

    params = {
        "parameters": "PRECTOTCORR",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start,
        "end": end,
        "format": "JSON"
    }

    try:
        response = requests.get(url, params=params, timeout=30)

        if response.status_code != 200:
            print(f"❌ API error {response.status_code}")
            return pd.DataFrame()

        data = response.json()

        rainfall = data.get('properties', {}).get('parameter', {}).get('PRECTOTCORR')

        if rainfall is None:
            print(f"⚠️ No data for ({lat}, {lon})")
            return pd.DataFrame()

        df = pd.DataFrame(list(rainfall.items()), columns=['date', 'rainfall'])

        return df

    except Exception as e:
        print(f"❌ Error for ({lat}, {lon}): {e}")
        return pd.DataFrame()


# -----------------------------
# MAIN LOOP
# -----------------------------
all_data = []

print("🌧️ Fetching OLD data (2000–2014)...\n")

for i, row in cities.iterrows():
    city = row['city']
    lat = row['lat']
    lon = row['lon']

    print(f"Processing {city} ({i+1}/{len(cities)})")

    # ✅ ALWAYS fetch old data (no skipping)
    df = fetch_rainfall(lat, lon, "20000101", "20141231")

    if not df.empty:
        df['city'] = city
        df['lat'] = lat
        df['lon'] = lon
        all_data.append(df)

    time.sleep(0.3)  # prevent API blocking


# -----------------------------
# COMBINE + CLEAN
# -----------------------------
if len(all_data) > 0:
    print("\n🔗 Combining datasets...")

    new_df = pd.concat(all_data, ignore_index=True)

    # Combine old + existing
    final_df = pd.concat([existing_df, new_df], ignore_index=True)

    # ✅ Strong duplicate removal
    final_df = final_df.drop_duplicates(subset=['date', 'city', 'lat', 'lon'])

    print("💾 Saving updated dataset...")

    final_df.to_csv(output_file, index=False)

    print("✅ DONE! Dataset updated successfully.")

else:
    print("⚠️ No new data fetched.")