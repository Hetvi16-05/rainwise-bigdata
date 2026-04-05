import pandas as pd
import requests
import time

input_path = "data/config/gujarat_city_names.csv"
output_path = "data/config/gujarat_cities.csv"

df = pd.read_csv(input_path)

rows = []

for city in df["city"]:

    query = f"{city}, Gujarat, India"

    url = "https://nominatim.openstreetmap.org/search"

    params = {
        "q": query,
        "format": "json",
        "limit": 1
    }

    print("Fetching:", city)

    r = requests.get(url, params=params, headers={"User-Agent": "rainwise-project"}).json()

    if len(r) == 0:
        print("Not found:", city)
        continue

    lat = r[0]["lat"]
    lon = r[0]["lon"]

    rows.append({
        "city": city,
        "lat": lat,
        "lon": lon
    })

    time.sleep(1)   # important (avoid block)

out = pd.DataFrame(rows)

out.to_csv(output_path, index=False)

print("Saved:", output_path)