import pandas as pd
import requests

cities_path = "data/config/gujarat_cities.csv"
output_path = "data/processed/gujarat_rainfall_history.csv"

start = "20000101"
end = "20251231"

cities = pd.read_csv(cities_path)

rows = []

for _, row in cities.iterrows():

    city = row["city"]
    lat = row["lat"]
    lon = row["lon"]

    print("Fetching:", city)

    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=PRECTOTCORR"
        f"&community=AG"
        f"&longitude={lon}"
        f"&latitude={lat}"
        f"&start={start}"
        f"&end={end}"
        "&format=JSON"
    )

    r = requests.get(url).json()

    try:
        data = r["properties"]["parameter"]["PRECTOTCORR"]
    except:
        print("No data:", city)
        continue

    for date, rain in data.items():

        rows.append({
            "date": date,
            "city": city,
            "lat": lat,
            "lon": lon,
            "rain_mm": rain
        })


df = pd.DataFrame(rows)

df.to_csv(output_path, index=False)

print("Saved:", output_path)