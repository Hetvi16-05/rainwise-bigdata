import json
import pandas as pd

# Load GeoJSON
with open("data/raw/rivers/metadata/cwc_station_master.geojson") as f:
    data = json.load(f)

rows = []

for feature in data["features"]:
    props = feature["properties"]
    coords = feature["geometry"]["coordinates"]

    rows.append({
        "station_id": props.get("station_id") or props.get("code") or props.get("id"),
        "station_name": props.get("name"),
        "river": props.get("river"),
        "state": props.get("state"),
        "agency": props.get("agency"),
        "lat": coords[1],
        "lon": coords[0]
    })

df = pd.DataFrame(rows)

df.to_csv("data/raw/static/rivers/metadata/cwc_station_master.csv", index=False)

print("Total stations:", len(df))
print(df.head())