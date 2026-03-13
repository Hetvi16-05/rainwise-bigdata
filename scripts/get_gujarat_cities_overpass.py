import requests
import pandas as pd
import os

OUTPUT = "data/config/gujarat_cities.csv"


def fetch_cities():

    url = "https://overpass-api.de/api/interpreter"

    query = """
    [out:json];
    area["name"="Gujarat"]->.searchArea;

    (
      node["place"~"city|town"](area.searchArea);
    );

    out body;
    """

    response = requests.post(url, data=query)

    data = response.json()

    cities = []

    for el in data["elements"]:

        name = el["tags"].get("name")

        lat = el.get("lat")
        lon = el.get("lon")

        if name and lat and lon:

            cities.append({
                "city": name,
                "lat": lat,
                "lon": lon
            })

    return cities


def main():

    cities = fetch_cities()

    df = pd.DataFrame(cities)

    os.makedirs("data/config", exist_ok=True)

    df.to_csv(OUTPUT, index=False)

    print("✅ Cities saved:", len(df))


if __name__ == "__main__":
    main()