import requests


def get_location():
    try:
        r = requests.get("http://ip-api.com/json/")
        d = r.json()

        lat = d.get("lat")
        lon = d.get("lon")
        city = d.get("city")
        state = d.get("regionName")

        return lat, lon, city, state

    except:
        return None, None, None, None


def get_weather(lat, lon):

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}"
        f"&longitude={lon}"
        "&current=temperature_2m,precipitation,relative_humidity_2m,wind_speed_10m"
    )

    r = requests.get(url)

    return r.json()


def main():

    lat, lon, city, state = get_location()

    print("Location:")
    print("City:", city)
    print("State:", state)
    print("Lat:", lat)
    print("Lon:", lon)

    if lat is None:
        print("Location failed")
        return

    data = get_weather(lat, lon)

    current = data.get("current", {})

    print("\nWeather:")
    print("Temp:", current.get("temperature_2m"))
    print("Rain:", current.get("precipitation"))
    print("Humidity:", current.get("relative_humidity_2m"))
    print("Wind:", current.get("wind_speed_10m"))


if __name__ == "__main__":
    main()
