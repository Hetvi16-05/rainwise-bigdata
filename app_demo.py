import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Flood Prediction Demo", layout="centered")

st.title("🌊 Flood Prediction System (Pure ML)")
st.markdown("Machine Learning + GIS Based Flood Risk Prediction")

# ----------------------
# LOAD MODEL
# ----------------------
model = joblib.load("models/final_flood_model.pkl")

# ----------------------
# LOAD CITY DATA
# ----------------------
cities_df = pd.read_csv("data/config/gujarat_cities.csv")
cities_df.columns = cities_df.columns.str.lower()

city_list = cities_df["city"].dropna().unique().tolist()

st.subheader("📍 Select Location")
city = st.selectbox("Select City", city_list)

# ----------------------
# GET LAT/LON
# ----------------------
city_row = cities_df[cities_df["city"] == city]

if not city_row.empty:
    city_lat = float(city_row["lat"].values[0])
    city_lon = float(city_row["lon"].values[0])
else:
    city_lat, city_lon = 0, 0

# ----------------------
# LOAD GIS DATA
# ----------------------
river_df = pd.read_csv("data/processed/gujarat_river_distance.csv")
elev_df = pd.read_csv("data/processed/gujarat_elevation.csv")

river_df.columns = river_df.columns.str.lower()
elev_df.columns = elev_df.columns.str.lower()

# ----------------------
# FIND NEAREST POINT
# ----------------------
def find_nearest(df, lat, lon):
    df["dist"] = (df["lat"] - lat)**2 + (df["lon"] - lon)**2
    return df.loc[df["dist"].idxmin()]

nearest_river = find_nearest(river_df, city_lat, city_lon)
nearest_elev = find_nearest(elev_df, city_lat, city_lon)

distance = float(nearest_river["river_distance"])
elevation = float(nearest_elev["elevation"])

# ----------------------
# SHOW LOCATION
# ----------------------
st.subheader("🌍 Location Risk Factors")
st.write(f"📍 Lat: {city_lat}, Lon: {city_lon}")
st.write(f"📏 Distance to River: {distance:.2f} m")
st.write(f"⛰ Elevation: {elevation:.2f} m")

# ----------------------
# WEATHER INPUT
# ----------------------
st.subheader("🧪 Simulated Weather")
rain = st.slider("Rain (last 1 hour mm)", 0.0, 100.0, 10.0)

# ----------------------
# PREDICTION
# ----------------------
if st.button("🔍 Predict Flood Risk"):

    # Multi-day rain simulation
    rain3 = rain * 3
    rain7 = rain * 7

    # Feature engineering (same as training)
    rain_intensity = rain7 / 7
    rain_ratio = rain3 / (rain7 + 1)

    # 🔥 NEW FEATURES (must match training)
    heavy_rain_flag = int(rain3 > 150)
    extreme_rain_flag = int(rain7 > 300)
    river_risk = 1 / (distance + 1)

    # NASA simulation
    nasa_avg = rain * 0.8
    nasa_max = rain * 1.2
    nasa_std = rain * 0.3

    # ----------------------
    # CREATE DATAFRAME (MATCH TRAINING EXACTLY)
    # ----------------------
    feature_names = [
        "rain3_mm",
        "rain7_mm",
        "rain_intensity",
        "rain_ratio",
        "heavy_rain_flag",
        "extreme_rain_flag",
        "river_risk",
        "elevation_m",
        "distance_to_river_m",
        "lat",
        "lon",
        "nasa_avg_rain",
        "nasa_max_rain",
        "nasa_std_rain"
    ]

    features = pd.DataFrame([[
        rain3,
        rain7,
        rain_intensity,
        rain_ratio,
        heavy_rain_flag,
        extreme_rain_flag,
        river_risk,
        elevation,
        distance,
        city_lat,
        city_lon,
        nasa_avg,
        nasa_max,
        nasa_std
    ]], columns=feature_names)

    # ----------------------
    # PURE ML PREDICTION
    # ----------------------
    proba = model.predict_proba(features)[0][1]

    # ----------------------
    # OUTPUT
    # ----------------------
    st.subheader("📊 Result")

    st.write(f"📍 City: {city}")
    st.write(f"🌧 Rain (1h): {rain:.2f} mm")
    st.write(f"Flood Probability: **{proba:.2f}**")

    # Thresholds
    if proba > 0.6:
        st.error("🚨 HIGH FLOOD RISK")
    elif proba > 0.3:
        st.warning("⚠️ MODERATE FLOOD RISK")
    else:
        st.success("✅ LOW RISK")

    # ----------------------
    # MAP
    # ----------------------
    st.subheader("🗺 Location Map")
    st.map(pd.DataFrame({
        "lat": [city_lat],
        "lon": [city_lon]
    }))