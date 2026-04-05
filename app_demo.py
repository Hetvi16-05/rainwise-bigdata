import streamlit as st
import numpy as np
import joblib
import pandas as pd

st.set_page_config(page_title="Flood Prediction Demo", layout="centered")

st.title("🌊 Flood Prediction System (Demo Mode)")
st.markdown("Simulated weather + real GIS-based river distance & elevation")

# ----------------------
# LOAD MODEL
# ----------------------
model = joblib.load("models/flood_model_xgb.pkl")
threshold = joblib.load("models/threshold.pkl")

# ----------------------
# LOAD CITY DATA (WITH LAT/LON)
# ----------------------
cities_df = pd.read_csv("data/config/gujarat_cities.csv")
cities_df.columns = cities_df.columns.str.lower()

city_list = cities_df["city"].dropna().unique().tolist()

st.subheader("📍 Select Location")
city = st.selectbox("Select City", city_list)

# ----------------------
# GET CITY LAT/LON
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

distance_raw = float(nearest_river["river_distance"])
elevation = float(nearest_elev["elevation"])

# ----------------------
# NORMALIZE DISTANCE (IMPORTANT FIX)
# ----------------------
# Convert large meter values into usable ML scale
distance = distance_raw / 10   # <-- KEY FIX

# ----------------------
# SHOW LOCATION INFO
# ----------------------
st.subheader("🌍 Location Risk Factors")
st.write(f"📍 Lat: {city_lat}, Lon: {city_lon}")
st.write(f"📏 Distance to River (raw): {distance_raw:.2f} m")
st.write(f"⚙️ Normalized Distance: {distance:.2f}")
st.write(f"⛰ Elevation: {elevation:.2f} m")

# ----------------------
# DEMO WEATHER INPUT
# ----------------------
st.subheader("🧪 Simulated Weather")

rain = st.slider("Rain (last 1 hour mm)", 0.0, 100.0, 10.0)
clouds = st.slider("Cloud Cover (%)", 0.0, 100.0, 50.0)
humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0)

# ----------------------
# OPTIONAL BOOST MODE (FOR DEMO)
# ----------------------
boost = st.checkbox("⚡ Flood Sensitivity Mode", value=True)

if boost:
    distance = distance / 2   # make model more sensitive

# ----------------------
# PREDICTION
# ----------------------
if st.button("🔍 Predict Flood Risk"):

    # simulate multi-day rain
    rain3 = rain * 3 + (clouds / 100) * 5
    rain7 = rain * 7 + (humidity / 100) * 10

    # ----------------------
    # FEATURE ENGINEERING
    # ----------------------
    rain_trend = rain3 - rain7
    rain_intensity = rain3 / 3

    log_rain3 = np.log1p(rain3)
    log_rain7 = np.log1p(rain7)

    # IMPORTANT FIX: scaled river effect
    river_risk = rain3 / (distance + 1)

    features = np.array([[
        rain3,
        rain7,
        rain_trend,
        rain_intensity,
        log_rain3,
        log_rain7,
        river_risk,
        elevation,
        distance
    ]])

    # ----------------------
    # MODEL PREDICTION
    # ----------------------
    proba = model.predict_proba(features)[0][1]
    pred = int(proba > threshold)

    # ----------------------
    # OUTPUT
    # ----------------------
    st.subheader("📊 Result")

    st.write(f"📍 City: {city}")
    st.write(f"🌧 Rain (1h): {rain} mm")
    st.write(f"☁ Clouds: {clouds}%")
    st.write(f"💧 Humidity: {humidity}%")

    st.write(f"Flood Probability: **{proba:.2f}**")

    if pred == 1:
        if proba > 0.85:
            st.error("🚨 HIGH FLOOD RISK")
        elif proba > 0.6:
            st.warning("⚠️ MODERATE FLOOD RISK")
        else:
            st.info("⚠️ LOW FLOOD RISK")
    else:
        st.success("✅ SAFE")