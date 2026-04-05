import streamlit as st
import numpy as np
import joblib
import requests
import pandas as pd

st.set_page_config(page_title="Flood Prediction", layout="centered")

st.title("🌊 Real-Time Flood Prediction System")
st.markdown("AI-based flood risk detection for Gujarat")

# ----------------------
# LOAD MODEL
# ----------------------
model = joblib.load("models/flood_model_xgb.pkl")
threshold = joblib.load("models/threshold.pkl")

# ----------------------
# LOAD CITY LIST
# ----------------------
cities_df = pd.read_csv("data/config/gujarat_city_names.csv")
city_list = cities_df["city"].tolist()

st.subheader("📍 Select Location")
city = st.selectbox("Select City", city_list)

# ----------------------
# API CONFIG
# ----------------------
API_KEY = "a20148ccfa37a665bc1993dcfbf42197"

# ----------------------
# FETCH WEATHER DATA
# ----------------------
def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url).json()

    rain = response.get("rain", {}).get("1h", 0)
    clouds = response.get("clouds", {}).get("all", 0)
    humidity = response.get("main", {}).get("humidity", 0)

    return rain, clouds, humidity

# ----------------------
# PREDICTION
# ----------------------
if st.button("🌦️ Get Live Prediction"):

    rain, clouds, humidity = get_weather(city)

    # ----------------------
    # BETTER RAIN ESTIMATION
    # ----------------------
    rain3 = rain * 3 + (clouds / 100) * 5
    rain7 = rain * 7 + (humidity / 100) * 10

    # default geo values
    elevation = 50
    distance = 500

    # ----------------------
    # FEATURE ENGINEERING (MATCH TRAINING)
    # ----------------------
    rain_trend = rain3 - rain7
    rain_intensity = rain3 / 3

    log_rain3 = np.log1p(rain3)
    log_rain7 = np.log1p(rain7)

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