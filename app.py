import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from src.utils.features import feature_engineering
from src.utils.realtime_data import get_all_realtime, get_pipeline_status

st.set_page_config(page_title="RAINWISE - Flood & Rainfall Prediction", layout="centered")

# ----------------------
# SIDEBAR
# ----------------------
st.sidebar.title("🌊 RAINWISE")
st.sidebar.markdown("**AI Prediction System**")
page = st.sidebar.radio(
    "Select Model",
    ["🌧 Rainfall Prediction", "🌊 Flood Prediction"],
    index=0
)

st.sidebar.divider()

# Pipeline status in sidebar
pipeline_status = get_pipeline_status()
st.sidebar.subheader("📡 Pipeline Status")
if pipeline_status["weather_log_exists"]:
    st.sidebar.success("Weather ✅")
else:
    st.sidebar.error("Weather ❌")
if pipeline_status["rainfall_log_exists"]:
    st.sidebar.success("Rainfall ✅")
else:
    st.sidebar.error("Rainfall ❌")
if pipeline_status["river_log_exists"]:
    st.sidebar.success("River ✅")
else:
    st.sidebar.error("River ❌")

if pipeline_status["last_run"]:
    st.sidebar.caption(f"Last run: {pipeline_status['last_run']}")

st.sidebar.divider()
st.sidebar.markdown("""
```
☁️ Atmosphere / Pipeline
    ↓
🌧 Rainfall Model
    ↓
🌊 Flood Model
    ↓
🚨 Alert System
```
""")

# ----------------------
# SHARED DATA
# ----------------------
cities_df = pd.read_csv("data/config/gujarat_cities.csv")
cities_df.columns = cities_df.columns.str.lower()

river_df = pd.read_csv("data/processed/gujarat_river_distance.csv")
elev_df = pd.read_csv("data/processed/gujarat_elevation.csv")
river_df.columns = river_df.columns.str.lower()
elev_df.columns = elev_df.columns.str.lower()

def find_nearest(df, lat, lon):
    df = df.copy()
    df["dist"] = (df["lat"] - lat)**2 + (df["lon"] - lon)**2
    return df.loc[df["dist"].idxmin()]


# ================================================================
# PAGE 1: RAINFALL PREDICTION
# ================================================================
if page == "🌧 Rainfall Prediction":
    st.title("🌧 Rainfall Prediction")
    st.markdown("Predict rainfall from **atmospheric conditions** or **real-time pipeline data**.")

    rainfall_model = joblib.load("models/rainfall_model.pkl")

    city = st.selectbox("📍 Select City", cities_df["city"].unique(), key="rain_city")
    row = cities_df[cities_df["city"] == city]
    lat = float(row["lat"].values[0])
    lon = float(row["lon"].values[0])

    # --- Pipeline Data ---
    realtime = get_all_realtime(city)

    input_mode = st.radio("📥 Input Source",
                          ["✏️ Manual Input", "📡 Pipeline Data"],
                          horizontal=True, key="rain_source")

    if input_mode == "📡 Pipeline Data" and realtime["has_weather"]:
        w = realtime["weather"]
        temperature = w["temperature"]
        humidity = w["humidity"]
        pressure = w["pressure"]
        wind_speed = w["wind_speed"]
        cloud_cover = w["cloud_cover"]

        st.success("✅ Using pipeline weather data!")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("🌡 Temp", f"{temperature}°C")
        col2.metric("💧 Humid", f"{humidity}%")
        col3.metric("📊 Press", f"{pressure} hPa")
        col4.metric("💨 Wind", f"{wind_speed} km/h")
        col5.metric("☁️ Clouds", f"{cloud_cover}%")
        st.caption(f"📡 From pipeline at {w['timestamp']}")
        data_source = "Pipeline"

    elif input_mode == "📡 Pipeline Data" and not realtime["has_weather"]:
        st.warning("⚠️ No pipeline weather data for this city. Using manual input.")
        input_mode = "✏️ Manual Input"
        data_source = "Manual"

    if input_mode == "✏️ Manual Input":
        col_a, col_b = st.columns(2)
        temperature = col_a.slider("🌡️ Temperature (°C)", 15.0, 48.0, 32.0, step=0.5, key="rain_temp")
        humidity = col_b.slider("💧 Humidity (%)", 20.0, 100.0, 55.0, step=1.0, key="rain_hum")
        col_c, col_d = st.columns(2)
        pressure = col_c.slider("📊 Pressure (hPa)", 985.0, 1035.0, 1012.0, step=0.5, key="rain_pres")
        wind_speed = col_d.slider("💨 Wind Speed (km/h)", 0.0, 80.0, 12.0, step=1.0, key="rain_wind")
        cloud_cover = st.slider("☁️ Cloud Cover (%)", 0.0, 100.0, 30.0, step=1.0, key="rain_cloud")
        data_source = "Manual"

    # --- River Info ---
    if realtime["has_river"]:
        rv = realtime["river"]
        st.divider()
        col_rv1, col_rv2, col_rv3 = st.columns(3)
        col_rv1.metric(f"🏞 {rv['river']} River", f"{rv['level']:.1f} m")
        col_rv2.metric("⚠️ Danger Level", f"{rv['danger']} m")
        col_rv3.metric("Status", rv["status"])

    if st.button("🔍 Predict Rainfall", key="rain_btn"):
        atmos = np.array([[temperature, humidity, pressure, wind_speed, cloud_cover]])
        predicted_rain = max(0.0, float(rainfall_model.predict(atmos)[0]))

        st.divider()
        st.subheader("📊 Predicted Rainfall")
        col_r1, col_r2 = st.columns(2)
        col_r1.metric("Predicted Rainfall", f"{predicted_rain:.2f} mm")

        if predicted_rain > 50:
            col_r2.metric("Intensity", "🔴 Very Heavy")
            st.error(f"🚨 Very heavy: {predicted_rain:.1f} mm — Flash flood risk!")
            st.toast("🌧️ Heavy rainfall alert!", icon="🚨")
        elif predicted_rain > 20:
            col_r2.metric("Intensity", "🟠 Heavy")
            st.warning(f"⚠️ Heavy: {predicted_rain:.1f} mm")
        elif predicted_rain > 5:
            col_r2.metric("Intensity", "🟡 Moderate")
            st.info(f"🌦 Moderate: {predicted_rain:.1f} mm")
        else:
            col_r2.metric("Intensity", "🟢 Light / None")
            st.success(f"☀️ Light: {predicted_rain:.1f} mm")

        summary_df = pd.DataFrame({
            "Parameter": ["City", "Temperature", "Humidity", "Pressure", "Wind", "Clouds",
                          "Predicted Rain", "Source"],
            "Value": [city, f"{temperature}°C", f"{humidity}%", f"{pressure} hPa",
                      f"{wind_speed} km/h", f"{cloud_cover}%", f"{predicted_rain:.2f} mm", data_source]
        })
        st.table(summary_df)
        st.download_button("📥 Download", summary_df.to_csv(index=False),
                           f"rainfall_{city}.csv", "text/csv")

    st.divider()
    st.subheader("📊 Model Evaluation")
    st.image("outputs/rainfall_actual_vs_predicted.png", caption="Actual vs Predicted (R²=0.917)")
    st.image("outputs/rainfall_feature_importance.png", caption="Feature Importance")

    st.subheader("🗺 Map")
    st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))


# ================================================================
# PAGE 2: FLOOD PREDICTION
# ================================================================
elif page == "🌊 Flood Prediction":
    st.title("🌊 Flood Prediction")
    st.markdown("Predict flood risk from **rainfall & geography** or **real-time pipeline data**.")

    flood_model = joblib.load("models/flood_model.pkl")

    city = st.selectbox("📍 Select City", cities_df["city"].unique(), key="flood_city")
    row = cities_df[cities_df["city"] == city]
    lat = float(row["lat"].values[0])
    lon = float(row["lon"].values[0])

    distance = float(find_nearest(river_df, lat, lon)["river_distance"])
    elevation = float(find_nearest(elev_df, lat, lon)["elevation"])

    st.subheader("🌍 Location & Geography")
    col1, col2 = st.columns(2)
    col1.metric("Latitude", f"{lat:.4f}")
    col2.metric("Longitude", f"{lon:.4f}")
    col3, col4 = st.columns(2)
    col3.metric("Distance to River", f"{distance:.0f} m")
    col4.metric("Elevation", f"{elevation:.0f} m")

    # --- Pipeline Data ---
    realtime = get_all_realtime(city)

    input_mode = st.radio("📥 Input Source",
                          ["✏️ Manual Input", "📡 Pipeline Data"],
                          horizontal=True, key="flood_source")

    if input_mode == "📡 Pipeline Data" and realtime["has_rainfall"]:
        rt_rain = realtime["rainfall"]["precipitation_mm"]
        rain = st.slider("Rainfall (mm)", 0.0, 100.0, min(float(rt_rain), 100.0), step=0.5, key="flood_rain")
        st.caption(f"📡 Pipeline value: {rt_rain} mm")
        data_source = "Pipeline"
    else:
        if input_mode == "📡 Pipeline Data":
            st.warning("⚠️ No pipeline rainfall data. Using manual input.")
        rain = st.slider("Rainfall (mm)", 0.0, 100.0, 10.0, step=0.5, key="flood_rain")
        data_source = "Manual"

    # --- River Info ---
    if realtime["has_river"]:
        rv = realtime["river"]
        st.divider()
        col_rv1, col_rv2, col_rv3 = st.columns(3)
        col_rv1.metric(f"🏞 {rv['river']} River", f"{rv['level']:.1f} m")
        col_rv2.metric("⚠️ Danger Level", f"{rv['danger']} m")
        col_rv3.metric("Status", rv["status"])

        if rv["status"] == "Above Danger":
            st.error(f"🚨 {rv['river']} River is ABOVE DANGER level!")
        elif rv["status"] == "Warning":
            st.warning(f"⚠️ {rv['river']} River at WARNING level!")

    threshold = st.slider("🎯 Alert Threshold", 0.1, 0.9, 0.5, step=0.05, key="flood_thresh")

    if st.button("🔍 Predict Flood Risk", key="flood_btn"):
        features = np.array([[rain, elevation, distance, lat, lon]])
        proba = flood_model.predict_proba(features)[0][1]

        st.divider()
        st.subheader("📊 Flood Risk Assessment")
        col_f1, col_f2 = st.columns(2)
        col_f1.metric("Flood Probability", f"{proba:.2f}")

        if proba > threshold:
            if proba > 0.8:
                col_f2.metric("Risk Level", "🔴 HIGH")
                st.error("🚨 HIGH FLOOD RISK — Evacuate!")
            elif proba > 0.6:
                col_f2.metric("Risk Level", "🟠 SIGNIFICANT")
                st.error("🔴 SIGNIFICANT RISK!")
            else:
                col_f2.metric("Risk Level", "🟡 MODERATE")
                st.warning("⚠️ MODERATE RISK!")
        else:
            col_f2.metric("Risk Level", "🟢 LOW")
            st.success("✅ LOW RISK — Safe.")

        # Alert
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        river_info = f" | 🏞 {rv['river']}: {rv['status']}" if realtime["has_river"] else ""

        if proba > 0.6:
            alert_level = "🔴 CRITICAL" if proba > 0.8 else "🟠 WARNING"
            st.error(f"""
**{alert_level} FLOOD ALERT — {city}**
📅 {timestamp} | 🌧️ {rain}mm | 🌊 {proba:.2f}{river_info}
📡 Source: {data_source}
📱 SMS → District Collector, NDRF, Municipal Corp
""")
            st.toast(f"🚨 ALERT: {city}!", icon="🚨")

        summary = pd.DataFrame({
            "Parameter": ["City", "Rainfall", "Elevation", "River Distance",
                          "Flood Probability", "Source", "River Status"],
            "Value": [city, f"{rain} mm", f"{elevation:.0f} m", f"{distance:.0f} m",
                      f"{proba:.2f}", data_source,
                      rv["status"] if realtime["has_river"] else "N/A"]
        })
        st.table(summary)
        st.download_button("📥 Download", summary.to_csv(index=False),
                           f"flood_{city}.csv", "text/csv")

    st.divider()
    st.subheader("📊 Model Evaluation")
    st.image("outputs/flood_confusion_matrix.png", caption="Confusion Matrix")
    st.image("outputs/flood_feature_importance.png", caption="Feature Importance")

    st.subheader("🗺 Map")
    st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))