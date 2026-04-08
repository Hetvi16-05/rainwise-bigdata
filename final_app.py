import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime

from src.utils.features import feature_engineering
from src.utils.realtime_data import get_all_realtime, get_pipeline_status

st.set_page_config(
    page_title="RAINWISE — Real-Time Flood Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------
# SESSION STATE
# ----------------------
if "alert_history" not in st.session_state:
    st.session_state.alert_history = []

# ----------------------
# LOAD MODELS (cached)
# ----------------------
@st.cache_resource
def load_models():
    return joblib.load("models/flood_model.pkl"), joblib.load("models/rainfall_model.pkl")

@st.cache_data
def load_city_data():
    df = pd.read_csv("data/config/gujarat_cities.csv")
    df.columns = df.columns.str.lower()
    return df

@st.cache_data
def load_gis_data():
    r = pd.read_csv("data/processed/gujarat_river_distance.csv")
    e = pd.read_csv("data/processed/gujarat_elevation.csv")
    r.columns = r.columns.str.lower()
    e.columns = e.columns.str.lower()
    return r, e

flood_model, rainfall_model = load_models()
cities_df = load_city_data()
river_df, elev_df = load_gis_data()

def find_nearest(df, lat, lon):
    df = df.copy()
    df["dist"] = (df["lat"] - lat)**2 + (df["lon"] - lon)**2
    return df.loc[df["dist"].idxmin()]

# ----------------------
# WEATHER API (fallback)
# ----------------------
def fetch_weather_api(lat, lon):
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,cloud_cover,precipitation",
            "timezone": "Asia/Kolkata"
        }, timeout=10)
        data = r.json()
        if "error" in data and data["error"]:
            return None, data.get("reason", "API error")
        c = data["current"]
        return {
            "temperature": c["temperature_2m"], "humidity": c["relative_humidity_2m"],
            "pressure": c["surface_pressure"], "wind_speed": c["wind_speed_10m"],
            "cloud_cover": c["cloud_cover"], "precipitation": c["precipitation"],
            "time": c["time"]
        }, None
    except Exception as e:
        return None, str(e)


# ================================================================
# HEADER
# ================================================================
st.title("🌊 RAINWISE — Real-Time Flood Intelligence")
st.markdown("Live data → AI Rainfall Prediction → Flood Risk → 🚨 Automated Alerts")

# ================================================================
# PIPELINE ARCHITECTURE
# ================================================================
with st.expander("🔄 Real-Time Pipeline Architecture", expanded=False):
    st.markdown("""
    ```
    📡 Data Collection Pipeline (run_realtime_pipeline.py)
    ├── realtime_weather.py   → Weather logs
    ├── gpm_fetcher.py (NEW)  → 🛰️ Satellite rainfall (ERA5)
    ├── realtime_rainfall.py  → Uses satellite data
    ├── realtime_river.py     → River level logs
    └── build_dataset.py      → Merged dataset + GIS features
         ↓
    ┌─────────────────────────────────────────────┐
    │  🛰️ Satellite Rainfall (ERA5/GPM)           │
    │  Real satellite-derived precipitation data   │
    ├─────────────────────────────────────────────┤
    │  🌧️ Stage 1: Rainfall Model (XGBoost)      │
    │  Atmospheric inputs → Predicted rainfall     │
    ├─────────────────────────────────────────────┤
    │  🌊 Stage 2: Flood Model (XGBoost)          │
    │  Rainfall + Geography → Flood probability    │
    ├─────────────────────────────────────────────┤
    │  🏞️ Stage 3: River Level Monitoring          │
    │  Live river levels vs danger thresholds       │
    ├─────────────────────────────────────────────┤
    │  🚨 Stage 4: Alert System                    │
    │  Auto-triggers SMS/Push if risk > threshold   │
    └─────────────────────────────────────────────┘
    ```
    """)

st.divider()

# ================================================================
# SIDEBAR
# ================================================================
st.sidebar.title("⚙️ Settings")

city = st.sidebar.selectbox("📍 Select City", cities_df["city"].unique())
row = cities_df[cities_df["city"] == city]
lat = float(row["lat"].values[0])
lon = float(row["lon"].values[0])

distance = float(find_nearest(river_df, lat, lon)["river_distance"])
elevation = float(find_nearest(elev_df, lat, lon)["elevation"])

threshold = st.sidebar.slider("🎯 Alert Threshold", 0.1, 0.9, 0.5, step=0.05)

st.sidebar.divider()

# Pipeline status
st.sidebar.subheader("📡 Pipeline Health")
p_status = get_pipeline_status()
st.sidebar.markdown(
    f"Weather: {'✅' if p_status['weather_log_exists'] else '❌'} | "
    f"Satellite: {'✅' if p_status.get('satellite_log_exists') else '❌'} | "
    f"Rainfall: {'✅' if p_status['rainfall_log_exists'] else '❌'} | "
    f"River: {'✅' if p_status['river_log_exists'] else '❌'}"
)
if p_status["last_run"]:
    st.sidebar.caption(f"Last pipeline run: {p_status['last_run']}")

st.sidebar.divider()
st.sidebar.subheader("🌍 Geography")
st.sidebar.metric("Elevation", f"{elevation:.0f} m")
st.sidebar.metric("River Distance", f"{distance:.0f} m")
st.sidebar.metric("Coordinates", f"{lat:.2f}, {lon:.2f}")

# ================================================================
# DATA SOURCE SELECTION
# ================================================================
st.subheader(f"📡 Real-Time Data for {city}")

# Try pipeline first, then API, then manual
realtime = get_all_realtime(city)

col_src1, col_src2, col_src3 = st.columns(3)

data_source = "Manual"

# --- Try Pipeline ---
if realtime["pipeline_active"] and realtime["has_weather"]:
    w = realtime["weather"]
    temperature = w["temperature"]
    humidity = w["humidity"]
    pressure = w["pressure"]
    wind_speed = w["wind_speed"]
    cloud_cover = w["cloud_cover"]
    actual_precipitation = w.get("precipitation", 0)
    weather_time = w["timestamp"]
    data_source = "Pipeline"
    st.success(f"✅ Using **Real-Time Pipeline** data (collected at {weather_time})")
else:
    # Fallback: try API
    api_data, api_error = fetch_weather_api(lat, lon)
    if api_data:
        temperature = api_data["temperature"]
        humidity = api_data["humidity"]
        pressure = api_data["pressure"]
        wind_speed = api_data["wind_speed"]
        cloud_cover = api_data["cloud_cover"]
        actual_precipitation = api_data["precipitation"]
        weather_time = api_data["time"]
        data_source = "Open-Meteo API"
        st.info(f"📡 Using **Open-Meteo API** (pipeline data not available)")
    else:
        # Final fallback: manual
        st.warning("⚠️ No pipeline or API data. Using manual input.")
        col_a, col_b, col_c = st.columns(3)
        temperature = col_a.number_input("🌡 Temp (°C)", value=32.0, step=0.5)
        humidity = col_b.number_input("💧 Humidity (%)", value=55.0, step=1.0)
        pressure = col_c.number_input("📊 Pressure (hPa)", value=1012.0, step=0.5)
        col_d, col_e = st.columns(2)
        wind_speed = col_d.number_input("💨 Wind (km/h)", value=12.0, step=1.0)
        cloud_cover = col_e.number_input("☁️ Clouds (%)", value=30.0, step=1.0)
        actual_precipitation = None
        weather_time = "Manual"
        data_source = "Manual"

# Display weather cards (if not manual)
if data_source != "Manual":
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🌡 Temp", f"{temperature}°C")
    c2.metric("💧 Humid", f"{humidity}%")
    c3.metric("📊 Press", f"{pressure} hPa")
    c4.metric("💨 Wind", f"{wind_speed} km/h")
    c5.metric("☁️ Clouds", f"{cloud_cover}%")

# ================================================================
# SATELLITE RAINFALL DATA
# ================================================================
if realtime.get("has_satellite") and realtime["satellite"]:
    sat = realtime["satellite"]
    st.divider()
    st.subheader("🛰️ Satellite Rainfall (ERA5)")

    sat_c1, sat_c2, sat_c3, sat_c4 = st.columns(4)
    sat_c1.metric("24h Total", f"{sat['rainfall_mm']:.1f} mm")
    sat_c2.metric("Max Hourly", f"{sat['hourly_max_mm']:.1f} mm")
    sat_c3.metric("Hours w/ Rain", f"{sat['hours_with_rain']}h")
    sat_c4.metric("Source", sat["source"])
    st.caption(f"📡 Satellite data from {sat['timestamp']}")

# ================================================================
# RIVER LEVEL MONITORING
# ================================================================
if realtime["has_river"]:
    rv = realtime["river"]
    st.divider()
    st.subheader(f"🏞️ River Discharge — {rv['river']} ({rv['station']})")

    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    col_r1.metric("Discharge", f"{rv['level']:.1f} m³/s")
    col_r2.metric("Warning Limit", f"{rv['warning']} m³/s")
    col_r3.metric("Danger Limit", f"{rv['danger']} m³/s")
    col_r4.metric("Status", rv["status"])

    if rv["status"] == "Above Danger":
        st.error(f"🚨 {rv['river']} River is ABOVE DANGER LEVEL! ({rv['level']:.1f} m³/s / {rv['danger']} m³/s)")
    elif rv["status"] == "Warning":
        st.warning(f"⚠️ {rv['river']} River at WARNING level ({rv['level']:.1f} m³/s / {rv['warning']} m³/s)")

# ================================================================
# AI PREDICTIONS
# ================================================================
st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🌧 Stage 1 — Rainfall")
    atmos = np.array([[temperature, humidity, pressure, wind_speed, cloud_cover]])
    predicted_rain = float(max(0.0, float(rainfall_model.predict(atmos)[0])))
    st.metric("Predicted Rainfall", f"{predicted_rain:.2f} mm")

    if actual_precipitation:
        st.metric("Actual (live)", f"{actual_precipitation} mm/h")

    if predicted_rain > 50:
        st.error(f"🚨 **Very Heavy** — {predicted_rain:.1f} mm")
    elif predicted_rain > 20:
        st.warning(f"⚠️ **Heavy** — {predicted_rain:.1f} mm")
    elif predicted_rain > 5:
        st.info(f"🌦 **Moderate** — {predicted_rain:.1f} mm")
    else:
        st.success(f"☀️ **Light** — {predicted_rain:.1f} mm")

with col_right:
    st.subheader("🌊 Stage 2 — Flood Risk")
    flood_features = np.array([[predicted_rain, elevation, distance, lat, lon]])
    proba = float(flood_model.predict_proba(flood_features)[0][1])
    st.metric("Flood Probability", f"{proba:.2f}")

    if proba > threshold:
        if proba > 0.8:
            st.error("🚨 **CRITICAL** — Evacuate now!")
        elif proba > 0.6:
            st.error("🔴 **WARNING** — Take precautions!")
        else:
            st.warning("⚠️ **WATCH** — Monitor!")
    else:
        st.success("✅ **SAFE**")

# ================================================================
# ALERT SYSTEM
# ================================================================
st.divider()
st.subheader("🚨 Stage 3 — Alert System")

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

if proba > 0.6:
    alert_level = "CRITICAL" if proba > 0.8 else "WARNING"
elif proba > 0.3:
    alert_level = "WATCH"
else:
    alert_level = "SAFE"

# Build river context
river_context = ""
if realtime["has_river"]:
    rv = realtime["river"]
    river_context = f"\n🏞️ River: {rv['river']} — {rv['status']} ({rv['level']:.1f}m³/s / {rv['danger']}m³/s)"

alert = {
    "timestamp": timestamp, "city": city, "lat": float(round(lat, 4)), "lon": float(round(lon, 4)),
    "rainfall_mm": float(round(predicted_rain, 2)), "flood_probability": float(round(proba, 3)),
    "elevation_m": int(round(elevation)), "river_distance_m": int(round(distance)),
    "alert_level": alert_level, "source": data_source,
    "river_status": rv["status"] if realtime["has_river"] else "N/A",
    "recipients": "District Collector, NDRF, Municipal Corp" if alert_level in ["CRITICAL", "WARNING"] else "Monitoring Team"
}

st.session_state.alert_history.append(alert)

if alert_level == "CRITICAL":
    st.error(f"""
**🔴 CRITICAL FLOOD ALERT — {city}**
📅 {timestamp} | 📍 {lat:.4f}, {lon:.4f}
🌧️ Rainfall: {predicted_rain:.1f}mm | 🌊 Probability: {proba:.2f}{river_context}
📡 Source: {data_source}

**🚨 ACTION: EVACUATE. Deploy NDRF rescue teams.**
📱 SMS → {alert['recipients']}
""")
    st.toast(f"🚨 CRITICAL: {city}!", icon="🚨")
elif alert_level == "WARNING":
    st.warning(f"""
**🟠 WARNING — {city}**
📅 {timestamp} | 🌧️ {predicted_rain:.1f}mm | 🌊 {proba:.2f}{river_context}
📡 Source: {data_source}
📱 Notified: {alert['recipients']}
""")
    st.toast(f"⚠️ WARNING: {city}!", icon="⚠️")
elif alert_level == "WATCH":
    st.info(f"🟡 **WATCH** — {city} | Rain: {predicted_rain:.1f}mm | Flood: {proba:.2f}")
else:
    st.success(f"🟢 **SAFE** — {city} | Rain: {predicted_rain:.1f}mm | Flood: {proba:.2f}")

with st.expander("📋 Alert JSON"):
    st.json(alert)

# ================================================================
# REPORT
# ================================================================
st.divider()
st.subheader("📋 Full Report")

report_df = pd.DataFrame({
    "Parameter": ["City", "Latitude", "Longitude", "Elevation", "River Distance",
                   "Temperature", "Humidity", "Pressure", "Wind", "Clouds",
                   "Predicted Rain", "Flood Probability", "Alert Level",
                   "River Status", "Data Source", "Timestamp"],
    "Value": [city, f"{lat:.4f}", f"{lon:.4f}", f"{elevation:.0f} m", f"{distance:.0f} m",
              f"{temperature}°C", f"{humidity}%", f"{pressure} hPa",
              f"{wind_speed} km/h", f"{cloud_cover}%",
              f"{predicted_rain:.2f} mm", f"{proba:.3f}", alert_level,
              rv["status"] if realtime["has_river"] else "N/A",
              data_source, weather_time]
})
st.table(report_df)

col_dl1, col_dl2 = st.columns(2)
col_dl1.download_button("📥 Report (CSV)", report_df.to_csv(index=False),
                        f"report_{city}.csv", "text/csv")
col_dl2.download_button("📥 Alert (JSON)", json.dumps(alert, indent=2),
                        f"alert_{city}.json", "application/json")

# ================================================================
# ALERT HISTORY
# ================================================================
if len(st.session_state.alert_history) > 1:
    st.divider()
    st.subheader("📜 Alert History")
    hist = pd.DataFrame(st.session_state.alert_history)
    st.dataframe(hist[["timestamp", "city", "rainfall_mm", "flood_probability",
                       "alert_level", "river_status", "source"]], use_container_width=True)

# ================================================================
# MODEL EVALUATION
# ================================================================
st.divider()
st.subheader("📊 Model Evaluation")

tab1, tab2 = st.tabs(["🌊 Flood Model", "🌧 Rainfall Model"])
with tab1:
    c1, c2 = st.columns(2)
    c1.image("outputs/flood_confusion_matrix.png", caption="Confusion Matrix")
    c2.image("outputs/flood_feature_importance.png", caption="Feature Importance")
with tab2:
    c3, c4 = st.columns(2)
    c3.image("outputs/rainfall_actual_vs_predicted.png", caption="R²=0.917")
    c4.image("outputs/rainfall_feature_importance.png", caption="Feature Importance")

# MAP
st.divider()
st.subheader("🗺 Map")
st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))
