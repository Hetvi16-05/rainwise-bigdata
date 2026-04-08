import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from src.utils.features import feature_engineering
from src.utils.realtime_data import get_all_realtime, get_pipeline_status

st.set_page_config(page_title="Flood Prediction Demo", layout="centered")

st.title("🌊 Flood Prediction Demo")
st.markdown("Predict flood risk from **rainfall & geography** using XGBoost Classifier.")

# ----------------------
# PIPELINE VISUALIZATION
# ----------------------
with st.expander("🔄 How the Pipeline Works", expanded=False):
    st.markdown("""
    ```
    📡 Real-Time Pipeline (run_realtime_pipeline.py)
    │  ├── realtime_weather.py  → Weather logs
    │  ├── realtime_rainfall.py → Rainfall logs
    │  └── realtime_river.py    → River level logs
    │
    ↓  build_dataset.py merges all data
    │
    📥 User Input OR Pipeline Data
         ↓
    🗺️ GIS Lookup (Elevation, River Distance)
         ↓
    🤖 Flood Model (XGBoost Classifier)
         ↓
    📊 Risk Assessment + 🚨 Alert
    ```
    """)

# ----------------------
# LOAD MODEL
# ----------------------
flood_model = joblib.load("models/flood_model.pkl")

# ----------------------
# CITY DATA
# ----------------------
cities_df = pd.read_csv("data/config/gujarat_cities.csv")
cities_df.columns = cities_df.columns.str.lower()

city = st.selectbox("📍 Select City", cities_df["city"].unique())

row = cities_df[cities_df["city"] == city]
lat = float(row["lat"].values[0])
lon = float(row["lon"].values[0])

# ----------------------
# GIS DATA
# ----------------------
river_df = pd.read_csv("data/processed/gujarat_river_distance.csv")
elev_df = pd.read_csv("data/processed/gujarat_elevation.csv")
river_df.columns = river_df.columns.str.lower()
elev_df.columns = elev_df.columns.str.lower()

def find_nearest(df, lat, lon):
    df = df.copy()
    df["dist"] = (df["lat"] - lat)**2 + (df["lon"] - lon)**2
    return df.loc[df["dist"].idxmin()]

distance = float(find_nearest(river_df, lat, lon)["river_distance"])
elevation = float(find_nearest(elev_df, lat, lon)["elevation"])

# ----------------------
# LOCATION INFO
# ----------------------
st.subheader("🌍 Location & Geography")
col1, col2 = st.columns(2)
col1.metric("Latitude", f"{lat:.4f}")
col2.metric("Longitude", f"{lon:.4f}")

col3, col4 = st.columns(2)
col3.metric("Distance to River", f"{distance:.0f} m")
col4.metric("Elevation", f"{elevation:.0f} m")

# ----------------------
# REAL-TIME PIPELINE DATA
# ----------------------
st.subheader("📡 Real-Time Pipeline Data")

realtime = get_all_realtime(city)

if realtime["pipeline_active"]:
    st.success("✅ Pipeline data available!")

    rt_cols = st.columns(3)

    # Rainfall from pipeline
    if realtime["has_rainfall"]:
        rt_rain = realtime["rainfall"]["precipitation_mm"]
        rt_cols[0].metric("🌧 Pipeline Rainfall", f"{rt_rain} mm")
    else:
        rt_rain = None
        rt_cols[0].metric("🌧 Rainfall", "No data")

    # River level from pipeline
    if realtime["has_river"]:
        rv = realtime["river"]
        rt_cols[1].metric(f"🏞 {rv['river']} River", f"{rv['level']:.1f}m")
        if rv["status"] == "Above Danger":
            st.error(f"🚨 {rv['river']} River is ABOVE DANGER level! ({rv['level']:.1f}m / {rv['danger']}m)")
        elif rv["status"] == "Warning":
            st.warning(f"⚠️ {rv['river']} River at WARNING level ({rv['level']:.1f}m / {rv['warning']}m)")
        else:
            st.caption(f"River status: {rv['status']} (Level: {rv['level']:.1f}m, Danger: {rv['danger']}m)")
    else:
        rt_cols[1].metric("🏞 River", "No data")

    # Weather from pipeline
    if realtime["has_weather"]:
        w = realtime["weather"]
        rt_cols[2].metric("🌡 Temperature", f"{w['temperature']}°C")
    else:
        rt_cols[2].metric("🌡 Weather", "No data")

    use_pipeline = st.checkbox("📡 Use pipeline rainfall for prediction", value=rt_rain is not None and rt_rain > 0)
else:
    st.info("ℹ️ No pipeline data found. Run `python src/data_collection/run_realtime_pipeline.py` to collect data.")
    use_pipeline = False
    rt_rain = None

# ----------------------
# RAINFALL INPUT
# ----------------------
st.subheader("🌧 Rainfall Input")

if use_pipeline and rt_rain is not None:
    rain = st.slider("Rainfall (mm)", 0.0, 100.0, min(float(rt_rain), 100.0), step=0.5,
                     help="Pre-filled from real-time pipeline data")
    st.caption(f"📡 Value from pipeline: {rt_rain} mm")
else:
    rain = st.slider("Rainfall (mm)", 0.0, 100.0, 10.0, step=0.5)

threshold = st.slider("🎯 Alert Threshold", 0.1, 0.9, 0.5, step=0.05)

# ----------------------
# PREDICT
# ----------------------
if st.button("🔍 Predict Flood Risk"):
    features = np.array([[rain, elevation, distance, lat, lon]])
    proba = flood_model.predict_proba(features)[0][1]

    st.divider()
    st.subheader("📊 Flood Risk Assessment")

    col_f1, col_f2 = st.columns(2)
    col_f1.metric("Flood Probability", f"{proba:.2f}")

    if proba > threshold:
        if proba > 0.8:
            col_f2.metric("Risk Level", "🔴 HIGH")
            st.error("🚨 HIGH FLOOD RISK — Evacuate low-lying areas immediately!")
        elif proba > 0.6:
            col_f2.metric("Risk Level", "🟠 SIGNIFICANT")
            st.error("🔴 SIGNIFICANT FLOOD RISK — Take precautionary measures!")
        else:
            col_f2.metric("Risk Level", "🟡 MODERATE")
            st.warning("⚠️ MODERATE FLOOD RISK — Stay alert and monitor conditions!")
    else:
        col_f2.metric("Risk Level", "🟢 LOW")
        st.success("✅ LOW FLOOD RISK — Conditions are currently safe.")

    # --- ALERT SYSTEM ---
    st.divider()
    st.subheader("🚨 Alert System")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data_source = "Real-Time Pipeline" if use_pipeline else "Manual Input"

    # Combine with river data if available
    river_alert = ""
    if realtime["has_river"] and realtime["river"]["status"] != "Normal":
        rv = realtime["river"]
        river_alert = f"\n🏞️ River: {rv['river']} at {rv['level']:.1f}m ({rv['status']})"

    if proba > 0.6:
        alert_level = "🔴 CRITICAL" if proba > 0.8 else "🟠 WARNING"
        st.error(f"""
**{alert_level} FLOOD ALERT — {city}**

📅 {timestamp} | 📍 {city} ({lat:.4f}, {lon:.4f})
🌧️ Rainfall: {rain} mm | 🌊 Probability: {proba:.2f}
⛰️ Elevation: {elevation:.0f}m | River Distance: {distance:.0f}m{river_alert}
📡 Data Source: {data_source}

**Action:** {"🚨 EVACUATE immediately!" if proba > 0.8 else "⚠️ Prepare for flooding."}
📱 SMS sent to: District Collector, NDRF, Municipal Corp
""")
        st.toast(f"🚨 FLOOD ALERT for {city}!", icon="🚨")

        sms_log = {
            "timestamp": timestamp,
            "city": city,
            "rainfall_mm": rain,
            "flood_probability": round(proba, 3),
            "alert_level": "CRITICAL" if proba > 0.8 else "WARNING",
            "data_source": data_source,
            "river_status": realtime["river"]["status"] if realtime["has_river"] else "N/A",
            "recipients": "District Collector, NDRF, Municipal Corp"
        }
        st.json(sms_log)
    elif proba > threshold:
        st.info(f"ℹ️ Moderate alert for {city} at {timestamp}.")
    else:
        st.success(f"✅ No alert needed. {city} is safe at {timestamp}.")

    # Download
    summary_df = pd.DataFrame({
        "Parameter": ["City", "Rainfall", "Elevation", "River Distance", "Flood Probability",
                      "Data Source", "River Status"],
        "Value": [city, f"{rain} mm", f"{elevation:.0f} m", f"{distance:.0f} m",
                  f"{proba:.2f}", data_source,
                  realtime["river"]["status"] if realtime["has_river"] else "N/A"]
    })
    st.table(summary_df)
    st.download_button("📥 Download Report", summary_df.to_csv(index=False),
                       f"flood_report_{city}.csv", "text/csv")

# ----------------------
# MODEL EVALUATION
# ----------------------
st.divider()
st.subheader("📊 Flood Model Evaluation")
st.image("outputs/flood_confusion_matrix.png", caption="Confusion Matrix")
st.image("outputs/flood_feature_importance.png", caption="Feature Importance")

st.subheader("🗺 Map")
st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))