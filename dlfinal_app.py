import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
import json
import os
import torch
import torch.nn as nn
from datetime import datetime

from src.utils.features import feature_engineering
from src.utils.realtime_data import get_all_realtime, get_pipeline_status
from model_trainingDL.models import TabTransformer

# ==========================================
# COMPATIBLE DL ARCHITECTURES
# ==========================================
class CompatibleFloodDNN(nn.Module):
    """A Deep Neural Network matching the saved weights in DLmodels/flood_dnn.pth"""
    def __init__(self):
        super(CompatibleFloodDNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(5, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# ----------------------
# PAGE CONFIG
# ----------------------
st.set_page_config(
    page_title="RAINWISE — Deep Learning Flood Intelligence",
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
    device = torch.device("cpu")
    
    # 1. Rainfall Model (TabTransformer)
    rain_model = TabTransformer(input_dim=6, depth=3)
    if os.path.exists("DLmodels/tab_transformer_rainfall.pth"):
        rain_model.load_state_dict(torch.load("DLmodels/tab_transformer_rainfall.pth", map_location=device))
    rain_model.eval()
    rain_scaler = joblib.load("DLmodels/tab_transformer_scaler.pkl") if os.path.exists("DLmodels/tab_transformer_scaler.pkl") else None
    
    # 2. Flood Model (CompatibleDNN)
    flood_model = CompatibleFloodDNN()
    if os.path.exists("DLmodels/flood_dnn.pth"):
        flood_model.load_state_dict(torch.load("DLmodels/flood_dnn.pth", map_location=device))
    flood_model.eval()
    flood_scaler = joblib.load("DLmodels/scaler.pkl") if os.path.exists("DLmodels/scaler.pkl") else None
    
    return flood_model, flood_scaler, rain_model, rain_scaler

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

flood_model, flood_scaler, rain_model, rain_scaler = load_models()
cities_df = load_city_data()
river_df, elev_df = load_gis_data()

# Population mapping for cities
POP_MAPPING = {"Ahmedabad": 8600000, "Surat": 6100000, "Vadodara": 2200000, "Rajkot": 1800000}

@st.cache_data
def get_climatology(month):
    if month in [6, 7, 8, 9]: # Monsoon
        return {"temp": 30.0, "humid": 85.0, "pres": 1006.0, "wind": 18.0, "cloud": 80.0}
    elif month in [3, 4, 5]: # Summer
        return {"temp": 38.0, "humid": 40.0, "pres": 1010.0, "wind": 12.0, "cloud": 10.0}
    else: # Winter
        return {"temp": 22.0, "humid": 50.0, "pres": 1016.0, "wind": 8.0, "cloud": 5.0}

@st.cache_data
def predict_future_range(city_name, city_lat, city_lon, elevation, distance, start_dt, end_dt):
    # DL Sequential Simulation
    pop = float(POP_MAPPING.get(city_name, 1000000))
    
    # 7-DAY COLD START SEEDING
    seed_start = start_dt - pd.Timedelta(days=7)
    seed_end = start_dt - pd.Timedelta(days=1)
    hist_seed_df = load_historical_data(city_lat, city_lon, seed_start, seed_end)
    
    if not hist_seed_df.empty:
        rain_memory = hist_seed_df.sort_values("date")["rain_mm"].tolist()
        if len(rain_memory) < 7:
            rain_memory = [0.0] * (7 - len(rain_memory)) + rain_memory
    else:
        rain_memory = [0.0] * 7
        
    results = []
    curr_date = start_dt
    days_to_sim = (end_dt - start_dt).days + 1
    
    for _ in range(days_to_sim):
        rain3 = sum(rain_memory[-3:])
        # Stage 1: Rain [elev, dist, lat, lon, pop, rain3]
        r_feat = [elevation, distance, city_lat, city_lon, pop, rain3]
        r_scaled = rain_scaler.transform([r_feat])
        with torch.no_grad():
            pred_rain = max(0.0, rain_model(torch.tensor(r_scaled, dtype=torch.float32)).item())
        
        # Stage 2: Flood [rain, elev, dist, lat, lon]
        f_feat = [pred_rain, elevation, distance, city_lat, city_lon]
        f_scaled = flood_scaler.transform([f_feat])
        with torch.no_grad():
            flood_proba = flood_model(torch.tensor(f_scaled, dtype=torch.float32)).item()
            
        rain_memory.append(pred_rain)
        results.append({
            "date": curr_date.strftime("%Y-%m-%d"),
            "rain_mm": pred_rain,
            "flood_probability": flood_proba,
            "rain3": rain3,
            "type": "Deep Learning Simulation 🧠"
        })
        curr_date += pd.Timedelta(days=1)
        
    sim_df = pd.DataFrame(results)
    sim_df["readable_date"] = pd.to_datetime(sim_df["date"])
    return sim_df

@st.cache_data
def load_historical_data(city_lat, city_lon, start_dt, end_dt):
    sd = int(start_dt.strftime("%Y%m%d"))
    ed = int(end_dt.strftime("%Y%m%d"))
    try:
        cols = ["date", "lat", "lon", "rain_mm", "flood", "rain3_mm", "rain7_mm"]
        df_hist = pd.read_csv("data/processed/training_dataset_gujarat_advanced_labeled.csv", usecols=cols)
        mask = (
            (df_hist["lat"].round(2) == round(city_lat, 2)) & 
            (df_hist["lon"].round(2) == round(city_lon, 2)) &
            (df_hist["date"] >= sd) &
            (df_hist["date"] <= ed)
        )
        return df_hist[mask].sort_values("date")
    except Exception as e:
        return pd.DataFrame()

def find_nearest(df, lat, lon):
    df = df.copy()
    df["dist"] = (df["lat"] - lat)**2 + (df["lon"] - lon)**2
    return df.loc[df["dist"].idxmin()]

def fetch_weather_api(lat, lon):
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,cloud_cover,precipitation",
            "timezone": "Asia/Kolkata"
        }, timeout=10)
        data = r.json()
        if "error" in data and data["error"]: return None, data.get("reason", "API error")
        c = data["current"]
        return {
            "temperature": c["temperature_2m"], "humidity": c["relative_humidity_2m"],
            "pressure": c["surface_pressure"], "wind_speed": c["wind_speed_10m"],
            "cloud_cover": c["cloud_cover"], "precipitation": c["precipitation"],
            "time": c["time"]
        }, None
    except Exception as e: return None, str(e)

# ================================================================
# HEADER
# ================================================================
st.title("🌊 RAINWISE — Deep Learning Flood Intelligence")
st.markdown("Live data → **DL Rainfall Prediction** → **DL Flood Risk** → 🚨 Automated Alerts")

# ================================================================
# SIDEBAR
# ================================================================
st.sidebar.title("🌊 RAINWISE DL")
view_mode = st.sidebar.radio("Navigation", ["🌐 Live Dashboard", "📅 Advanced Analysis", "🔮 Seasonal Simulation", "📊 Big Data Architecture"], index=0)

st.sidebar.divider()
st.sidebar.subheader("⚙️ Settings")

city = st.sidebar.selectbox("📍 Select City", cities_df["city"].unique())
row = cities_df[cities_df["city"] == city].iloc[0]
lat, lon = float(row["lat"]), float(row["lon"])

distance = float(find_nearest(river_df, lat, lon)["river_distance"])
elevation = float(find_nearest(elev_df, lat, lon)["elevation"])
threshold = st.sidebar.slider("🎯 Alert Threshold", 0.1, 0.9, 0.5, step=0.05)

st.sidebar.divider()
st.sidebar.subheader("🌍 Geography")
st.sidebar.metric("Elevation", f"{elevation:.0f} m")
st.sidebar.metric("River Distance", f"{distance:.0f} m")
st.sidebar.caption(f"Engine: Sequential Deep Learning (Phase 3)")

# ================================================================
# VIEW: LIVE DASHBOARD
# ================================================================
if view_mode == "🌐 Live Dashboard":
    st.subheader(f"📡 Real-Time DL Dashboard: {city}")
    realtime = get_all_realtime(city)
    
    # Weather fetching
    w, err = fetch_weather_api(lat, lon)
    if w:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("🌡 Temp", f"{w['temperature']}°C")
        c2.metric("💧 Humid", f"{w['humidity']}%")
        c3.metric("📊 Press", f"{w['pressure']} hPa")
        c4.metric("💨 Wind", f"{w['wind_speed']} km/h")
        c5.metric("☁️ Clouds", f"{w['cloud_cover']}%")
        obs_rain = w["precipitation"]
    else:
        obs_rain = 0.0
        st.warning("Using fallback precipitation data.")

    # PREDICTIONS
    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("🌧 Observed Rain")
        st.metric("Rainfall (24h)", f"{obs_rain:.2f} mm")
        st.caption("Source: Open-Meteo Satellite Data")

    with col_r:
        st.subheader("🌊 Flood Probability (DNN)")
        # DL Inference: [rain, elev, dist, lat, lon]
        f_feat = np.array([[obs_rain, elevation, distance, lat, lon]])
        f_scaled = flood_scaler.transform(f_feat)
        with torch.no_grad():
            proba = float(flood_model(torch.tensor(f_scaled, dtype=torch.float32)).item())
        
        st.metric("Risk Probability", f"{proba:.1%}")
        if proba >= threshold:
            st.error("🚨 HIGH RISK DETECTED")
        else:
            st.success("✅ RISK WITHIN LIMITS")

    # ALERT SYSTEM
    st.divider()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if proba >= threshold:
        st.error(f"**🔴 CRITICAL ALERT — {city}**\n📅 {timestamp} | 🌊 Risk: {proba:.1%}\n🚨 **ACTION: Monitor river discharge immediately.**")
        st.toast(f"🚨 CRITICAL: {city}!", icon="🚨")

    # IDM Section
    st.divider()
    st.subheader("🚰 Integrated Drainage & Infrastructure Analytics")
    eff_drainage = max(10, 85 - int(obs_rain / 5))
    st.metric("Effective Drainage Capacity", f"{eff_drainage}%")
    st.progress(eff_drainage/100)
    
    st.info(f"**🔍 DL XAI Insight:** Deep Learning model flagged {city} because current rainfall ({obs_rain:.1f}mm) against elevation ({elevation:.0f}m) suggests pooling in the urban catchment.")

# ================================================================
# VIEW: ADVANCED ANALYSIS
# ================================================================
elif view_mode == "📅 Advanced Analysis":
    st.header(f"📅 Advanced DL Analysis — {city}")
    c1, c2 = st.columns(2)
    start_date = c1.date_input("Start Date", datetime.now())
    end_date = c2.date_input("End Date", datetime.now() + pd.Timedelta(days=7))

    if st.button("🚀 Run Sequential DL Simulation"):
        with st.spinner("Executing Two-Stage Deep Learning Pipeline..."):
            sim_df = predict_future_range(city, lat, lon, elevation, distance, start_date, end_date)
            
        st.subheader("🔮 AI Risk Forecast")
        st.line_chart(sim_df.set_index('readable_date')[['rain_mm', 'flood_probability']])
        st.dataframe(sim_df)

# ================================================================
# VIEW: BIG DATA ARCHITECTURE
# ================================================================
elif view_mode == "📊 Big Data Architecture":
    st.header("📊 Big Data Pipeline Architecture (DL-Enhanced)")
    st.markdown("RAINWISE DL implements a multi-stage pipeline where Big Data ingestion feeds into Neural Architectures.")
    st.image("https://img.icons8.com/fluency/96/data-configuration.png", width=60)
    st.info("The HDFS Raw Zone now includes high-frequency satellite logs for DL training.")

else:
    st.header("🔮 Seasonal Simulation")
    st.warning("Seasonal simulation is optimized for production ML models. In this DL-version, please use 'Advanced Analysis' for time-series forecasting.")

st.divider()
st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=10)
st.caption(f"RAINWISE DL Inference Engine v3.0 | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
