import streamlit as st
import pandas as pd
import plotly.express as px
import os
import time
import pymongo

# ==========================================
# PAGE CONFIG & DYNAMIC DESIGN
# ==========================================
st.set_page_config(
    page_title="RAINWISE | 3-Tier Data Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main { background-color: #0E1117; color: #FAFAFA; }
    .metric-card {
        background-color: #1E212B; padding: 20px; border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); border-left: 5px solid #00E5FF;
        margin-bottom: 20px;
    }
    .metric-title { font-size: 1.1rem; color: #A0AAB2; margin-bottom: 5px; }
    .metric-value { font-size: 2.2rem; font-weight: bold; color: #00E5FF; }
    .header-style {
        text-align: center; background: -webkit-linear-gradient(#00E5FF, #007BFF);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 3rem; font-weight: 800; margin-bottom: 0px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; font-size: 1.2rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="header-style">🌊 RAINWISE HADOOP LIVE DASHBOARD</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #A0AAB2; font-size: 1.2rem;'>Big Data Visualization: Real-Time Stream → Processed Data → Raw Data</p>", unsafe_allow_html=True)
st.markdown("---")

# ==========================================
# DATA LOADING LOGIC
# ==========================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
REALTIME_FILE = os.path.join(BASE_DIR, "data", "raw", "realtime", "realtime_dataset.csv")
PROCESSED_FILE = os.path.join(BASE_DIR, "data", "processed", "bi_dashboard_ready_PERFECT.csv")

@st.cache_data(ttl=2)
def load_realtime_data():
    if not os.path.exists(REALTIME_FILE): return pd.DataFrame()
    try:
        df = pd.read_csv(REALTIME_FILE, low_memory=False)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        numeric_cols = ['lat', 'lon', 'rain_mm', 'elevation_m', 'distance_to_river_m', 'precipitation_mm_rain', 'hourly_max_mm_sat']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except: return pd.DataFrame()

@st.cache_data(ttl=60)
def load_processed_data():
    if not os.path.exists(PROCESSED_FILE): return pd.DataFrame()
    try:
        df = pd.read_csv(PROCESSED_FILE, low_memory=False)
        for col in ['Flood_Risk_Score', 'level', 'danger', 'distance_to_river_m', 'elevation_m', 'rain_mm', 'humidity_percent']:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Ensure Risk_Level exists for treemap
        if 'Risk_Level' not in df.columns and 'Flood_Risk_Score' in df.columns:
            df['Risk_Level'] = pd.cut(df['Flood_Risk_Score'], bins=[-1, 50, 80, 100], labels=['Low', 'Warning', 'Critical'])
        return df
    except: return pd.DataFrame()

df_realtime = load_realtime_data()
df_processed = load_processed_data()

@st.cache_resource
def get_mongo_client():
    try:
        return pymongo.MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=2000)
    except:
        return None

mongo_client = get_mongo_client()

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.title("⚙️ Controls")
auto_refresh = st.sidebar.checkbox("🟢 Enable Auto-Refresh", value=True)
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 2, 30, 5)

if auto_refresh:
    st.sidebar.success(f"Dashboard updating every {refresh_rate}s")
else:
    st.sidebar.warning("Auto-refresh paused")

# ==========================================
# TABS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(["📡 Real-Time Stream", "⚙️ Processed Data (BI)", "🗄️ Raw Data Warehouse", "🍃 MongoDB NoSQL"])

# -----------------------------------------------------------------------------
# TAB 1: REAL-TIME STREAM
# -----------------------------------------------------------------------------
with tab1:
    st.markdown("### 📡 Live Ingestion Metrics")
    if df_realtime.empty:
        st.warning("⏳ Waiting for real-time data from HDFS...")
    else:
        total_rows = len(df_realtime)
        latest_timestamp = df_realtime['timestamp'].max()
        time_str = latest_timestamp.strftime('%H:%M:%S') if pd.notnull(latest_timestamp) else "N/A"
        
        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="metric-card"><div class="metric-title">Total HDFS Rows</div><div class="metric-value">{total_rows:,}</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-title">Latest Batch Time</div><div class="metric-value">{time_str}</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-title">Kafka Throughput</div><div class="metric-value">212 msg/cycle</div></div>', unsafe_allow_html=True)

        df_sorted = df_realtime.sort_values(by='timestamp', ascending=False)
        latest_batch = df_sorted.head(212).dropna(subset=['lat', 'lon', 'rain_mm'])

        st.markdown("---")
        colA, colB = st.columns(2)
        
        # VISUAL 1: Radar Map (KEPT)
        with colA:
            st.markdown("#### 1. Live Rainfall Radar 🗺️")
            if not latest_batch.empty:
                fig1 = px.scatter_mapbox(
                    latest_batch, lat="lat", lon="lon", color="rain_mm", size="rain_mm", 
                    hover_name="city", color_continuous_scale=px.colors.sequential.Tealgrn,
                    size_max=15, zoom=5.5, mapbox_style="carto-darkmatter"
                )
                fig1.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor="#0E1117", font_color="#FAFAFA")
                st.plotly_chart(fig1, use_container_width=True)
                
        # VISUAL 2: Ingestion Trend (KEPT - Native Line Chart)
        with colB:
            st.markdown("#### 2. Kafka Ingestion Volume Trend 📈")
            df_realtime['minute'] = df_realtime['timestamp'].dt.floor('Min')
            trend = df_realtime.groupby('minute').size().reset_index(name='records').tail(15)
            trend.set_index('minute', inplace=True)
            st.line_chart(trend['records'], color="#00E5FF")

        # VISUAL 3: High Wind Speed Warning (NEW BAR GRAPH)
        st.markdown("#### 3. Real-Time High Wind / Storm Warning 🌪️")
        st.markdown("*Tracking the top 10 cities experiencing the highest wind speeds in real-time.*")
        if 'wind_speed_kmh' in df_realtime.columns:
            top_wind = latest_batch.nlargest(10, 'wind_speed_kmh')
            fig3 = px.bar(top_wind, x='wind_speed_kmh', y='city', orientation='h', color='wind_speed_kmh', color_continuous_scale="Purpor")
            fig3.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor="#0E1117", paper_bgcolor="#0E1117")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Wind speed data currently unavailable.")

# -----------------------------------------------------------------------------
# TAB 2: PROCESSED DATA (BI)
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("### ⚙️ Big Data Processing & Business Intelligence")
    if df_processed.empty:
        st.warning("Processed BI dataset not found. Run Spark pipeline to generate data.")
    else:
        df_p = df_processed.dropna(subset=['Flood_Risk_Score'])
        
        # Load advanced labeled dataset specifically for the heat map
        ADVANCED_CSV = os.path.join(BASE_DIR, "data", "processed", "training_dataset_gujarat_advanced_labeled.csv")
        try:
            df_adv = pd.read_csv(ADVANCED_CSV, low_memory=False, usecols=['lat', 'lon', 'population_2026', 'city'])
        except Exception:
            df_adv = pd.DataFrame()

        c1, c2, c3 = st.columns(3)
        avg_risk = df_p['Flood_Risk_Score'].mean() if not df_p.empty else 0
        extreme_count = len(df_p[df_p['Risk_Level'] == 'Extreme']) if 'Risk_Level' in df_p.columns else 0
        
        c1.markdown(f'<div class="metric-card"><div class="metric-title">Total Processed Rows</div><div class="metric-value">{len(df_p):,}</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-title">Avg State Flood Risk</div><div class="metric-value">{avg_risk:.2f}</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-title">Cities at Extreme Risk</div><div class="metric-value" style="color: #FF007F;">{extreme_count}</div></div>', unsafe_allow_html=True)

        colA, colB = st.columns(2)
        
        # VISUAL 4: Geospatial Population/Flood Heatmap (NEW MAP)
        with colA:
            st.markdown("#### 1. Advanced Geospatial Heat Map (Urban Risk) 🗺️")
            st.markdown("*Density heatmap showing urban population centers highly vulnerable to flooding.*")
            if not df_adv.empty and 'lat' in df_adv.columns and 'lon' in df_adv.columns:
                # Filter strictly for Gujarat's geographic bounding box
                gujarat_only = df_adv[
                    (df_adv['lat'] >= 20.0) & (df_adv['lat'] <= 25.0) & 
                    (df_adv['lon'] >= 68.0) & (df_adv['lon'] <= 75.0)
                ]
                # We use a random sample if it's too large, or just plot all
                plot_map = gujarat_only.sample(min(2000, len(gujarat_only))) if len(gujarat_only) > 2000 else gujarat_only
                
                fig4 = px.density_mapbox(
                    plot_map, lat='lat', lon='lon', z='population_2026', radius=20,
                    center=dict(lat=22.2587, lon=71.1924), zoom=5,
                    mapbox_style="carto-darkmatter", color_continuous_scale="Inferno"
                )
                fig4.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor="#0E1117")
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.info("Advanced labeled dataset not available for heatmap.")

        # VISUAL 5: Risk Category Treemap (NEW)
        with colB:
            st.markdown("#### 2. Risk Category Treemap 🎯")
            st.markdown("*Demonstrates Volume — processing thousands of city status updates simultaneously.*")
            if 'Risk_Level' in df_p.columns:
                risk_counts = df_p['Risk_Level'].value_counts().reset_index()
                risk_counts.columns = ['Risk_Level', 'Count']
                fig5 = px.treemap(risk_counts, path=['Risk_Level'], values='Count',
                                  color='Risk_Level', color_discrete_map={'Low': '#00E5FF', 'Warning': '#FFC107', 'Critical': '#FF007F'},
                                  template="plotly_dark")
                fig5.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor="#0E1117")
                st.plotly_chart(fig5, use_container_width=True)

        # VISUAL 6: Rainfall vs Humidity Scatter (NEW)
        st.markdown("#### 3. Rainfall vs. Humidity (Risk Analysis) 🧠")
        st.markdown("*Demonstrates Predictive Analytics — high humidity plus high rain equals critical flood risk.*")
        if 'rain_mm' in df_p.columns and 'humidity_percent' in df_p.columns and 'Risk_Level' in df_p.columns:
            plot_sample = df_p.sample(min(1500, len(df_p))) # Sample for performance
            fig6 = px.scatter(plot_sample, x='rain_mm', y='humidity_percent', color='Risk_Level',
                              color_discrete_map={'Low': '#00E5FF', 'Warning': '#FFC107', 'Critical': '#FF007F'},
                              opacity=0.7, template="plotly_dark")
            fig6.update_layout(xaxis_title='Rainfall (mm)', yaxis_title='Humidity (%)',
                               plot_bgcolor="#0E1117", paper_bgcolor="#0E1117")
            st.plotly_chart(fig6, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 3: RAW DATA WAREHOUSE
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("### 🗄️ Raw Data Warehouse Exploration")
    if df_realtime.empty:
        st.warning("Waiting for raw data blocks...")
    else:
        raw_nodes = df_realtime.drop_duplicates(subset=['city'])
        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="metric-card"><div class="metric-title">Unique Data Nodes</div><div class="metric-value">{len(raw_nodes)}</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-title">Raw Dimensions</div><div class="metric-value">{df_realtime.shape[1]} cols</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-title">Missing Raw Values</div><div class="metric-value">{df_realtime.isnull().sum().sum():,}</div></div>', unsafe_allow_html=True)

        colA, colB = st.columns(2)
        
        # VISUAL 7: Multi-Modal Data Lakes (NEW)
        with colA:
            st.markdown("#### 1. Multi-Modal Data Source Distribution 🗂️")
            st.markdown("*Showcasing the 'Variety' of your Big Data (Ground Archives vs. Satellites).*")
            if 'source' in df_realtime.columns:
                src_dist = df_realtime['source'].value_counts().reset_index()
                src_dist.columns = ['Source', 'Count']
                fig7 = px.pie(src_dist, values='Count', names='Source', hole=0.3, template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Prism)
                fig7.update_layout(plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", margin={"r":0,"t":0,"l":0,"b":0})
                st.plotly_chart(fig7, use_container_width=True)
            
        # VISUAL 8: Ground Truth vs Satellite Discrepancy (NEW)
        with colB:
            st.markdown("#### 2. Ground Truth vs. Satellite Discrepancy 🛰️")
            st.markdown("*Validating the need for AI: Raw satellite data often misaligns with ground sensors.*")
            if 'precipitation_mm_rain' in df_realtime.columns and 'hourly_max_mm_sat' in df_realtime.columns:
                fig8 = px.scatter(df_realtime, x='precipitation_mm_rain', y='hourly_max_mm_sat', hover_name='city',
                                  color='precipitation_mm_rain', color_continuous_scale="Blues", template="plotly_dark")
                fig8.update_layout(xaxis_title="Ground Sensor Rain (mm)", yaxis_title="Satellite Rain (mm)",
                                   plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", margin={"r":0,"t":0,"l":0,"b":0})
                st.plotly_chart(fig8, use_container_width=True)

        # VISUAL 9: Raw Geographic Topography Contour (NEW)
        st.markdown("#### 3. Raw Geographic Topography (2D Contour) ⛰️")
        st.markdown("*Visualizing the foundational relationship between elevation and historical rainfall trends.*")
        if 'elevation_m' in df_realtime.columns and 'rain_mm' in df_realtime.columns:
            fig9 = px.density_heatmap(df_realtime, x="elevation_m", y="rain_mm", 
                                      template="plotly_dark", color_continuous_scale="Plasma", nbinsx=30, nbinsy=30)
            fig9.update_layout(xaxis_title="Elevation (m)", yaxis_title="Rainfall (mm)", plot_bgcolor="#0E1117", paper_bgcolor="#0E1117")
            st.plotly_chart(fig9, use_container_width=True)
# -----------------------------------------------------------------------------
# TAB 4: MONGODB NOSQL DEMOGRAPHICS
# -----------------------------------------------------------------------------
with tab4:
    st.markdown("### 🍃 MongoDB NoSQL Document Storage")
    st.markdown("*City-level metadata and 2026 projected demographics loaded instantly via B-Tree Indexing.*")
    
    if mongo_client:
        try:
            db = mongo_client['rainwise_db']
            col_cities = db['city_summaries']
            
            c1, c2 = st.columns(2)
            
            # Query 1: Top 10 Fastest Growing Cities
            with c1:
                st.markdown("#### 1. Fastest Growing Cities (2026 Projections) 📊")
                st.code('db.city_summaries.find({}, {"_id": 0}).sort("projected_pop_2026", -1).limit(10)', language='javascript')
                
                cursor_top = col_cities.find({}, {"_id": 0}).sort("projected_pop_2026", -1).limit(10)
                df_top = pd.DataFrame(list(cursor_top))
                
                if not df_top.empty:
                    fig10 = px.bar(df_top, x='projected_pop_2026', y='city', orientation='h', color='projected_pop_2026', color_continuous_scale="Greens")
                    fig10.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", margin={"r":0,"t":0,"l":0,"b":0})
                    st.plotly_chart(fig10, use_container_width=True)
            
            # Query 2: Population Growth Comparison Scatter
            with c2:
                st.markdown("#### 2. Urban Explosion (Current vs Projected) 💥")
                st.code('db.city_summaries.find({}, {"_id": 0, "city": 1, "population": 1, "projected_pop_2026": 1})', language='javascript')
                
                cursor_all = col_cities.find({}, {"_id": 0, "city": 1, "population": 1, "projected_pop_2026": 1})
                df_all = pd.DataFrame(list(cursor_all))
                
                if not df_all.empty:
                    fig11 = px.scatter(df_all, x='population', y='projected_pop_2026', hover_name='city', color='projected_pop_2026', color_continuous_scale="Viridis")
                    fig11.update_layout(xaxis_title="Current Population", yaxis_title="Projected 2026 Population", plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", margin={"r":0,"t":0,"l":0,"b":0})
                    st.plotly_chart(fig11, use_container_width=True)
                    
        except Exception as e:
            st.error(f"MongoDB Error: {e}")
    else:
        st.warning("MongoDB is currently offline or unreachable at localhost:27017.")

# ==========================================
# AUTO-REFRESH TRIGGER
# ==========================================
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
