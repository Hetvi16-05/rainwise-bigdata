import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Define Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "bi_dashboard_ready.csv")
REPORT_DIR = os.path.join(BASE_DIR, "reports", "viva_visuals")
os.makedirs(REPORT_DIR, exist_ok=True)

# Set Style
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams['figure.dpi'] = 300

def generate():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Data not found at {DATA_PATH}. Run generate_dashboard_data.py first.")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"📊 Generating 6 Visuals from {len(df)} records...")

    # --- 1. Gujarat Risk Map (Geospatial) ---
    plt.figure(figsize=(10, 8))
    scatter = sns.scatterplot(data=df, x='lon', y='lat', size='rain_mm', 
                             hue='Risk_Level', palette='RdYlGn_r', sizes=(20, 200), alpha=0.7)
    plt.title("Gujarat Flood Risk Assessment (Real-Time Ingestion)", fontsize=15)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig(os.path.join(REPORT_DIR, "gujarat_risk_map.png"))
    plt.close()

    # --- 2. Big Data Veracity Audit (Data Quality) ---
    plt.figure(figsize=(8, 8))
    quality_bins = pd.cut(df['Data_Quality_Score'], bins=[0, 70, 85, 100], labels=['Poor', 'Fair', 'Good'])
    quality_counts = quality_bins.value_counts()
    plt.pie(quality_counts, labels=quality_counts.index, autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99'])
    plt.title("PySpark Veracity Audit: Data Integrity Share", fontsize=15)
    plt.savefig(os.path.join(REPORT_DIR, "veracity_audit_pie.png"))
    plt.close()

    # --- 3. Rainfall vs. Risk Correlation (ML Model Logic) ---
    plt.figure(figsize=(10, 6))
    sns.regplot(data=df, x='rain_mm', y='Flood_Risk_Score', scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    plt.title("Rainfall Intensity vs. Predicted Flood Risk Score", fontsize=15)
    plt.xlabel("Rainfall (mm)")
    plt.ylabel("Risk Score (0-100)")
    plt.savefig(os.path.join(REPORT_DIR, "rainfall_vs_risk.png"))
    plt.close()

    # --- 4. Elevation Shielding (Physical Logic) ---
    plt.figure(figsize=(10, 6))
    df['Elevation_Tier'] = pd.cut(df['elevation_m'], bins=[-1, 20, 100, 1000], labels=['Lowland', 'Midland', 'Highland'])
    sns.barplot(data=df, x='Elevation_Tier', y='Flood_Risk_Score', palette='viridis')
    plt.title("Impact of Elevation on Flood Vulnerability", fontsize=15)
    plt.savefig(os.path.join(REPORT_DIR, "elevation_shield.png"))
    plt.close()

    # --- 5. Distance to River Impact ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='distance_to_river_m', y='Flood_Risk_Score', color='blue')
    plt.title("Proximity to River vs. Risk Sensitivity", fontsize=15)
    plt.xlabel("Distance to Nearest River (m)")
    plt.savefig(os.path.join(REPORT_DIR, "river_proximity.png"))
    plt.close()

    # --- 6. Top 10 High-Risk Cities ---
    plt.figure(figsize=(12, 8))
    top_10 = df.nlargest(10, 'Flood_Risk_Score')
    sns.barplot(data=top_10, x='Flood_Risk_Score', y='city', palette='Reds_r')
    plt.title("Top 10 High-Risk Locations in Gujarat (Current Cycle)", fontsize=15)
    plt.savefig(os.path.join(REPORT_DIR, "top_10_cities.png"))
    plt.close()

    print(f"✅ Success! 6 Visuals generated in: {REPORT_DIR}")

if __name__ == "__main__":
    generate()
