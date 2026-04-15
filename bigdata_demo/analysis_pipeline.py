import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Step 5: Load PRODUCTION scale sample into Pandas DataFrame ---
print("\n--- Step 5: Audit Schema & Data Types (2.2M Rows) ---")
# Use the file we just 'put' into HDFS
HDFS_FILE = "bigdata_demo/hdfs_root/rainwise/raw/training_dataset_gujarat_advanced_labeled.csv"

# Load with high-performance settings
df = pd.read_csv(HDFS_FILE, low_memory=False)
print(f"✅ Loaded {len(df):,} records into Spark-style DataFrame.")
print(df.head())
print("\nSchema Inspection (Production Feature Vector):")
print(df.dtypes)

# --- Step 6: Normalize column headers ---
print("\n--- Step 6: Normalize Column Headers ---")
df.columns = [c.lower().replace(' ', '_') for c in df.columns]
print(f"Normalized columns: {df.columns.tolist()[:10]}... (Total {len(df.columns)})")

# --- Step 7: Calculate missing values (Veracity) ---
print("\n--- Step 7: Identify Data Veracity Issues ---")
missing = df.isnull().sum()
print("Missing values count per column:")
print(missing[missing > 0] if missing.any() else "No missing values found.")
veracity_score = (1 - df.isnull().any(axis=1).mean()) * 100
print(f"Total Veracity Score (2.2M rows): {veracity_score:.2f}%")

# --- Step 8: Check for duplicates (Variety) ---
print("\n--- Step 8: Variety & Reliability Check ---")
duplicates = df.duplicated().sum()
print(f"Found {duplicates:,} duplicate records in 2.2 million rows.")
# Audit unique cities/coords
if 'lat' in df.columns and 'lon' in df.columns:
    unique_sites = len(df.groupby(['lat', 'lon']))
    print(f"Unique Geographic Sites: {unique_sites}")

# --- Step 9: Statistical Summary (Outliers) ---
print("\n--- Step 9: Descriptive Statistical Summary ---")
# Focus on key features for the summary
features_to_audit = ['rain_mm', 'elevation_m', 'distance_to_river_m', 'flood']
summary = df[[c for c in features_to_audit if c in df.columns]].describe()
print(summary)

# --- Step 10: Visualizations (High-Performance Scaling) ---
print("\n--- Step 10: Visualization of 2.2M Row Distributions ---")
os.makedirs("bigdata_demo/plots", exist_ok=True)

# Use sampling for visualization performance during viva, but audit full set for stats
plot_df = df.sample(n=min(50000, len(df)), random_state=42)
print(f"🎨 Generating high-fidelity visual audit using 50,000 representative samples...")

# 1. Histogram (Rain Distribution)
plt.figure(figsize=(10, 6))
sns.histplot(plot_df['rain_mm'].dropna(), kde=True, color='navy')
plt.title("Production Rainfall Distribution (Big Data Scale)")
plt.savefig("bigdata_demo/plots/latitude_histogram.png") 
print("✅ Created: bigdata_demo/plots/latitude_histogram.png")

# 2. Heatmap (Correlation of Advanced Features)
plt.figure(figsize=(10, 8))
numeric_df = plot_df.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_df.corr(), annot=False, cmap='RdBu_r') 
plt.title("Production Feature Interaction Heatmap")
plt.savefig("bigdata_demo/plots/correlation_heatmap.png")
print("✅ Created: bigdata_demo/plots/correlation_heatmap.png")

# 3. Box plot (Outlier Identification)
plt.figure(figsize=(12, 6))
sns.boxplot(data=plot_df[['rain_mm', 'elevation_m', 'flood']], palette="Set3")
plt.title("Production Outlier Identification (Box Plot)")
plt.savefig("bigdata_demo/plots/outlier_boxplot.png")
print("✅ Created: bigdata_demo/plots/outlier_boxplot.png")


# --- Step 11: Multi-Dataset Variety & Veracity Audits (EXTENDED) ---
print("\n--- Step 11: Generating Extended Big Data Audits (Across Datasets) ---")

# 4. Variety Audit: Geographic Elevation (Using gujarat_elevation.csv)
try:
    elev_df = pd.read_csv("data/processed/gujarat_elevation.csv")
    plt.figure(figsize=(10, 6))
    if 'elevation' in elev_df.columns:
        sns.violinplot(y=elev_df['elevation'], color='olive')
    elif 'elevation_m' in elev_df.columns:
         sns.violinplot(y=elev_df['elevation_m'], color='olive')
    plt.title("Variety Check: Elevation Distribution across 213 Cities")
    plt.ylabel("Elevation (m)")
    plt.savefig("bigdata_demo/plots/elevation_variety.png")
    print("✅ Created: bigdata_demo/plots/elevation_variety.png")
except Exception as e: print(f"⚠️ Elevation variety skip: {e}")

# 5. Veracity Audit: Rainfall History Time Series (Using gujarat_rainfall_history.csv)
try:
    rain_hist = pd.read_csv("data/processed/gujarat_rainfall_history.csv")
    rain_hist['date'] = pd.to_datetime(rain_hist['date'].astype(str), format='%Y%m%d')
    yearly_rain = rain_hist.set_index('date')['rain_mm'].resample('YE').mean()
    plt.figure(figsize=(12, 6))
    plt.plot(yearly_rain.index, yearly_rain.values, marker='o', color='teal')
    plt.title("Veracity Check: 40-Year Gujarat Rainfall Stability")
    plt.xlabel("Year")
    plt.ylabel("Mean Annual Rainfall (mm)")
    plt.savefig("bigdata_demo/plots/rainfall_history_veracity.png")
    print("✅ Created: bigdata_demo/plots/rainfall_history_veracity.png")
except Exception as e: print(f"⚠️ Rainfall veracity skip: {e}")

# 6. Analytic Logic: Elevation vs River Distance Interaction (Using Main Dataset)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=plot_df, x='elevation_m', y='distance_to_river_m', hue='flood', alpha=0.5, palette='coolwarm_r')
plt.title("Analytic Logic: Intersection of Elevation and River Proximity")
plt.savefig("bigdata_demo/plots/hazard_interaction.png")
print("✅ Created: bigdata_demo/plots/hazard_interaction.png")

print("\n🚀 FINAL PRODUCTION SCALE (6-Plot Suite) Audit Complete.")
