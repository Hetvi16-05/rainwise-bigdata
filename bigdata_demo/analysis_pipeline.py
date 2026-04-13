import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Step 5: Load sample into Pandas DataFrame ---
print("\n--- Step 5: Audit Schema & Data Types ---")
df = pd.read_csv("bigdata_demo/hdfs_root/rainwise/raw/india_grid.csv")
print(df.head())
print("\nSchema Inspection:")
print(df.dtypes)

# --- Step 6: Normalize column headers ---
print("\n--- Step 6: Normalize Column Headers ---")
# Example: Convert 'Latitude' to 'latitude' if it were capitalized
df.columns = [c.lower().replace(' ', '_') for c in df.columns]
print(f"Normalized columns: {df.columns.tolist()}")

# --- Step 7: Calculate missing values (Veracity) ---
print("\n--- Step 7: Identify Data Veracity Issues ---")
# Manually inject a few nulls for demonstration if none exist
if df.isnull().sum().sum() == 0:
    print("Self-Correction: Injecting nulls for demonstration...")
    df.loc[0, 'latitude'] = None
    df.loc[10, 'longitude'] = None

missing = df.isnull().sum()
print("Missing values count:")
print(missing)
print(f"Total Veracity Score: {(1 - df.isnull().any(axis=1).mean())*100:.1f}%")

# --- Step 8: Check for duplicates (Variety) ---
print("\n--- Step 8: Variety & Reliability Check ---")
duplicates = df.duplicated().sum()
print(f"Found {duplicates} duplicate records.")
print(f"Unique Coords: {len(df.groupby(['latitude', 'longitude']))}")

# --- Step 9: Statistical Summary (Outliers) ---
print("\n--- Step 9: Descriptive Statistical Summary ---")
summary = df.describe()
print(summary)

# --- Step 10: Visualizations ---
print("\n--- Step 10: Visualization of Distributions ---")
os.makedirs("bigdata_demo/plots", exist_ok=True)

# 1. Histogram (latitude distribution)
plt.figure(figsize=(10, 6))
sns.histplot(df['latitude'].dropna(), kde=True, color='skyblue')
plt.title("Distribution of Latitude (India Grid)")
plt.savefig("bigdata_demo/plots/latitude_histogram.png")
print("✅ Saved: bigdata_demo/plots/latitude_histogram.png")

# 2. Heatmap (Correlation)
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.savefig("bigdata_demo/plots/correlation_heatmap.png")
print("✅ Saved: bigdata_demo/plots/correlation_heatmap.png")

# 3. Box plot (Outliers)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, palette="Set2")
plt.title("Outlier Identification (Box Plot)")
plt.savefig("bigdata_demo/plots/outlier_boxplot.png")
print("✅ Saved: bigdata_demo/plots/outlier_boxplot.png")

print("\n🚀 10-Step Big Data Pipeline Demonstration Complete.")
