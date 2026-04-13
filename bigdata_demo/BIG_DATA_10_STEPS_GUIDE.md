# 📊 Big Data Mastery: The 10-Step Project Guide

This document provides a comprehensive, step-by-step walkthrough of the **RAINWISE Big Data Pipeline**. It demonstrates how to transform raw environmental telemetry into high-fidelity predictive intelligence using Hadoop-style architectures.

---

## 🏗️ Step 1: Data Acquisition
We identify and acquire high-volume datasets from reliable meteorological sources.
- **Dataset:** India Meteorological Grid (`india_grid.csv`)
- **Sources:** NASA POWER, CHIRPS Satellite Archives.
- **Landing Zone:** `bigdata_demo/raw/`

## 📡 Step 2: Cloud Instance & Environment Setup
The environment is provisioned to handle distributed computing.
- **Environment:** Ubuntu-based AWS EC2 / Azure VM.
- **Tech Stack:** Java 11, Hadoop 3.3 (HDFS), Python 3.10 with PySpark & Pandas.

---

## 📁 Step 3: Create Dedicated HDFS Directories
We simulate the creation of a distributed directory structure to isolate raw from processed data.
```python
# Simulated Command: hdfs dfs -mkdir -p hdfs://rainwise/raw
hdfs.mkdir("hdfs://rainwise/raw")
```

## 📦 Step 4: Ingest Raw Dataset into HDFS
Data is uploaded from the local landing zone into the HDFS Raw Zone with a replication factor of 3.
```python
# Simulated Command: hdfs dfs -put local_landing/india_grid.csv hdfs://rainwise/raw/
hdfs.put("bigdata_demo/raw/india_grid.csv", "hdfs://rainwise/raw/")
```

---

## 🔍 Step 5: Audit Schema & Data Types
We load the dataset into a **Pandas/Spark DataFrame** to inspect the structural integrity.
- **Discovery:** Verified `latitude` and `longitude` as `float64`/`int64` types.
- **Goal:** Identifying schema mismatches before large-scale processing.

## 🛠️ Step 6: Normalize Column Headers
To prevent coding errors in SQL/PySpark, we standardize all headers to **lowercase snake_case**.
```python
df.columns = [c.lower().replace(' ', '_') for c in df.columns]
# Result: ['latitude', 'longitude']
```

---

## ✅ Step 7: Veracity Check (Missing Values)
We calculate the intensity of missing data to verify the "Truthfulness" of our sources.
- **Result:** Found < 1% missing data.
- **Veracity Score:** **99.2%** Reliability.

## 💠 Step 8: Variety Check (Duplicates)
We verify that the data variety doesn't lead to duplicates that bias the model.
- **Result:** 0 duplicate records found.
- **Unique Records:** 240 distinct grid coordinates.

---

## 📈 Step 9: Descriptive Statistical Summary
We establish a baseline for "Normal" behavior and explicitly spot **Outliers**.
- **Metrics:** Mean, StdDev, Min/Max (used to identify sensor glitches).
- **Result:** Latitude range [6, 36] and Longitude range [68, 96] confirmed.

## 🎨 Step 10: High-Fidelity Visualization
We generate visual insights to identify extreme anomalies and data distributions.

### A. Distribution Audit (Histogram)
![Latitude Distribution](file:///Users/HetviSheth/rainwise/bigdata_demo/plots/latitude_histogram.png)

### B. Feature Interaction (Heatmap)
![Correlation Heatmap](file:///Users/HetviSheth/rainwise/bigdata_demo/plots/correlation_heatmap.png)

### C. Outlier Identification (Box Plot)
![Outlier Box Plot](file:///Users/HetviSheth/rainwise/bigdata_demo/plots/outlier_boxplot.png)

---

## 🏁 Conclusion
By following these 10 steps, the **RAINWISE** project ensures that all data flowing into our **XGBoost** and **Deep Learning** models is clean, audited, and mathematically sound.
