# 🏗️ RAINWISE: Technical Architecture & Data Pipeline

This document provides a deep technical breakdown of how RAINWISE processes diverse datasets into high-fidelity flood predictions.

---

## 📡 1. Data Collection & Acquisition

RAINWISE integrates multiple real-time and historical sources to create a multi-layered environmental view.

| Source | Data Points | Update Frequency | Purpose |
| :--- | :--- | :--- | :--- |
| **NASA POWER** | Rain, Temp, Humidity, Wind | Daily | Baseline weather features |
| **CHIRPS** | Global Satellite Rainfall | Daily (Historical) | High-resolution spatial rainfall |
| **CWC India** | River Levels & Discharge | Real-time | Ground-truth hydrology |
| **OpenStreetMap** | Waterways, Rivers, Canals | Static | Geospatial boundary & river mapping |
| **NASA SRTM** | Elevation (DEM) | Static | Topographical analysis |

---

## 🗺️ 2. Geospatial Feature Engineering

This is the "Brain" of the project where raw spatial files are converted into machine-learning features.

### A. TIF Image to CSV (Elevation & Slope)
We use the **`rasterio`** library to sample pixel data from Digital Elevation Model (DEM) `.tif` files.
*   **Process:**
    1.  The `.tif` file is opened as a coordinate-indexed raster.
    2.  For every GPS coordinate (lat/lon) in our data, we use the `src.index(lon, lat)` function to convert spatial coordinates into **pixel row/column indices**.
    3.  The specific pixel value at that index is extracted (e.g., `elevation_m` in meters).
    4.  The result is appended to the Rainfall CSV, effectively "joining" spatial imagery with tabular data.

### B. River Distance Calculation
Calculated using **GeoPandas** (`gpd`) and **Shapely** for exact geometric distance.
*   **Process:**
    1.  **Coordinate Projection:** We project the default GPS CRS (EPSG:4326) to **EPSG:3857 (Web Mercator)**. This is crucial because EPSG:3857 uses **meters** as units, allowing for accurate Euclidean distance calculation.
    2.  **Point Creation:** Every weather station/city is converted into a `Point` geometry.
    3.  **Minimum Distance:** We use the `.distance()` function to find the minimum distance from each point to the nearest line segment in the **Global Rivers Shapefile**.
    4.  **Unit Conversion:** The distance is calculated in meters (`distance_to_river_m`) and saved to the processed dataset.

---

## 🌊 3. Data Flow & Hadoop Layering

Data moves through a multi-stage pipeline, mimicking HDFS architecture.

### 📁 `data/raw` (The Landing Zone)
*   Contains original API responses (JSON/CSV) and raw `.tif`/`.shp` files.
*   **Rule:** This data is never modified (Immutable Layer).

### 📁 `data/interim` (The Staging Zone)
*   Data is cleaned and standardized.
*   **Actions:** Normalizing headers (snake_case), clipping global files to India's boundaries, and fixing coordinate precision.

### 📁 `data/gis` (The Spatial Analysis Zone)
*   Contains intermediate geospatial outputs.
*   **Contents:** Clipped Shapefiles, processed GeoPackages, and drainage density grids.

### 📁 `data/processed` (The Gold Zone)
*   The final, "clean" datasets used for PySpark and XGBoost training.
*   **Action:** Merging weather, elevation, and river distance into a single unified row.

---

## 🤖 4. Feature Selection Strategy

Why did we choose these specific features?

1.  **Cumulative Rainfall (3D, 7D):** Floods rarely happen due to just one hour of rain; they are caused by the **Saturation** of the soil. 3-day and 7-day lags represent soil moisture levels.
2.  **Elevation & Slope:** Used to calculate **Runoff Velocity**. High slope + Low elevation cities are at high risk of flash floods.
3.  **River Proximity:** The primary spatial indicator of flood risk.
4.  **Drainage Density:** Calculated by counting river intersections in a grid—representing how efficiently a region drains water.
