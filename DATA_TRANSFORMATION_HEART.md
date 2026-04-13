# 🛰️ RAINWISE: The Technical Heart of Data Transformation

This document details how the RAINWISE system transformed nearly **50GB of raw satellite and geospatial data** into clean, light-weight **CSV files** (kilobytes) optimized for machine learning models.

## 📊 Summary Table for Project Viva

| Raw Data | Original Format | Processed Format | Main Tool Used | Key Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **CHIRPS** | `.tif.gz` (9k files) | `rainfall_mm` (CSV) | `gdalwarp` + `rasterstats` | Daily history from 2000-2025 |
| **SRTM** | `.tif` (Tiles) | `elevation_m` (CSV) | `rasterio` | Elevation for runoff velocity |
| **OSM** | `.osm.pbf` (Binary) | `distance_to_river` (CSV) | `osmnx` + `geopandas` | Geospatial flood risk indicators |
| **GADM** | `.shp` (Vector) | `lat_lon` & `state` (CSV) | `geopandas` | Administrative grouping/filtering |

---

## 1. 🛰️ CHIRPS Dataset (Satellite Rainfall)
*   **Raw Data:** ~9,000+ files. Daily global rainfall maps from **2000 to 2025** in `.tif.gz` format.
*   **The "Unzipping" Phase:** Used `gunzip` to extract raw `.tif` files. Each global file is ~100MB.
*   **The "India Crop" (Gdalwarp):** Using `src/preprocessing/clip_chirps_to_india.py`, we applied the **Boundary Shapefile** (`gadm41_IND_1.shp`) as a cutline. This reduced file size by 90% by isolating the Indian subcontinent.
*   **The "CSV Conversion":** Handled in `src/preprocessing/build_rainfall_history_from_chirps.py`. We used `rasterstats` to calculate the **Mean Rainfall** for specific city centroids. This turned large raster images into a single historical time-series in `state_daily_features.csv`.

## 2. ⛰️ Elevation Dataset (DEM/SRTM)
*   **Raw Data:** SRTM (Shuttle Radar Topography Mission) tiles covering Gujarat.
*   **The "Merging" Phase:** Used `gdal_merge` to stitch multiple `.tif` tiles into a unified `merged_dem.tif`.
*   **The "Pixel Lookup":** Implemented in `src/preprocessing/extract_elevation.py` using `rasterio`.
    1.  **Coordinate Mapping:** Converted City Lat/Lon into Pixel Row/Column indexing.
    2.  **Value Extraction:** Read the specific pixel value representing height in meters.
*   **The CSV Result:** Saved into `elevation_features.csv`, providing the "Slope" context for flood velocity.

## 3. 🗺️ OSM River Data (Hydrology)
*   **Raw Data:** `india-latest.osm.pbf` (1.5GB binary file).
*   **The "River Extraction":** Processed via `src/preprocessing/extract_rivers_osmnx.py` using `osmnx` to filter for `waterway=river`, `stream`, or `canal`.
*   **The "Gujarat Filter":** Clipped the national river network to Gujarat boundaries.
*   **The "Distance Calculation":** Used `geopandas` in `src/preprocessing/compute_river_distance.py` to calculate the Euclidean distance between city centers and the nearest water body.
*   **The CSV Result:** Converted a 1.5GB binary file into `river_distance.csv`, telling the model: *"Ahmedabad is 200m from a river."*

## 4. 📐 Boundary Dataset (GADM Shapefiles)
*   **Raw Data:** `gadm41_IND_1.shp` (Vector file).
*   **Master Reference:**
    *   **Level 0:** Cropped project to India boundaries.
    *   **Level 1:** Filtered for `NAME_1 == 'Gujarat'`.
    *   **Centroids:** Extracted district centroids to establish the `lat`/`lon` anchor points for all other datasets.

## 📂 The Final Training Dataset
The file `data/processed/training_dataset_gujarat_advanced_labeled.csv` is the "Gold Standard" dataset created by joining all the above features.

*   **How it was created:** 
    1.  A temporal join was performed between the **CHIRPS Rainfall History** and the **GIS Geospatial Features** (Elevation + River Distance).
    2.  **Feature Engineering:** Calculated 3-day and 7-day rolling rainfall totals to capture soil saturation.
    3.  **Labeling:** Flood labels (`0` or `1`) were assigned based on intensity thresholds (e.g., >60mm in 3 days) and historical records.
*   **Record Count:** This file contains **2,279,281 records**, covering every district in Gujarat with daily granularity across 25 years.
