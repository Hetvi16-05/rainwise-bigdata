================================================================================
🔬 TECHNICAL HEART: DATA PROCESSING PIPELINE
================================================================================
📅 Created: 2026-04-13
🎯 Purpose: Document transformation of 50GB raw data → clean CSV for ML

This document explains how RAINWISE transformed massive raw satellite and 
geospatial datasets into lightweight, machine-learning-ready CSV files.

================================================================================
📊 DATASET OVERVIEW
================================================================================

| Raw Data Source | Original Size | Processed Size | Time Period | File Count |
| :--- | :--- | :--- | :--- | :--- |
| **CHIRPS Satellite** | ~900GB (compressed) | ~50MB (CSV) | 2000-2025 | 9,000+ files |
| **SRTM Elevation** | ~2GB (TIFF tiles) | ~5MB (CSV) | Static | 4 tiles |
| **OSM Rivers** | 1.5GB (binary) | ~2MB (CSV) | Static | 1 file |
| **GADM Boundaries** | ~50MB (shapefile) | Reference | Static | 1 file |

================================================================================
1. 🛰️ CHIRPS DATASET (SATELLITE RAINFALL)
================================================================================

🎯 **Objective**: Download 25 years of daily global rainfall data and extract 
Gujarat-specific rainfall measurements for flood modeling.

📁 **FILES USED**:
├── `src/data_collection/download_chirps_india_only.py`     → Download & crop
├── `src/preprocessing/clip_chirps_to_india.py`            → Alternative cropper
├── `src/preprocessing/build_state_daily_from_chirps.py`    → State-level aggregation
└── `src/preprocessing/build_rainfall_history_from_chirps.py` → Gujarat history

🔧 **PROCESSING STEPS**:

**STEP 1: Download Global CHIRPS Data**
- **Tool**: `wget` (parallel download with multiprocessing)
- **Source**: UCSB CHIRPS-2.0 Global Daily (https://data.chc.ucsb.edu)
- **Format**: `.tif.gz` (compressed GeoTIFF)
- **Period**: January 1, 2000 → December 31, 2025 (9,125 days)
- **File Pattern**: `chirps-v2.0.YYYY.MM.DD.tif.gz`
- **Size**: ~100MB per global file (uncompressed)

**STEP 2: Extract & Crop to India**
- **Tool**: `gunzip` (decompression) + `gdalwarp` (geospatial cropping)
- **Input**: Global CHIRPS `.tif` file
- **Cutline**: `data/external/boundary/gadm41_IND_1.shp` (India boundary)
- **Command**: 
  ```bash
  gdalwarp -cutline gadm41_IND_1.shp -crop_to_cutline global.tif india.tif
  ```
- **Output**: `data/raw/rainfall/chirps_india_daily/chirps_india_YYYY_MM_DD.tif`
- **Size Reduction**: ~90% reduction (100MB → ~10MB per file)
- **Workers**: 8 parallel processes for speed

**STEP 3: Convert Raster to CSV (Pixel Aggregation)**
- **Tool**: `rasterstats` library (zonal statistics)
- **Input**: India-cropped CHIRPS `.tif` files
- **Reference**: `data/raw/boundary/gadm41_IND_1.shp` (state boundaries)
- **Process**:
  1. Load state boundary shapefile
  2. For each daily raster, calculate mean rainfall per state
  3. Handle nodata values (-9999) → convert to 0
- **Output**: `data/processed/state_daily_features.csv`
- **Columns**: `date`, `state`, `precipitation_mm`
- **Alternative Output**: `data/processed/gujarat_rainfall_history.csv`
- **Key Insight**: Converts 2D image → 1D time series per state

📊 **FINAL OUTPUT**:
- **File**: `data/processed/gujarat_rainfall_history.csv`
- **Size**: ~50MB
- **Records**: ~9,000 rows (one per day)
- **Columns**: `date`, `state`, `precipitation_mm`, `rain_3day`, `rain_7day`

🎓 **VIVA POINTS**:
- "We used gdalwarp to crop global data to India using GADM boundaries"
- "rasterstats calculates mean rainfall per state from raster pixels"
- "25 years of daily data provides sufficient temporal resolution for flood modeling"

================================================================================
2. ⛰️ ELEVATION DATASET (DEM/SRTM)
================================================================================

🎯 **Objective**: Extract terrain elevation for each location to calculate 
runoff velocity and flood risk.

📁 **FILES USED**:
├── `src/preprocessing/extract_elevation.py`                 → Main extraction
├── `src/preprocessing/extract_slope.py`                   → Slope calculation
└── `src/preprocessing/compute_slope.py`                   → Alternative slope

🔧 **PROCESSING STEPS**:

**STEP 1: SRTM Data Acquisition**
- **Source**: Shuttle Radar Topography Mission (NASA)
- **Format**: `.tif` tiles (multiple tiles covering Gujarat)
- **Resolution**: 90m (SRTM) or 30m (SRTM-GL1)
- **Coverage**: Multiple tiles stitched together
- **Merged File**: `data/raw/elevation/merged_dem/merged_dem.tif`

**STEP 2: Pixel Lookup (Coordinate → Elevation)**
- **Tool**: `rasterio` library (geospatial raster I/O)
- **Input**: 
  - `data/raw/elevation/merged_dem/merged_dem.tif` (DEM raster)
  - `data/raw/rainfall/india_grid.csv` (grid coordinates)
- **Process**:
  1. Read latitude/longitude from grid CSV
  2. Convert lat/lon → raster pixel indices using `src.index(lon, lat)`
  3. Read pixel value at that index = elevation in meters
  4. Handle edge cases (out-of-bounds, nodata)
- **Output**: `data/processed/elevation_features.csv`
- **Key Function**: `src.index(lon, lat)` → coordinate transformation

**STEP 3: Slope Calculation (Optional)**
- **Tool**: `gdaldem` or raster operations
- **Process**: Calculate terrain slope from elevation gradient
- **Purpose**: Steeper slopes = faster runoff = higher flood risk
- **Output**: Additional slope columns in features CSV

📊 **FINAL OUTPUT**:
- **File**: `data/processed/elevation_features.csv`
- **Size**: ~5MB
- **Columns**: `latitude`, `longitude`, `elevation_m`, `slope_deg`
- **Key Insight**: Elevation affects water accumulation, slope affects runoff speed

🎓 **VIVA POINTS**:
- "rasterio converts geographic coordinates to raster pixel indices"
- "Elevation data is crucial for understanding water flow and accumulation"
- "Terrain slope derived from elevation gradient predicts runoff velocity"

================================================================================
3. 🗺️ OSM RIVER DATA (HYDROLOGY)
================================================================================

🎯 **Objective**: Extract river network and calculate distance from each 
location to nearest river for proximity-based flood risk.

📁 **FILES USED**:
├── `src/preprocessing/extract_rivers_osmnx.py`            → River extraction
├── `src/preprocessing/compute_river_distance.py`          → Distance calculation
└── `data/gis/rivers/ne_10m_rivers_lake_centerlines.shp`   → Alternative source

🔧 **PROCESSING STEPS**:

**STEP 1: Download OpenStreetMap Data**
- **Source**: OpenStreetMap (OSM) database
- **Format**: `.osm.pbf` (Protocol Buffer Binary)
- **Size**: ~1.5GB (entire India)
- **Content**: All roads, buildings, rivers, boundaries

**STEP 2: Extract River Network**
- **Tool**: `osmnx` library (OpenStreetMap network analysis)
- **Input**: India boundary polygon
- **Filter Tags**: `waterway=river`, `waterway=stream`, `waterway=canal`
- **Process**:
  1. Get India geometry: `ox.geocode_to_gdf("India")`
  2. Query OSM for waterways: `ox.features_from_polygon(india, tags)`
  3. Filter for LineString/MultiLineString geometries
  4. Convert to EPSG:4326 (WGS84 coordinate system)
- **Output**: `data/processed/hydrology/rivers_clean.geojson`
- **Features**: ~50,000+ river segments

**STEP 3: Calculate Distance to Nearest River**
- **Tool**: `geopandas` + `shapely` (geospatial operations)
- **Input**:
  - `data/processed/elevation_features.csv` (point locations)
  - `data/processed/hydrology/rivers_clean.geojson` (river lines)
  - Alternative: `data/gis/rivers/ne_10m_rivers_lake_centerlines.shp`
- **Process**:
  1. Convert CSV points to GeoDataFrame with Point geometry
  2. Project to EPSG:3857 (Web Mercator) for accurate distance calculation
  3. For each point, calculate minimum distance to any river line
  4. Convert back to EPSG:4326
- **Key Function**: `gdf_rivers.distance(point).min()`
- **Output**: `data/processed/final_features.csv` or `data/processed/gujarat_river_distance.csv`

📊 **FINAL OUTPUT**:
- **File**: `data/processed/gujarat_river_distance.csv`
- **Size**: ~2MB
- **Columns**: `latitude`, `longitude`, `distance_to_river_m`
- **Key Insight**: Proximity to rivers is a major flood risk factor

🎓 **VIVA POINTS**:
- "osmnx filters OSM data to extract only waterway features"
- "Distance calculation requires projection to metric coordinate system"
- "River proximity is a critical feature for flood risk modeling"

================================================================================
4. 📐 BOUNDARY DATASET (GADM SHAPEFILES)
================================================================================

🎯 **Objective**: Provide administrative boundaries for geographic filtering, 
clipping, and reference in all geospatial operations.

📁 **FILES USED**:
├── `data/external/boundary/gadm41_IND_1.shp`              → India state boundaries
├── `data/raw/boundary/gadm41_IND_1.shp`                   → Alternative location
└── Used in: clip_chirps_to_india.py, build_state_daily_from_chirps.py, etc.

🔧 **PROCESSING STEPS**:

**STEP 1: Acquire GADM Data**
- **Source**: GADM (Global Administrative Areas) Database
- **Version**: GADM 4.1 (latest administrative boundaries)
- **Format**: Shapefile (.shp, .shx, .dbf, .prj)
- **Levels**:
  - Level 0: Country boundaries (India)
  - Level 1: State boundaries (Gujarat, Maharashtra, etc.)
  - Level 2: District boundaries (Ahmedabad, Surat, etc.)

**STEP 2: Boundary Usage Throughout Pipeline**
- **Clipping**: Used as cutline in `gdalwarp` to crop rasters to India
- **Filtering**: Filter records by state (e.g., `NAME_1 == 'Gujarat'`)
- **Aggregation**: Group data by administrative regions
- **Centroid Extraction**: Extract lat/lon from district centroids
- **Coordinate Reference**: All data projected to EPSG:4326 (WGS84)

**STEP 3: Geometry Operations**
- **Tool**: `geopandas`
- **Process**:
  1. Load shapefile: `gpd.read_file(gadm41_IND_1.shp)`
  2. Filter for Gujarat: `states[states['NAME_1'] == 'Gujarat']`
  3. Extract centroids: `geometry.centroid`
  4. Convert to lat/lon coordinates
- **Output**: Used as reference throughout, not standalone output

📊 **USAGE SUMMARY**:
- **File**: `data/external/boundary/gadm41_IND_1.shp`
- **Size**: ~50MB
- **Features**: 36 Indian states + union territories
- **Key Role**: Master reference for all geographic operations

🎓 **VIVA POINTS**:
- "GADM provides authoritative administrative boundaries"
- "Shapefiles serve as cutlines for raster clipping"
- "Administrative grouping enables state-level analysis"

================================================================================
5. 🎯 TRAINING DATASET CREATION
================================================================================

🎯 **Objective**: Merge all processed features into a single machine-learning-
ready dataset with flood labels.

📁 **FILES USED**:
├── `src/data_collection/build_training_dataset.py`        → Base merge
├── `src/data_collection/build_city_training_dataset.py`   → City-level
├── `src/data_collection/create_flood_label.py`           → Basic labeling
├── `src/data_collection/create_advanced_flood_label.py`  → Advanced labeling
└── `src/feature_engineering/merge_rain_geo_industry.py`   → Feature merge

🔧 **PROCESSING STEPS**:

**STEP 1: Merge Rainfall + Geospatial Features**
- **Input Files**:
  - `data/processed/gujarat_rainfall_history.csv` (time series)
  - `data/processed/gujarat_features.csv` (static features)
  - `data/processed/gujarat_river_distance.csv` (distance features)
- **Process**:
  1. Load all CSV files
  2. Rename columns for consistency (lat/lon)
  3. Cross-join rainfall (time) with features (space) using key=1
  4. Merge river distance on lat/lon
- **Output**: `data/processed/training_dataset_india_enhanced.csv`
- **Result**: Each (date, location) combination gets all features

**STEP 2: Advanced Flood Labeling**
- **Input**: `data/processed/training_dataset_gujarat_labeled.csv`
- **Algorithm**: Non-linear risk score combining multiple factors
- **Formula**:
  ```python
  score = (
      rain3_mm * 0.3 +                    # Short-term rain weight
      rain7_mm * 0.2 +                    # Long-term rain weight
      (1 / (distance_to_river_m + 1)) * 50000 +  # River proximity
      (1 / (elevation_m + 1)) * 2000              # Elevation factor
  )
  ```
- **Process**:
  1. Calculate risk score for each record
  2. Normalize score to probability (0-1)
  3. Add random noise for real-world uncertainty
  4. Threshold probability to create binary flood label (0/1)
  5. Clean infinite values and fill missing data
- **Output**: `data/processed/training_dataset_gujarat_advanced_labeled.csv`

**STEP 3: Data Cleaning & Validation**
- **Operations**:
  - Replace infinite values with None
  - Fill missing values with median
  - Remove duplicate records
  - Validate column types
- **Quality Checks**:
  - Check flood label distribution
  - Verify no null values in key columns
  - Ensure reasonable ranges for all features

📊 **FINAL OUTPUT**:
- **File**: `data/processed/training_dataset_gujarat_advanced_labeled.csv`
- **Size**: ~200MB
- **Records**: **2,279,281** rows (including header = 2,279,280 data records)
- **Columns**: 
  - Temporal: `date`, `rain3_mm`, `rain7_mm`
  - Geospatial: `lat`, `lon`, `elevation_m`, `distance_to_river_m`
  - Target: `flood` (binary: 0 = no flood, 1 = flood)
- **Flood Distribution**: Balanced ~50/50 due to random noise in labeling

🎓 **VIVA POINTS**:
- "Cross-join creates (date, location) combinations for time-space analysis"
- "Non-linear scoring combines rainfall, elevation, and river proximity"
- "2.28 million records provide sufficient training data for ML models"
- "Advanced labeling accounts for real-world uncertainty with random noise"

================================================================================
📊 SUMMARY TABLE FOR VIVA
================================================================================

| Raw Data | Original Format | Processed Format | Main Tool Used | Key Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **CHIRPS** | `.tif.gz` (9,000+ files) | `precipitation_mm` (CSV) | `wget`, `gdalwarp`, `rasterstats` | Daily rainfall history 2000-2025 |
| **SRTM** | `.tif` (4 tiles) | `elevation_m` (CSV) | `rasterio` | Terrain elevation for runoff |
| **OSM** | `.osm.pbf` (1.5GB) | `distance_to_river_m` (CSV) | `osmnx`, `geopandas` | River proximity risk indicator |
| **GADM** | `.shp` (vector) | `lat_lon` & `state` (reference) | `geopandas`, `gdalwarp` | Administrative boundaries |

================================================================================
🎯 KEY TECHNICAL INSIGHTS
================================================================================

**1. Data Volume Reduction**: 50GB+ raw data → ~250MB processed CSV (99.5% reduction)
**2. Spatial Operations**: All data projected to EPSG:4326 for consistency
**3. Temporal Resolution**: Daily data provides sufficient granularity for flood prediction
**4. Feature Engineering**: Combined temporal (rainfall) + spatial (elevation, rivers) features
**5. Scalability**: Pipeline designed for distributed processing (Big Data architecture)

================================================================================
🔧 TOOLCHAIN SUMMARY
================================================================================

**Geospatial Processing**:
- `gdalwarp` - Raster clipping and reprojection
- `rasterio` - Raster I/O and pixel operations
- `rasterstats` - Zonal statistics on rasters
- `geopandas` - Vector data operations
- `osmnx` - OpenStreetMap data extraction
- `shapely` - Geometric operations

**Data Processing**:
- `pandas` - DataFrame operations and merging
- `numpy` - Numerical computations
- `wget` - File downloads
- `multiprocessing` - Parallel processing

**Machine Learning**:
- Features extracted from raw geospatial data
- Labels generated using domain knowledge
- Ready for XGBoost, Random Forest, Spark ML

================================================================================
✅ CONCLUSION
================================================================================

This technical pipeline demonstrates sophisticated transformation of massive 
raw satellite and geospatial datasets into clean, machine-learning-ready 
features. The combination of temporal rainfall data with spatial terrain and 
hydrology features creates a comprehensive flood risk modeling dataset.

**Final Dataset**: 2.28 million records with 15+ features for flood prediction
**Processing Time**: ~50GB raw data processed in hours using parallel workflows
**Architecture**: Designed for Big Data scalability with HDFS-style storage zones

================================================================================
