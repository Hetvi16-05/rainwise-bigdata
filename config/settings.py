# ===============================
# PROJECT CONFIGURATION
# ===============================

# City Information
CITY_NAME = "Vadodara"
LATITUDE = 22.3072
LONGITUDE = 73.1812
TIMEZONE = "Asia/Kolkata"

# ===============================
# API CONFIGURATION
# ===============================

OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"

# ===============================
# DATA PATHS
# ===============================

RAW_REALTIME_PATH = "data/raw/realtime"
COMBINED_REALTIME_PATH = "data/raw/realtime/combined"
STATIC_DATA_PATH = "data/raw/static"
PROCESSED_DATA_PATH = "data/processed"

# ===============================
# FLOOD THRESHOLDS (Editable)
# ===============================

HEAVY_RAIN_THRESHOLD_MM = 50
RIVER_DANGER_LEVEL_M = 6.0
SOIL_SATURATION_THRESHOLD = 0.8
