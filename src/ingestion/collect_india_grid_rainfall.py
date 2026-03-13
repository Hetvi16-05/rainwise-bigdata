import pandas as pd
import numpy as np
import os

OUTPUT_FILE = "data/raw/static/india_grid.csv"

# India bounding box
min_lat, max_lat = 6, 37
min_lon, max_lon = 68, 97

resolution = 2  # 2-degree grid (fast demo version)

latitudes = np.arange(min_lat, max_lat + 1, resolution)
longitudes = np.arange(min_lon, max_lon + 1, resolution)

grid = [(lat, lon) for lat in latitudes for lon in longitudes]

df = pd.DataFrame(grid, columns=["latitude", "longitude"])

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("✅ India grid created")
print("Total Grid Points:", len(df))