import pandas as pd

# Load features
elevation = pd.read_csv("data/processed/features/watershed_elevation.csv")
slope = pd.read_csv("data/processed/features/watershed_slope.csv")
rainfall = pd.read_csv("data/processed/features/watershed_rainfall.csv")
river_density = pd.read_csv("data/processed/hydrology/gujarat/features/gujarat_river_density.csv")

# Merge by index (all are 412 rows in same order)
df = pd.concat([elevation, slope, rainfall, river_density["river_density"]], axis=1)

df.to_csv("data/processed/features/final_features.csv", index=False)

print("✅ Final feature table created!")