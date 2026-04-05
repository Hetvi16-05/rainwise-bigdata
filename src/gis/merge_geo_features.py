import pandas as pd

grid = pd.read_csv("data/processed/gujarat_grid_025.csv")
elev = pd.read_csv("data/processed/gujarat_elevation.csv")
slope = pd.read_csv("data/processed/gujarat_slope.csv")
river = pd.read_csv("data/processed/gujarat_river_distance.csv")
land = pd.read_csv("data/processed/gujarat_landcover.csv")

df = grid.copy()

df["elevation"] = elev["elevation"]
df["slope"] = slope["slope"]
df["river_distance"] = river["river_distance"]
df["landcover"] = land["landcover"]

df.to_csv(
    "data/processed/gujarat_features_geo.csv",
    index=False
)

print("Saved gujarat_features_geo.csv")