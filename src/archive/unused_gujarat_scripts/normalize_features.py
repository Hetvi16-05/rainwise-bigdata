import pandas as pd

# Load data
slope = pd.read_csv("data/processed/features/watershed_slope.csv")
distance = pd.read_csv("data/processed/features/watershed_distance.csv")
river_density = pd.read_csv("data/processed/features/gujarat_river_density.csv")

# Merge
df = pd.concat([slope, distance, river_density], axis=1)

# Min-Max Normalization
def normalize(col):
    return (col - col.min()) / (col.max() - col.min())

df["slope_norm"] = normalize(df["avg_slope"])
df["distance_norm"] = normalize(df["avg_distance"])
df["river_density_norm"] = normalize(df["river_density"])

# Inverse distance (closer river = higher risk)
df["distance_inverse_norm"] = 1 - df["distance_norm"]

df.to_csv("data/processed/features/watershed_features_normalized.csv", index=False)

print("✅ Normalization complete!")