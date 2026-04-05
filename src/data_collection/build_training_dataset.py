import pandas as pd

rain = pd.read_csv("data/processed/gujarat_rainfall_history.csv")
feat = pd.read_csv("data/processed/gujarat_features.csv")
river = pd.read_csv("data/processed/gujarat_river_distance.csv")

print("Rain:", rain.shape)
print("Feat:", feat.shape)
print("River:", river.shape)

# rename feature columns
feat = feat.rename(
    columns={
        "latitude": "lat",
        "longitude": "lon"
    }
)

# repeat features for each date
rain["key"] = 1
feat["key"] = 1

df = rain.merge(feat, on="key").drop("key", axis=1)

print("After rain+feat:", df.shape)
print(df.columns)

# fix column names after merge
if "lat_y" in df.columns:
    df["lat"] = df["lat_y"]

if "lon_y" in df.columns:
    df["lon"] = df["lon_y"]

# merge river
df = df.merge(
    river,
    on=["lat", "lon"],
    how="left"
)

print("After river:", df.shape)

df.to_csv(
    "data/processed/training_dataset_india_enhanced.csv",
    index=False
)

print("Saved training_dataset_india_enhanced.csv")
