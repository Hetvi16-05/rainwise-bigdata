import pandas as pd

input_file = "data/processed/final_features_complete.csv"
output_file = "data/processed/gujarat_features_geo.csv"

df = pd.read_csv(input_file)

print("Total rows:", len(df))

# Gujarat bounds
df_gj = df[
    (df["latitude"] >= 20) &
    (df["latitude"] <= 25) &
    (df["longitude"] >= 68) &
    (df["longitude"] <= 75)
]

print("Gujarat rows:", len(df_gj))

# rename columns to match rainfall file
df_gj = df_gj.rename(columns={
    "latitude": "lat",
    "longitude": "lon",
    "elevation_m": "elevation",
    "distance_to_river_m": "river_distance",
    "slope_degree": "slope",
    "land_cover_class": "landcover"
})

df_gj = df_gj[[
    "lat",
    "lon",
    "elevation",
    "slope",
    "river_distance",
    "landcover"
]]

df_gj.to_csv(output_file, index=False)

print("Saved:", output_file)