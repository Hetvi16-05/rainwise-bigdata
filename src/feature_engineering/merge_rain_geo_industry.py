import pandas as pd

rain_file = "data/processed/gujarat_features.csv"
geo_file = "data/processed/gujarat_features_geo.csv"

out_file = "data/processed/training_dataset_gujarat_geo.csv"

rain = pd.read_csv(rain_file)
geo = pd.read_csv(geo_file)

print("Rain rows:", len(rain))
print("Geo rows:", len(geo))

# merge using nearest lat lon (rounding)
rain["lat"] = rain["lat"].round(2)
rain["lon"] = rain["lon"].round(2)

geo["lat"] = geo["lat"].round(2)
geo["lon"] = geo["lon"].round(2)

merged = rain.merge(
    geo,
    on=["lat", "lon"],
    how="left"
)

print("Merged rows:", len(merged))

merged.to_csv(out_file, index=False)

print("Saved:", out_file)