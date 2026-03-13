import pandas as pd
import numpy as np

out = "data/processed/gujarat_grid_025.csv"

lat_vals = np.arange(20, 25, 0.25)
lon_vals = np.arange(68, 75, 0.25)

rows = []

for lat in lat_vals:
    for lon in lon_vals:
        rows.append([lat, lon])

df = pd.DataFrame(rows, columns=["lat", "lon"])

print("Points:", len(df))

df.to_csv(out, index=False)

print("Saved:", out)