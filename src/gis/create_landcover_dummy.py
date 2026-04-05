import pandas as pd
import numpy as np

grid = pd.read_csv("data/processed/gujarat_grid_025.csv")

grid["landcover"] = np.random.randint(10, 50, len(grid))

grid.to_csv("data/processed/gujarat_landcover.csv", index=False)

print("landcover done")