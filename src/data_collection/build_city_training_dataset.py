import pandas as pd
import numpy as np

train = pd.read_csv(
    "data/processed/training_dataset_gujarat_labeled.csv"
)

cities = pd.read_csv(
    "data/config/gujarat_cities.csv"
)


def nearest(lat, lon, df):

    d = (
        (df["lat"] - lat) ** 2 +
        (df["lon"] - lon) ** 2
    )

    i = d.idxmin()

    return df.loc[i]


rows = []

for _, c in cities.iterrows():

    r = nearest(
        c["lat"],
        c["lon"],
        train
    )

    row = r.to_dict()

    row["city"] = c["city"]

    rows.append(row)


df = pd.DataFrame(rows)

df.to_csv(
    "data/processed/training_dataset_gujarat_city.csv",
    index=False
)

print("Saved city dataset")
