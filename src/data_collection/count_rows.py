import pandas as pd

files = [
    "data/processed/gujarat_features_new.csv",
    "data/processed/gujarat_features.csv"
]

for f in files:
    try:
        df = pd.read_csv(f)
        print(f, "->", len(df))
    except:
        print(f, "not found")