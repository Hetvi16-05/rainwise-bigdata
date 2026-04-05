import pandas as pd

path = "data/processed/gujarat_features.csv"

df = pd.read_csv(path)

print("Total rows:", len(df))

print("\nFlood counts:")
print(df["flood"].value_counts())

print("\nFlood percentage:")
print(df["flood"].value_counts(normalize=True) * 100)