import pandas as pd
files = [
    "data/processed/gujarat_rainfall_history.csv",
    "data/processed/gujarat_features.csv",
]
for f in files:
    print("\nFile:", f)
    df = pd.read_csv(f, nrows=0)
    
    print(df.columns.tolist())