import pandas as pd

input_path = "data/processed/training_dataset_gujarat.csv"
output_path = "data/processed/training_dataset_gujarat_labeled.csv"

df = pd.read_csv(input_path)

print("Shape:", df.shape)

# convert to mm if needed
if "precipitation_mm" in df.columns:
    df["precip_mm"] = df["precipitation_mm"] * 1000
else:
    df["precip_mm"] = df["rain_mm"] * 1000

if "rain_3day" in df.columns:
    df["rain3_mm"] = df["rain_3day"] * 1000
else:
    df["rain3_mm"] = df["precip_mm"]

if "rain_7day" in df.columns:
    df["rain7_mm"] = df["rain_7day"] * 1000
else:
    df["rain7_mm"] = df["precip_mm"]

def create_flood(row):
    if row["rain7_mm"] > 100:
        return 1
    elif row["rain3_mm"] > 50:
        return 1
    else:
        return 0


df["flood"] = df.apply(create_flood, axis=1)

print(df["flood"].value_counts())

df.to_csv(output_path, index=False)

print("Saved:", output_path)
