import pandas as pd

input_path = "data/processed/training_dataset_gujarat.csv"
output_path = "data/processed/training_dataset_gujarat_labeled.csv"

df = pd.read_csv(input_path)

print("Before:", df["flood"].value_counts())


# convert to mm (if needed)
df["precip_mm"] = df["precipitation_mm"] * 1000
df["rain3_mm"] = df["rain_3day"] * 1000
df["rain7_mm"] = df["rain_7day"] * 1000


# create new flood label
def create_flood(row):
    if row["rain7_mm"] > 100:
        return 1
    elif row["rain3_mm"] > 50:
        return 1
    else:
        return 0


df["flood_new"] = df.apply(create_flood, axis=1)


print("After:", df["flood_new"].value_counts())


df.to_csv(output_path, index=False)

print("Saved:", output_path)