import pandas as pd

old_path = "data/processed/gujarat_features.csv"
new_path = "data/processed/gujarat_features_new.csv"

output_path = "data/processed/gujarat_features.csv"


print("Loading old file...")
df_old = pd.read_csv(old_path)

print("Loading new file...")
df_new = pd.read_csv(new_path)

print("Old rows:", len(df_old))
print("New rows:", len(df_new))


# merge
df = pd.concat([df_old, df_new], ignore_index=True)


print("After concat:", len(df))


# remove duplicates using city + date
df = df.drop_duplicates(subset=["city", "date"])


print("After removing duplicates:", len(df))


# sort for safety
df = df.sort_values(["city", "date"])


# save back
df.to_csv(output_path, index=False)

print("Saved merged file:", output_path)