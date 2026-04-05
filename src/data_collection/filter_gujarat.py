import pandas as pd

input_path = "data/processed/training_dataset_india_enhanced.csv"
output_path = "data/processed/training_dataset_gujarat.csv"

df = pd.read_csv(input_path)

print("Original shape:", df.shape)

# Data already Gujarat → no filter needed
df_gujarat = df.copy()

print("Gujarat shape:", df_gujarat.shape)

df_gujarat.to_csv(output_path, index=False)

print("Saved to:", output_path)
