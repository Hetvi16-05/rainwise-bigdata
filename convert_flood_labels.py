import pandas as pd

# 1️⃣ Read the downloaded EM-DAT file
file_path = "data/raw/flood_labels/public_emdat_custom_request_2026-02-23_87a30cde-509d-4987-96aa-4e7ea9fd242f.xlsx"
df = pd.read_excel(file_path)

# 2️⃣ Keep only Flood disasters
df = df[df["Disaster Type"] == "Flood"]

# 3️⃣ Select required columns
df = df[["Start Year", "Start Month", "Start Day", "Country", "Location"]]

# 4️⃣ Create proper date column
# Fill missing month/day with 1
df["Start Month"] = df["Start Month"].fillna(1)
df["Start Day"] = df["Start Day"].fillna(1)

df["date"] = pd.to_datetime(
    df["Start Year"].astype(int).astype(str) + "-" +
    df["Start Month"].astype(int).astype(str) + "-" +
    df["Start Day"].astype(int).astype(str),
    format="%Y-%m-%d",
    errors="coerce"
)

# 5️⃣ Create flood label
df["flood"] = 1

# 6️⃣ Keep only final columns
df = df[["date", "Location", "flood"]]

# 7️⃣ Save cleaned file
df.to_csv("data/processed/flood_labels_clean.csv", index=False)

print("Flood labels cleaned and saved successfully ✅")