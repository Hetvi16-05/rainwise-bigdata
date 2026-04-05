import pandas as pd
import geopandas as gpd

# ==============================
# Load India state names
# ==============================
states_shp = "data/raw/boundary/gadm41_IND_1.shp"
states_gdf = gpd.read_file(states_shp)

state_names = states_gdf["NAME_1"].unique().tolist()

# Handle old names
state_alias = {
    "Orissa": "Odisha",
    "Uttaranchal": "Uttarakhand"
}

# ==============================
# Load flood events
# ==============================
flood_df = pd.read_csv("data/processed/flood_labels_clean.csv")

flood_df["date"] = pd.to_datetime(flood_df["date"])

records = []

# ==============================
# Match state names in Location text
# ==============================
for _, row in flood_df.iterrows():
    location_text = str(row["Location"])

    # Replace aliases
    for old, new in state_alias.items():
        location_text = location_text.replace(old, new)

    for state in state_names:
        if state in location_text:
            records.append({
                "date": row["date"],
                "state": state,
                "flood": 1
            })

# Create cleaned state-level flood label dataset
clean_labels = pd.DataFrame(records)

# Remove duplicates
clean_labels = clean_labels.drop_duplicates()

clean_labels.to_csv(
    "data/processed/flood_labels_state_level.csv",
    index=False
)

print("✅ Clean state-level flood labels created")
print("Total flood rows:", len(clean_labels))
print("Unique flood states:", clean_labels["state"].nunique())