import pandas as pd

print("📥 Loading rainfall data...")

df = pd.read_csv("data/processed/gujarat_rainfall_history.csv")

# Rename
df = df.rename(columns={"rain_mm": "rainfall"})

print("🧠 Creating features...")

df = df.sort_values(by=["lat","lon","date"])

# ✅ FIXED rolling (IMPORTANT)
df['rain_1day'] = df['rainfall']

df['rain_3day'] = df.groupby(['lat','lon'])['rainfall']\
    .transform(lambda x: x.rolling(3, min_periods=1).sum())

df['rain_7day'] = df.groupby(['lat','lon'])['rainfall']\
    .transform(lambda x: x.rolling(7, min_periods=1).sum())

print("🌊 Creating flood label...")

df['flood'] = (
    (df['rain_3day'] > 60) |
    (df['rain_7day'] > 120)
).astype(int)

print("🧹 Cleaning...")

df = df.dropna()

df = df[[
    'rain_1day',
    'rain_3day',
    'rain_7day',
    'flood'
]]

print("💾 Saving...")

df.to_csv("data/processed/model_final_working.csv", index=False)

print("✅ DONE!")