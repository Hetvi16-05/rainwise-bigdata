import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE

# =========================
# LOAD DATA
# =========================
df = pd.read_parquet("data/bigdata/final.parquet")

print("📊 Rows:", len(df))

# =========================
# FEATURE ENGINEERING (STRONGER)
# =========================
df["rain_intensity"] = df["rain7_mm"] / 7
df["rain_ratio"] = df["rain3_mm"] / (df["rain7_mm"] + 1)

# 🔥 NEW IMPORTANT FEATURES
df["heavy_rain_flag"] = (df["rain3_mm"] > 150).astype(int)
df["extreme_rain_flag"] = (df["rain7_mm"] > 300).astype(int)
df["river_risk"] = 1 / (df["distance_to_river_m"] + 1)

# =========================
# FEATURES
# =========================
features = [
    "rain3_mm",
    "rain7_mm",
    "rain_intensity",
    "rain_ratio",
    "heavy_rain_flag",
    "extreme_rain_flag",
    "river_risk",
    "elevation_m",
    "distance_to_river_m",
    "lat",
    "lon",
    "nasa_avg_rain",
    "nasa_max_rain",
    "nasa_std_rain"
]

df = df[features + ["flood"]].dropna()

X = df[features]
y = df["flood"]

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 🔥 HANDLE IMBALANCE (CRITICAL)
# =========================
smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# =========================
# TRAIN MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# EVALUATE
# =========================
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)

print("🔥 AUC:", auc)

# =========================
# SAVE
# =========================
joblib.dump(model, "models/final_flood_model.pkl")

print("✅ Model saved")