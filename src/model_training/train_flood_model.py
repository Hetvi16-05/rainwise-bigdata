import pandas as pd
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import precision_recall_curve

import xgboost as xgb

# ----------------------
# paths
# ----------------------
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "data/processed/training_dataset_gujarat_advanced_labeled.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_DIR = os.path.join(BASE_DIR, "plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ----------------------
# load data
# ----------------------
print("📥 Loading dataset...")
df = pd.read_csv(DATA_FILE)

TARGET = "flood"

# =========================================================
# 🧠 FEATURE ENGINEERING (NEW FEATURES)
# =========================================================
print("\n🧠 Creating advanced features...")

df["rain_trend"] = df["rain3_mm"] - df["rain7_mm"]
df["rain_intensity"] = df["rain3_mm"] / 3

df["log_rain3"] = np.log1p(df["rain3_mm"])
df["log_rain7"] = np.log1p(df["rain7_mm"])

df["river_risk"] = df["rain3_mm"] / (df["distance_to_river_m"] + 1)

# features list
features = [
    "rain3_mm",
    "rain7_mm",
    "rain_trend",
    "rain_intensity",
    "log_rain3",
    "log_rain7",
    "river_risk",
    "elevation_m",
    "distance_to_river_m"
]

# =========================================================
# DATA PREPARATION
# =========================================================
X = df[features].copy()
y = df[TARGET]

# cleaning
X = X.replace([float("inf"), -float("inf")], np.nan)
X = X.fillna(X.median())

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# ⚡ TRAIN XGBOOST
# =========================================================
print("\n⚡ Training XGBoost...")

scale_pos_weight = (len(y) - sum(y)) / sum(y)

xgb_model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

# =========================================================
# 📊 DEFAULT METRICS
# =========================================================
print("\n📊 Default Threshold (0.5) Report:")

default_pred = xgb_model.predict(X_test)

print(classification_report(y_test, default_pred))
print("Accuracy:", accuracy_score(y_test, default_pred))
print("F1 Score:", f1_score(y_test, default_pred))

# =========================================================
# 🎯 SMART THRESHOLD TUNING
# =========================================================
print("\n🎯 Finding Best Balanced Threshold...")

proba = xgb_model.predict_proba(X_test)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test, proba)

best_thresh = 0.5
best_score = 0

for p, r, t in zip(precisions, recalls, thresholds):
    if r >= 0.70:
        score = p * r
        if score > best_score:
            best_score = score
            best_thresh = t

print(f"✅ Selected Threshold: {best_thresh:.4f}")

# =========================================================
# 📊 FINAL METRICS
# =========================================================
final_pred = (proba > best_thresh).astype(int)

print("\n📊 Final Tuned Model Report:")
print(classification_report(y_test, final_pred))

print("Final Accuracy:", accuracy_score(y_test, final_pred))
print("Final F1 Score:", f1_score(y_test, final_pred))

# =========================================================
# 💾 SAVE MODEL
# =========================================================
joblib.dump(xgb_model, os.path.join(MODEL_DIR, "flood_model_xgb.pkl"))
joblib.dump(best_thresh, os.path.join(MODEL_DIR, "threshold.pkl"))

print("\n💾 Model saved")
print("💾 Threshold saved")

# =========================================================
# 📊 FEATURE IMPORTANCE
# =========================================================
print("\n📊 Feature Importance...")

importance_df = pd.DataFrame({
    "feature": features,
    "importance": xgb_model.feature_importances_
}).sort_values(by="importance", ascending=False)

print(importance_df)

plt.figure()
plt.barh(importance_df["feature"], importance_df["importance"])
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.gca().invert_yaxis()

plot_path = os.path.join(PLOT_DIR, "feature_importance.png")
plt.savefig(plot_path)

print("📈 Saved:", plot_path)

print("\n✅ TRAINING COMPLETE")