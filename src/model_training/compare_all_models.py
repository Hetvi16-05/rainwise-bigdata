import pandas as pd
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

# Regression Algorithms
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor

# Classification Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Project imports
from src.utils.features import feature_engineering

# ================================================================
# LOGGING & SETUP
# ================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()
os.makedirs("outputs", exist_ok=True)

# ================================================================
# LOAD DATA
# ================================================================
logger.info("📂 Loading dataset...")
DATA_PATH = "data/processed/training_dataset_gujarat_advanced_labeled.csv"
df = pd.read_csv(DATA_PATH, low_memory=False)
df.columns = df.columns.str.lower()

# Sample for speed (SVM and kNN are slow)
SAMPLE_SIZE = 20000
if len(df) > SAMPLE_SIZE:
    df = df.sample(n=SAMPLE_SIZE, random_state=42)
logger.info(f"Using {len(df)} samples for comparison.")

# ================================================================
# PREPROCESS DATA
# ================================================================
logger.info("🛠️ Preprocessing data and generating features...")

# 1. Extract Month from YYYYMMDD date format
if 'date' in df.columns:
    df['month'] = (df['date'] // 100) % 100
else:
    df['month'] = datetime.now().month

# 2. Generate Synthetic Atmospheric Features (matching train_rainfall_model.py logic)
np.random.seed(42)
rain = df["rain_mm"].values

# Temperature (°C)
base_temp = np.random.uniform(28, 42, size=len(rain))
rain_cooling = np.clip(rain * 0.15, 0, 12)
df["temperature"] = base_temp - rain_cooling + np.random.normal(0, 1.5, size=len(rain))

# Humidity (%)
base_humidity = np.random.uniform(30, 60, size=len(rain))
rain_humidity_boost = np.clip(rain * 1.5, 0, 40)
df["humidity"] = base_humidity + rain_humidity_boost + np.random.normal(0, 5, size=len(rain))

# Pressure (hPa)
base_pressure = np.random.uniform(1008, 1020, size=len(rain))
rain_pressure_drop = np.clip(rain * 0.3, 0, 20)
df["pressure"] = base_pressure - rain_pressure_drop + np.random.normal(0, 2, size=len(rain))

# Wind Speed (km/h)
base_wind = np.random.uniform(5, 15, size=len(rain))
rain_wind_boost = np.clip(rain * 0.5, 0, 30)
df["wind_speed"] = base_wind + rain_wind_boost + np.random.normal(0, 3, size=len(rain))

# Cloud Cover (%)
base_cloud = np.random.uniform(10, 40, size=len(rain))
rain_cloud_boost = np.clip(rain * 2.0, 0, 60)
df["cloud_cover"] = base_cloud + rain_cloud_boost + np.random.normal(0, 5, size=len(rain))

# ================================================================
# TASK 1: RAINFALL PREDICTION (REGRESSION)
# ================================================================
logger.info("🌧 Phase 1: Comparing Rainfall Regression Models...")

# Features: month, temp, humidity, pressure, wind, clouds
reg_features = ["month", "temperature", "humidity", "pressure", "wind_speed", "cloud_cover"]
reg_target = "rain_mm"

df_reg = df[reg_features + [reg_target]].dropna()
X_reg = df_reg[reg_features].values
y_reg = df_reg[reg_target].values

Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg_models = {
    "Linear Regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1),
    "k-Nearest Neighbors": Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor(n_neighbors=5))])
}

reg_results = []

for name, model in reg_models.items():
    logger.info(f"  Training {name}...")
    model.fit(Xr_train, yr_train)
    preds = model.predict(Xr_test)
    
    r2 = r2_score(yr_test, preds)
    mae = mean_absolute_error(yr_test, preds)
    rmse = np.sqrt(mean_squared_error(yr_test, preds))
    
    reg_results.append({
        "Model": name,
        "R2 Score": r2,
        "MAE": mae,
        "RMSE": rmse
    })

reg_summary = pd.DataFrame(reg_results).sort_values("R2 Score", ascending=False)
logger.info("\nRainfall Regression Performance:\n" + reg_summary.to_string(index=False))
reg_summary.to_csv("outputs/model_comparison_regression.csv", index=False)

# ================================================================
# TASK 2: FLOOD RISK (CLASSIFICATION)
# ================================================================
logger.info("🌊 Phase 2: Comparing Flood Classification Models...")

# Features: rain_mm, elevation_m, distance_to_river_m, lat, lon (before engineering)
clf_features = ["rain_mm", "elevation_m", "distance_to_river_m", "lat", "lon"]
clf_target = "flood"

df_clf = df[clf_features + [clf_target]].dropna()
X_clf = df_clf[clf_features].values
y_clf = df_clf[clf_target].values

Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

# Common pipeline element: Feature Engineering
def get_clf_pipeline(model):
    return Pipeline([
        ("engineering", FunctionTransformer(feature_engineering)),
        ("scaler", StandardScaler()),
        ("model", model)
    ])

clf_models = {
    "Logistic Regression": get_clf_pipeline(LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
    "Naive Bayes": get_clf_pipeline(GaussianNB()),
    "Decision Tree": get_clf_pipeline(DecisionTreeClassifier(max_depth=10, random_state=42)),
    "Random Forest": get_clf_pipeline(RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)),
    "XGBoost": get_clf_pipeline(XGBClassifier(n_estimators=100, scale_pos_weight=10, use_label_encoder=False, eval_metric='logloss', random_state=42)),
    "SVM": get_clf_pipeline(SVC(probability=True, kernel='rbf', C=1.0, class_weight='balanced', random_state=42)),
    "k-Nearest Neighbors": get_clf_pipeline(KNeighborsClassifier(n_neighbors=5))
}

clf_results = []
plt.figure(figsize=(10, 8))

for name, model in clf_models.items():
    logger.info(f"  Training {name}...")
    model.fit(Xc_train, yc_train)
    preds = model.predict(Xc_test)
    probas = model.predict_proba(Xc_test)[:, 1]
    
    acc = accuracy_score(yc_test, preds)
    prec = precision_score(yc_test, preds, zero_division=0)
    rec = recall_score(yc_test, preds, zero_division=0)
    f1 = f1_score(yc_test, preds, zero_division=0)
    auc = roc_auc_score(yc_test, probas)
    
    clf_results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc
    })
    
    # ROC Curve Plotting
    fpr, tpr, _ = roc_curve(yc_test, probas)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

# Finalize ROC plot
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Flood Models ROC Curves Comparison')
plt.legend()
plt.savefig("outputs/flood_models_roc_comparison.png")
plt.close()

clf_summary = pd.DataFrame(clf_results).sort_values("ROC-AUC", ascending=False)
logger.info("\nFlood Classification Performance:\n" + clf_summary.to_string(index=False))
clf_summary.to_csv("outputs/model_comparison_classification.csv", index=False)

# ================================================================
# VISUALIZATIONS
# ================================================================
logger.info("📊 Generating visualization charts...")

# Regression R2 Plot
plt.figure(figsize=(10, 6))
sns.barplot(x="R2 Score", y="Model", data=reg_summary, palette="viridis")
plt.title("Rainfall Prediction (Regression) - R2 Score Comparison")
plt.xlim(0, 1)
plt.savefig("outputs/regression_r2_comparison.png")

# Classification ROC-AUC Plot
plt.figure(figsize=(10, 6))
sns.barplot(x="ROC-AUC", y="Model", data=clf_summary, palette="magma")
plt.title("Flood Risk Prediction (Classification) - ROC-AUC Comparison")
plt.xlim(0.5, 1)
plt.savefig("outputs/classification_auc_comparison.png")

logger.info(f"✅ Comparison complete! Check files in 'outputs/' and CSVs.")
