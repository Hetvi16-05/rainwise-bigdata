import pandas as pd
import numpy as np
import joblib
import logging
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
    accuracy_score
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression

# 🔥 IMPORTANT IMPORT (NO PICKLE ERROR)
from src.utils.features import feature_engineering

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# =========================
# LOAD DATA
# =========================
logger.info("📂 Loading dataset...")

df = pd.read_csv(
    "data/processed/training_dataset_gujarat_advanced_labeled.csv",
    low_memory=False
)

df.columns = df.columns.str.lower()
logger.info(f"Rows: {len(df)}")

# =========================
# USE ORIGINAL rain_mm (DAILY RAINFALL 0-93mm)
# =========================
logger.info(f"rain_mm range: {df['rain_mm'].min():.2f} - {df['rain_mm'].max():.2f}")
logger.info(f"Flood rate: {df['flood'].mean()*100:.2f}%")

# =========================
# FEATURES (RAW INPUT ONLY)
# =========================
features = [
    "rain_mm",
    "elevation_m",
    "distance_to_river_m",
    "lat",
    "lon"
]

target = "flood"

df = df[features + [target]].dropna()

X = df[features].values
y = df[target].values

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# CLASS DISTRIBUTION
# =========================
logger.info(f"Class distribution — No flood: {(y_train==0).sum()}, Flood: {(y_train==1).sum()}")

# =========================
# PIPELINE MODEL (Logistic Regression with StandardScaler)
# =========================
logger.info("🚀 Building Logistic Regression pipeline model...")

model = Pipeline([
    ("feature_engineering", FunctionTransformer(feature_engineering)),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    ))
])

# =========================
# TRAIN
# =========================
logger.info("🏋️ Training Logistic Regression model...")
model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
logger.info("📊 Evaluating model...")

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
y_test_proba = model.predict_proba(X_test)[:, 1]

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
auc = roc_auc_score(y_test, y_test_proba)

logger.info(f"Train Accuracy: {train_acc:.4f}")
logger.info(f"Test Accuracy: {test_acc:.4f}")
logger.info(f"ROC-AUC: {auc:.4f}")

logger.info("\n📄 Classification Report:\n" +
            classification_report(y_test, y_test_pred))

# =========================
# SPOT CHECK: verify predictions make sense
# =========================
logger.info("🔍 Spot-checking predictions at different rainfall levels...")
test_cases = [
    ("Low rain (5mm)", [5.0, 28.0, 93224.0, 20.37, 72.91]),
    ("Moderate rain (25mm)", [25.0, 28.0, 93224.0, 20.37, 72.91]),
    ("Heavy rain (50mm)", [50.0, 28.0, 93224.0, 20.37, 72.91]),
    ("Very heavy rain (80mm)", [80.0, 28.0, 93224.0, 20.37, 72.91]),
    ("Heavy rain + close to river", [50.0, 10.0, 500.0, 21.2, 72.8]),
    ("Heavy rain + low elevation", [60.0, 5.0, 2000.0, 21.2, 72.8]),
]

for label, feats in test_cases:
    p = model.predict_proba(np.array([feats]))[0][1]
    logger.info(f"  {label}: {p:.3f}")

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_test_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.savefig("outputs/flood_logistic_confusion_matrix.png")
plt.close()

# =========================
# ROC CURVE
# =========================
fpr, tpr, _ = roc_curve(y_test, y_test_proba)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.legend()
plt.title("Logistic Regression ROC Curve")
plt.savefig("outputs/flood_logistic_roc_curve.png")
plt.close()

# =========================
# FEATURE IMPORTANCE (Coefficients)
# =========================
importances = model.named_steps["classifier"].coef_[0]

feature_names = [
    "rain3", "rain7", "rain_intensity", "rain_ratio",
    "river_risk", "drainage_factor", "elevation", "distance", "lat", "lon"
]

indices = np.argsort(np.abs(importances))[::-1]

plt.figure()
sns.barplot(
    x=np.abs(importances[indices]),
    y=np.array(feature_names)[indices]
)
plt.title("Logistic Regression Feature Importance (Absolute Coefficients)")
plt.savefig("outputs/flood_logistic_feature_importance.png")
plt.close()

# =========================
# SAVE MODEL
# =========================
joblib.dump(model, "models/flood_logistic_model.pkl")

logger.info("✅ Logistic Regression pipeline model saved successfully!")
