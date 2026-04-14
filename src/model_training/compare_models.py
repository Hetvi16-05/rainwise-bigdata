import pandas as pd
import numpy as np
import joblib
import logging
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)

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
# FEATURES
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

logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
logger.info(f"Class distribution — No flood: {(y_train==0).sum()}, Flood: {(y_train==1).sum()}")

# =========================
# LOAD MODELS
# =========================
logger.info("📂 Loading trained models...")

models = {}
model_names = ["XGBoost", "SVM", "Logistic Regression"]
model_files = [
    "models/flood_model.pkl",
    "models/flood_svm_model.pkl",
    "models/flood_logistic_model.pkl"
]

for name, file in zip(model_names, model_files):
    if os.path.exists(file):
        logger.info(f"  Loading {name} from {file}...")
        models[name] = joblib.load(file)
    else:
        logger.warning(f"  {name} not found at {file}. Skipping.")

if not models:
    logger.error("❌ No models found! Please train models first.")
    exit(1)

# =========================
# EVALUATE ALL MODELS
# =========================
logger.info("📊 Evaluating all models...")

results = []

for name, model in models.items():
    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluating: {name}")
    logger.info(f"{'='*50}")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    auc = roc_auc_score(y_test, y_test_proba)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    
    logger.info(f"Train Accuracy: {train_acc:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"ROC-AUC: {auc:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")
    
    # Store results
    results.append({
        "Model": name,
        "Train Accuracy": train_acc,
        "Test Accuracy": test_acc,
        "ROC-AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

# =========================
# COMPARISON TABLE
# =========================
logger.info("\n" + "="*70)
logger.info("📊 MODEL COMPARISON TABLE")
logger.info("="*70)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("ROC-AUC", ascending=False)

print("\n" + results_df.to_string(index=False))

# Save comparison to CSV
results_df.to_csv("outputs/model_comparison.csv", index=False)
logger.info("\n💾 Comparison saved to outputs/model_comparison.csv")

# =========================
# FIND BEST MODEL
# =========================
best_model = results_df.iloc[0]
logger.info("\n" + "="*70)
logger.info(f"🏆 BEST MODEL: {best_model['Model']}")
logger.info("="*70)
logger.info(f"ROC-AUC: {best_model['ROC-AUC']:.4f}")
logger.info(f"Test Accuracy: {best_model['Test Accuracy']:.4f}")
logger.info(f"F1-Score: {best_model['F1-Score']:.4f}")

# =========================
# SPOT CHECK COMPARISON
# =========================
logger.info("\n" + "="*70)
logger.info("🔍 SPOT CHECK: Predictions at different rainfall levels")
logger.info("="*70)

test_cases = [
    ("Low rain (5mm)", [5.0, 28.0, 93224.0, 20.37, 72.91]),
    ("Moderate rain (25mm)", [25.0, 28.0, 93224.0, 20.37, 72.91]),
    ("Heavy rain (50mm)", [50.0, 28.0, 93224.0, 20.37, 72.91]),
    ("Very heavy rain (80mm)", [80.0, 28.0, 93224.0, 20.37, 72.91]),
    ("Heavy rain + close to river", [50.0, 10.0, 500.0, 21.2, 72.8]),
    ("Heavy rain + low elevation", [60.0, 5.0, 2000.0, 21.2, 72.8]),
]

for label, feats in test_cases:
    logger.info(f"\n{label}:")
    for name, model in models.items():
        p = model.predict_proba(np.array([feats]))[0][1]
        logger.info(f"  {name}: {p:.3f}")

# =========================
# RECOMMENDATION
# =========================
logger.info("\n" + "="*70)
logger.info("💡 RECOMMENDATION")
logger.info("="*70)

if best_model['Model'] == "XGBoost":
    logger.info("✅ XGBoost shows the best performance. It's recommended for production.")
    logger.info("   Reasons: High ROC-AUC, good balance of precision and recall.")
elif best_model['Model'] == "SVM":
    logger.info("✅ SVM shows the best performance. It's recommended for production.")
    logger.info("   Reasons: Strong ROC-AUC, robust to overfitting.")
else:
    logger.info("✅ Logistic Regression shows the best performance. It's recommended for production.")
    logger.info("   Reasons: High interpretability, good baseline performance.")

logger.info("\n🎉 Model comparison complete!")
