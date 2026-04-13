import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import joblib

# Setup Directories
os.makedirs("bigdata_demo/plots", exist_ok=True)

# 1. Load Data (Using a sample of 50k rows for efficiency)
df = pd.read_csv("data/processed/training_dataset_gujarat_advanced_labeled.csv", nrows=50000)
df.columns = df.columns.str.lower()

# 2. Basic Feature Engineering
# (Simplified extraction for demonstration)
features = ["rain_mm", "elevation_m", "distance_to_river_m", "lat", "lon"]
target = "flood"

df = df[features + [target]].dropna()
X = df[features].values
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Training & Evaluation
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB()
}

results = []

print("\n🚀 Starting Multi-Algorithm Comparison for Flood Classification...")

for name, model in models.items():
    print(f"\n--- Evaluating {name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    results.append({"Model": name, "Accuracy": acc})
    
    # Save Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix: {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(f"bigdata_demo/plots/flood_cm_{name.lower().replace(' ', '_')}.png")
    plt.close()

# 4. Summary Output
report_df = pd.DataFrame(results)
print("\n📊 Final Comparison Summary:")
print(report_df)

report_df.to_csv("bigdata_demo/flood_model_comparison.csv", index=False)
print("\n✅ Results saved to bigdata_demo/flood_model_comparison.csv")
