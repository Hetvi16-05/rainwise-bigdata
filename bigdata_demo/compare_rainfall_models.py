import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import joblib

# 1. Load Data (Using a sample of 50k rows for efficiency)
df = pd.read_csv("data/processed/training_dataset_gujarat_advanced_labeled.csv", nrows=50000)
df.columns = df.columns.str.lower()
rain = df["rain_mm"].values

# 2. Replicate Synthetic Atmospheric Features
np.random.seed(42)
base_temp = np.random.uniform(28, 42, size=len(rain))
rain_cooling = np.clip(rain * 0.15, 0, 12)
temperature = np.clip(base_temp - rain_cooling + np.random.normal(0, 1.5, size=len(rain)), 15, 48)

base_humidity = np.random.uniform(30, 60, size=len(rain))
rain_humidity_boost = np.clip(rain * 1.5, 0, 40)
humidity = np.clip(base_humidity + rain_humidity_boost + np.random.normal(0, 5, size=len(rain)), 20, 100)

base_pressure = np.random.uniform(1008, 1020, size=len(rain))
rain_pressure_drop = np.clip(rain * 0.3, 0, 20)
pressure = np.clip(base_pressure - rain_pressure_drop + np.random.normal(0, 2, size=len(rain)), 985, 1035)

base_wind = np.random.uniform(5, 15, size=len(rain))
rain_wind_boost = np.clip(rain * 0.5, 0, 30)
wind_speed = np.clip(base_wind + rain_wind_boost + np.random.normal(0, 3, size=len(rain)), 0, 80)

base_cloud = np.random.uniform(10, 40, size=len(rain))
rain_cloud_boost = np.clip(rain * 2.0, 0, 60)
cloud_cover = np.clip(base_cloud + rain_cloud_boost + np.random.normal(0, 5, size=len(rain)), 0, 100)

# Build Feature Matrix
features = ["temperature", "humidity", "pressure", "wind_speed", "cloud_cover"]
target = "rain_mm"
df["temperature"] = temperature
df["humidity"] = humidity
df["pressure"] = pressure
df["wind_speed"] = wind_speed
df["cloud_cover"] = cloud_cover

df_model = df[features + [target]].dropna()
X = df_model[features].values
y = df_model[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Training & Evaluation
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42)
}

results = []

print("\n🚀 Starting Multi-Algorithm Comparison for Rainfall Regression...")

for name, model in models.items():
    print(f"\n--- Evaluating {name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None) # No negative rain
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    
    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
    
    # Save Prediction Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test[:5000], y_pred[:5000], alpha=0.3, s=10)
    plt.plot([0, y_test.max()], [0, y_test.max()], '--', color='red')
    plt.title(f"Actual vs Predicted: {name} (R²={r2:.3f})")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.savefig(f"bigdata_demo/plots/rainfall_r2_{name.lower().replace(' ', '_')}.png")
    plt.close()

# 4. Summary Output
report_df = pd.DataFrame(results)
print("\n📊 Final Comparison Summary:")
print(report_df)

report_df.to_csv("bigdata_demo/rainfall_model_comparison.csv", index=False)
print("\n✅ Results saved to bigdata_demo/rainfall_model_comparison.csv")
