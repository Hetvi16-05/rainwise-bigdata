import torch
import joblib
import numpy as np
import os
from model_trainingDL.models import TabTransformer, FloodDNN

# ==========================================
# CONFIGURATION
# ==========================================
RAIN_MODEL_PATH = "DLmodels/tab_transformer_rainfall.pth"
RAIN_SCALER_PATH = "DLmodels/tab_transformer_scaler.pkl"
FLOOD_MODEL_PATH = "DLmodels/flood_model.pth"
FLOOD_SCALER_PATH = "DLmodels/flood_scaler.pkl"

# Check for existence of models
for path in [RAIN_MODEL_PATH, RAIN_SCALER_PATH, FLOOD_MODEL_PATH, FLOOD_SCALER_PATH]:
    if not os.path.exists(path):
        print(f"❌ Missing: {path}")

# ==========================================
# LOAD MODELS
# ==========================================
device = torch.device("cpu")

# 1. Rainfall Model (TabTransformer)
rain_model = TabTransformer(input_dim=6) # elev, dist, lat, lon, pop, rain3
rain_model.load_state_dict(torch.load(RAIN_MODEL_PATH, map_location=device))
rain_model.eval()
rain_scaler = joblib.load(RAIN_SCALER_PATH)

# 2. Flood Model (FloodDNN)
flood_model = FloodDNN(input_dim=7) # elev, dist, lat, lon, pop, rain, rain3
flood_model.load_state_dict(torch.load(FLOOD_MODEL_PATH, map_location=device))
flood_model.eval()
flood_scaler = joblib.load(FLOOD_SCALER_PATH)

def run_dl_pipeline(geo_data, rain3):
    """
    geo_data: [elev, dist, lat, lon, pop]
    rain3: previous 3-day total rainfall
    """
    print("\n--- 🌊 RAINWISE DEEP LEARNING PIPELINE ---")
    
    # STAGE 1: Rainfall Prediction
    # Features: [elev, dist, lat, lon, pop, rain3]
    rain_input = geo_data + [rain3]
    rain_array = np.array([rain_input])
    rain_scaled = rain_scaler.transform(rain_array)
    rain_tensor = torch.tensor(rain_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        pred_rain = rain_model(rain_tensor).item()
        pred_rain = max(0.0, pred_rain)
    
    print(f"🌧️ Stage 1 (Rainfall): Predicted {pred_rain:.2f} mm")
    
    # STAGE 2: Flood Prediction
    # Features: [elev, dist, lat, lon, pop, rain, rain3]
    flood_input = geo_data + [pred_rain, rain3]
    flood_array = np.array([flood_input])
    flood_scaled = flood_scaler.transform(flood_array)
    flood_tensor = torch.tensor(flood_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        flood_proba = flood_model(flood_tensor).item()
    
    print(f"🌊 Stage 2 (Flood Risk): {flood_proba*100:.1f}% Probability")
    
    return pred_rain, flood_proba

if __name__ == "__main__":
    # Test Data: Ahmedabad
    test_geo = [53.0, 1200.0, 23.02, 72.57, 5633927] # elev, dist, lat, lon, pop
    test_rain3 = 45.0 # Previous 3 days total rain
    
    rain, flood = run_dl_pipeline(test_geo, test_rain3)
    
    if flood > 0.5:
        print("🚨 ALERT: Significant flood risk predicted based on rain forecast.")
    else:
        print("✅ SAFE: Predicted rainfall is within capacity limits.")
