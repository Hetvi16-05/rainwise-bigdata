import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import sys
from datetime import datetime

# Path Hacks
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)

from src.utils.hdfs_reader import HDFSReader
from model_trainingDL.models import TabTransformer
from dlfinal_app import CompatibleFloodDNN # Use the compatible one for weights

def incremental_update():
    print(f"🔄 Starting Incremental Model Update: {datetime.now()}")
    device = torch.device("cpu")
    
    # 1. Fetch Latest Data from HDFS
    df = HDFSReader.get_latest_realtime()
    if df.empty:
        print("⚠️ No data found in HDFS. Skipping update.")
        return
    
    # Pre-process for Training
    # We need labels. For Live data, we use heuristics if 'flood' is missing.
    if 'flood' not in df.columns:
        df['flood'] = (df['rain_mm'] > 60).astype(int) # Heuristic for live labels
        
    # ==========================================
    # UPDATE FLOOD MODEL (DNN)
    # ==========================================
    print("🧠 Updating FloodDNN...")
    flood_model = CompatibleFloodDNN()
    if os.path.exists("DLmodels/flood_dnn.pth"):
        flood_model.load_state_dict(torch.load("DLmodels/flood_dnn.pth", map_location=device))
    
    # Select features [rain, elev, dist, lat, lon]
    f_features = ['rain_mm', 'elevation_m', 'distance_to_river_m', 'lat', 'lon']
    X_f = torch.tensor(df[f_features].values, dtype=torch.float32)
    y_f = torch.tensor(df['flood'].values, dtype=torch.float32).view(-1, 1)
    
    optimizer_f = optim.Adam(flood_model.parameters(), lr=1e-4)
    criterion_f = nn.BCELoss()
    
    flood_model.train()
    for _ in range(5): # 5 micro-epochs
        optimizer_f.zero_grad()
        output = flood_model(X_f)
        loss = criterion_f(output, y_f)
        loss.backward()
        optimizer_f.step()
    
    torch.save(flood_model.state_dict(), "DLmodels/flood_dnn.pth")
    print(f"✅ FloodDNN Weights Updated. Loss: {loss.item():.4f}")

    # ==========================================
    # UPDATE RAINFALL MODEL (TabTransformer)
    # ==========================================
    print("🌧️ Updating TabTransformer...")
    rain_model = TabTransformer(input_dim=6, depth=3)
    if os.path.exists("DLmodels/tab_transformer_rainfall.pth"):
        rain_model.load_state_dict(torch.load("DLmodels/tab_transformer_rainfall.pth", map_location=device))
    
    # Features [elev, dist, lat, lon, pop, rain3]
    # For live update, we use current rain as target
    if 'population_2026' not in df.columns: df['population_2026'] = 1000000
    if 'rain3_mm' not in df.columns: df['rain3_mm'] = df['rain_mm'] # Placeholder
    
    r_features = ['elevation_m', 'distance_to_river_m', 'lat', 'lon', 'population_2026', 'rain3_mm']
    X_r = torch.tensor(df[r_features].values, dtype=torch.float32)
    y_r = torch.tensor(df['rain_mm'].values, dtype=torch.float32).view(-1, 1)
    
    optimizer_r = optim.Adam(rain_model.parameters(), lr=1e-5)
    criterion_r = nn.MSELoss()
    
    rain_model.train()
    for _ in range(5):
        optimizer_r.zero_grad()
        output = rain_model(X_r)
        loss = criterion_r(output, y_r)
        loss.backward()
        optimizer_r.step()
        
    torch.save(rain_model.state_dict(), "DLmodels/tab_transformer_rainfall.pth")
    print(f"✅ TabTransformer Weights Updated. Loss: {loss.item():.4f}")
    print("🚀 Models are now more 'Live' and adapted to recent climate patterns.")

if __name__ == "__main__":
    incremental_update()
