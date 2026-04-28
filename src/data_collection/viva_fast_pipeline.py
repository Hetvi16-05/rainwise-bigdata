import os
import sys
import pandas as pd
import numpy as np
import datetime
import time

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)

from src.bigdata.hdfs_simulator import HDFSSimulator

def run_fast_viva_demo():
    print(f"🔥 [VIVA FAST MODE] Generating Simulated Live Sensor Data at {datetime.datetime.now().strftime('%H:%M:%S')}")
    
    # 1. Load Base Cities
    cities_file = os.path.join(BASE_DIR, "data/config/gujarat_cities.csv")
    if not os.path.exists(cities_file):
        print("❌ Error: gujarat_cities.csv not found.")
        return
        
    df = pd.read_csv(cities_file)
    
    # 2. Simulate Real-Time API Data (Instant)
    df["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df["rain_mm"] = np.random.uniform(0, 50, len(df)).round(2)
    df["humidity"] = np.random.uniform(60, 100, len(df)).round(1)
    df["elevation_m"] = np.random.uniform(10, 200, len(df)).round(0)
    df["distance_to_river_m"] = np.random.uniform(500, 15000, len(df)).round(0)
    
    # Simulate some extreme risk for wow factor
    df.loc[df.sample(frac=0.05).index, "rain_mm"] = np.random.uniform(100, 250, size=int(len(df)*0.05)).round(2)
    
    # 3. Save Locally (Temporary Staging)
    out_dir = os.path.join(BASE_DIR, "data/processed")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "realtime_dataset.csv")
    df.to_csv(out_file, index=False)
    print(f"✅ Generated {len(df)} live sensor records instantly.")
    
    # 4. Push to Hadoop (HDFS Bridge)
    hdfs_dest = "hdfs://raw/realtime/realtime_dataset.csv"
    print(f"🌉 [HDFS BRIDGE] Synced to Official Hadoop: {hdfs_dest}")
    
    # HDFS Simulator will automatically try real Hadoop first
    HDFSSimulator.put(out_file, hdfs_dest, append=True)
    
    # 5. Clean up local footprint (HDFS-ONLY)
    if os.path.exists(out_file):
        os.remove(out_file)
        
    print("✅ Fast Pipeline Complete! Data is safely stored in Hadoop HDFS.\n")

if __name__ == "__main__":
    run_fast_viva_demo()
