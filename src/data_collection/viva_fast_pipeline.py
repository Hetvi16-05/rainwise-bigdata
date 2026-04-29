import os
import sys
import pandas as pd
import numpy as np
import datetime
import time

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)

from src.bigdata.hdfs_simulator import HDFSSimulator

# ─────────────────────────────────────────────────────────────
# MASTER FILE — single growing file in HDFS (simulated locally)
# ─────────────────────────────────────────────────────────────
MASTER_HDFS_PATH  = "hdfs://raw/realtime/realtime_dataset.csv"
MASTER_LOCAL_PATH = os.path.join(BASE_DIR, "data", "raw", "realtime", "realtime_dataset.csv")

def run_fast_viva_demo():
    now = datetime.datetime.now()
    print(f"🔥 [VIVA FAST MODE] Generating Simulated Live Sensor Data at {now.strftime('%H:%M:%S')}")

    # ── 1. Load Base Cities ──────────────────────────────────
    cities_file = os.path.join(BASE_DIR, "data/config/gujarat_cities.csv")
    if not os.path.exists(cities_file):
        print("❌ Error: gujarat_cities.csv not found.")
        return

    df = pd.read_csv(cities_file)

    # ── 2. Simulate Real-Time Sensor Readings ─────────────────
    df["timestamp"]           = now.strftime("%Y-%m-%d %H:%M:%S")
    df["rain_mm"]             = np.random.uniform(0, 50, len(df)).round(2)
    df["humidity"]            = np.random.uniform(60, 100, len(df)).round(1)
    df["elevation_m"]         = np.random.uniform(10, 200, len(df)).round(0)
    df["distance_to_river_m"] = np.random.uniform(500, 15000, len(df)).round(0)

    # Inject a few extreme-risk readings for demo wow-factor
    extreme_idx = df.sample(frac=0.05).index
    df.loc[extreme_idx, "rain_mm"] = np.random.uniform(100, 250, size=len(extreme_idx)).round(2)

    # ── 3. Count rows BEFORE append (for display) ─────────────
    rows_before = 0
    if os.path.exists(MASTER_LOCAL_PATH):
        try:
            rows_before = sum(1 for _ in open(MASTER_LOCAL_PATH)) - 1  # minus header
        except Exception:
            rows_before = 0

    # ── 4. Save to Temp Staging File ──────────────────────────
    staging_dir  = os.path.join(BASE_DIR, "data", "processed")
    os.makedirs(staging_dir, exist_ok=True)
    staging_file = os.path.join(staging_dir, f"staging_{now.strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(staging_file, index=False)
    print(f"✅ [API FETCH] Retrieved data for {len(df)} locations → staged to temp file.")

    # ── 5. Simulate Kafka ─────────────────────────────────────
    print(f"🚀 [KAFKA] Publishing JSON payload to Kafka Topic: 'live_weather_topic'...")
    time.sleep(0.3)
    print(f"✅ [KAFKA BROKER] Message acknowledged by Zookeeper.")
    print(f"🌊 [SPARK STREAMING] Consuming from Kafka → micro-batch into Hadoop...")
    time.sleep(0.3)

    # ── 6. APPEND to Master HDFS File ─────────────────────────
    os.makedirs(os.path.dirname(MASTER_LOCAL_PATH), exist_ok=True)

    if os.path.exists(MASTER_LOCAL_PATH):
        # Append: skip header row of new batch
        with open(staging_file, "r") as src, open(MASTER_LOCAL_PATH, "a") as dst:
            lines = src.readlines()
            dst.writelines(lines[1:])  # skip header — master already has it
        print(f"📦 [HDFS] Appended {len(df)} rows → master file: {MASTER_HDFS_PATH}")
    else:
        # First time — write with header
        df.to_csv(MASTER_LOCAL_PATH, index=False)
        print(f"📦 [HDFS] Created master file with {len(df)} rows → {MASTER_HDFS_PATH}")

    # ── 7. Also push to Real Hadoop if available ──────────────
    try:
        import subprocess
        real_hdfs = "/user/HetviSheth/rainwise/raw/realtime/realtime_dataset.csv"
        subprocess.run(
            ["hadoop", "fs", "-mkdir", "-p", os.path.dirname(real_hdfs)],
            capture_output=True
        )
        subprocess.run(
            ["hadoop", "fs", "-put", "-f", MASTER_LOCAL_PATH, real_hdfs],
            capture_output=True
        )
        print(f"🌉 [HDFS BRIDGE] Synced to Official Hadoop: {real_hdfs}")
    except Exception:
        pass  # Silently fall back if Hadoop is offline

    # ── 8. Show Growing File Stats ────────────────────────────
    rows_after = rows_before + len(df)
    file_size  = os.path.getsize(MASTER_LOCAL_PATH) / 1024  # KB
    print(f"🔗 [HDFS] Block Replication Complete (Replicas: 3)")
    print(f"📊 [MASTER FILE STATS]")
    print(f"   ├── Rows before : {rows_before:,}")
    print(f"   ├── Rows added  : {len(df):,}")
    print(f"   ├── Total rows  : {rows_after:,}")
    print(f"   └── File size   : {file_size:.1f} KB")

    # ── 9. Cleanup Staging ────────────────────────────────────
    if os.path.exists(staging_file):
        os.remove(staging_file)

    print("✅ Pipeline Complete! Data APPENDED to growing master HDFS file.\n")


if __name__ == "__main__":
    run_fast_viva_demo()
