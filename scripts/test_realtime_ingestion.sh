#!/bin/bash

# ==============================================================================
# 🚀 RAINWISE - Real-Time Ingestion & Spark Processing Test
# ==============================================================================

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' 

echo -e "${BLUE}Starting Real-Time Big Data Integration Test...${NC}"

# STEP 1: Trigger Real-time Data Collection
echo -e "\n📡 Step 1: Triggering Real-time Ingestion..."
/Applications/miniconda3/bin/python src/data_collection/build_dataset.py

# STEP 2: Verify in HDFS
echo -e "\n📂 Step 2: Verifying HDFS Ingestion (Simulated)..."
./jps.sh
./jps.sh | grep -q "NameNode" || ./start-dfs.sh
echo -e "Listing hdfs://raw/realtime/:"
/Applications/miniconda3/bin/python -c "from src.bigdata.hdfs_simulator import HDFSSimulator; HDFSSimulator.ls('hdfs://raw/realtime/')"

# STEP 3: Run Spark Pipeline
echo -e "\n⚡ Step 3: Running PySpark Pipeline for Live Audit & Prediction..."
./pyspark.sh

# STEP 4: Check Results
echo -e "\n🏁 Step 4: Checking Processed Results in HDFS..."
python3 -c "from src.bigdata.hdfs_simulator import HDFSSimulator; HDFSSimulator.ls('hdfs://processed/realtime_predictions/')"

echo -e "\n${GREEN}✅ Real-time Big Data Integration Test Complete!${NC}"
