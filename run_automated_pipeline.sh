#!/bin/bash

# ===================================================
# RAINWISE AUTOMATED PIPELINE
# ===================================================
# This script is triggered by cron. 
# It runs the live data collection and stores in HDFS.
# ===================================================

PROJECT_DIR="/Users/HetviSheth/rainwise"
PYTHON="/Applications/miniconda3/bin/python"

cd $PROJECT_DIR

echo "--- PIPELINE START: $(date) ---"
$PYTHON src/data_collection/run_realtime_pipeline.py
echo "--- PIPELINE END: $(date) ---"
echo ""
