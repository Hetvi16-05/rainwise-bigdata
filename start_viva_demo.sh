#!/bin/bash

# ===================================================
# RAINWISE VIVA DEMONSTRATION SCRIPT
# ===================================================
# This script simulates the "Invisible Scheduler" (Cron Job)
# but runs every 15 seconds instead of 30 minutes so you 
# can demonstrate live data ingestion to the examiner quickly!
# ===================================================

PROJECT_DIR="/Users/HetviSheth/rainwise"
PYTHON="/Applications/miniconda3/bin/python"

cd $PROJECT_DIR

echo "==================================================="
echo "🚀 STARTING CONTINUOUS BIG DATA INGESTION DEMO 🚀"
echo "==================================================="
echo "The system will now continuously fetch live sensor data"
echo "and push it to Hadoop HDFS every 15 seconds."
echo "Press [CTRL + C] to stop the demonstration."
echo "==================================================="
echo ""

while true
do
    echo "⏱️ [$(date '+%H:%M:%S')] Crontab triggered: Executing Data Fetch Pipeline..."

    # Run the blazing-fast presentation pipeline (takes 0.5 seconds instead of 20 mins)
    $PYTHON src/data_collection/viva_fast_pipeline.py

    echo "✅ Pipeline Cycle Complete."
    echo "⏳ Waiting 15 seconds for the next API fetch cycle..."
    echo "---------------------------------------------------"
    sleep 15
done

