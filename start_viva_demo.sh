#!/bin/bash

# ===================================================
# RAINWISE VIVA DEMONSTRATION SCRIPT
# Smart Scheduler: Month-Based Interval Control
# ===================================================
# Monsoon Months (June–Sep)    → Pipeline runs every 30 MINUTES
# Non-Monsoon Months (Oct–May) → Pipeline runs every 60 MINUTES
# ===================================================

PROJECT_DIR="/Users/HetviSheth/rainwise"
PYTHON="/Applications/miniconda3/bin/python"

# ---------------------------------------------------
# PREVENT MULTIPLE INSTANCES
# ---------------------------------------------------
LOCK_FILE="/tmp/rainwise_viva.lock"
if [ -f "$LOCK_FILE" ]; then
    echo "⚠️  WARNING: Another instance of start_viva_demo.sh is already running!"
    echo "   PID in lock: $(cat $LOCK_FILE)"
    echo "   Run: pkill -f start_viva_demo.sh  → to kill all instances first."
    exit 1
fi
echo $$ > "$LOCK_FILE"

# Cleanup lock file on exit (Ctrl+C or normal exit)
trap "rm -f $LOCK_FILE; echo ''; echo '🛑 Demo stopped. Lock file removed.'; exit 0" SIGINT SIGTERM EXIT

cd $PROJECT_DIR

# ---------------------------------------------------
# HELPER: Determine interval based on current month
# ---------------------------------------------------
get_interval() {
    CURRENT_MONTH=$(date +%-m)  # 1=Jan ... 12=Dec
    # Monsoon months: June(6), July(7), August(8), September(9)
    if [ "$CURRENT_MONTH" -ge 6 ] && [ "$CURRENT_MONTH" -le 9 ]; then
        echo 1800  # 30 minutes in seconds
    else
        echo 3600  # 60 minutes in seconds
    fi
}

get_interval_label() {
    CURRENT_MONTH=$(date +%-m)
    if [ "$CURRENT_MONTH" -ge 6 ] && [ "$CURRENT_MONTH" -le 9 ]; then
        echo "30 minutes (🌧️ Monsoon Season)"
    else
        echo "60 minutes (☀️ Non-Monsoon Season)"
    fi
}

# ---------------------------------------------------
# STARTUP BANNER
# ---------------------------------------------------
echo "==================================================="
echo "🚀 RAINWISE — SMART CONTINUOUS INGESTION DEMO 🚀"
echo "==================================================="
echo "📅 Current Month : $(date '+%B %Y')"
echo "⏱️  Pipeline Interval : $(get_interval_label)"
echo "==================================================="
echo "Press [CTRL + C] to stop the demonstration."
echo ""

# ---------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------
CYCLE=1
while true
do
    # Recalculate interval each cycle (handles midnight month rollover)
    INTERVAL=$(get_interval)
    LABEL=$(get_interval_label)

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔁 CYCLE #$CYCLE  |  $(date '+%d %b %Y — %H:%M:%S')"
    echo "📡 Schedule: Every $LABEL"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Run the pipeline
    $PYTHON src/data_collection/viva_fast_pipeline.py

    NEXT_RUN=$(date -v +${INTERVAL}S '+%H:%M:%S')
    echo "⏳ Next pipeline run at: $NEXT_RUN  (in $(( INTERVAL / 60 )) minutes)"
    echo ""

    CYCLE=$((CYCLE + 1))
    sleep $INTERVAL
done
