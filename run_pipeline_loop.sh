#!/bin/bash

cd /Users/HetviSheth/rainwise

PYTHON=/Applications/miniconda3/bin/python

LOG=/Users/HetviSheth/rainwise/pipeline.log

while true
do

MONTH=$(date +%m)

echo "------------------------" >> $LOG
date >> $LOG
echo "Current month: $MONTH" >> $LOG

if [[ "$MONTH" -ge 7 && "$MONTH" -le 10 ]]
then
    echo "Monsoon mode → run every 10 min" >> $LOG
    $PYTHON src/data_collection/run_realtime_pipeline.py >> $LOG 2>&1
    sleep 600
else
    echo "Normal mode → run every 60 min" >> $LOG
    $PYTHON src/data_collection/run_realtime_pipeline.py >> $LOG 2>&1
    sleep 3600
fi

done
