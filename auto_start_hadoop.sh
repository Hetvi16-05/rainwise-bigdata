#!/bin/bash

# ===================================================
# RAINWISE: HADOOP AUTO-START ON REBOOT
# ===================================================
# This script ensures that if your Mac restarts, Hadoop
# will automatically turn itself back on so the 
# continuous data pipeline doesn't crash.
# ===================================================

# 1. Wait 30 seconds for Mac to connect to Wi-Fi/Network
sleep 30

# 2. Set the exact paths so Cron knows where Java and Hadoop are
export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk-11.jdk/Contents/Home
export HADOOP_HOME=/Users/HetviSheth/Downloads/hadoop-3.3.6
export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin

# 3. Start the Hadoop Distributed File System (HDFS)
echo "--- Starting Hadoop DFS at $(date) ---" >> /Users/HetviSheth/rainwise/hadoop_autostart.log
$HADOOP_HOME/sbin/start-dfs.sh >> /Users/HetviSheth/rainwise/hadoop_autostart.log 2>&1

# 4. Start the YARN Resource Manager
echo "--- Starting Hadoop YARN at $(date) ---" >> /Users/HetviSheth/rainwise/hadoop_autostart.log
$HADOOP_HOME/sbin/start-yarn.sh >> /Users/HetviSheth/rainwise/hadoop_autostart.log 2>&1

echo "✅ Hadoop Cluster Online!" >> /Users/HetviSheth/rainwise/hadoop_autostart.log
