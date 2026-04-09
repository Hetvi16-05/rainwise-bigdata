# 🎬 RAINWISE: Step-by-Step Viva Demo Script
**Follow these exact steps for a flawless 10-minute presentation.**

---

### ⏱️ Phase 1: The "Everything is UP" Show (2 Minutes)
*   **Step 1**: Open your Browser to `http://10.32.44.92:8080` (Spark UI).
*   **Step 2**: Open your Browser to `http://localhost:9870` (HDFS UI).
*   **Say this**: *"Good morning Professor. As you can see, our RAINWISE Big Data cluster is live. My laptop is the **Master Node**, and we have the capacity to attach worker slaves. All our data is stored in the **HDFS (Hadoop Distributed File System)** with a replication factor of 3 for fault tolerance."*

---

### 🚀 Phase 2: The "Big Data Volume" Audit (3 Minutes)
*   **Step 1**: Open a Terminal and run: `./pyspark.sh`.
*   **Step 2**: Once Spark is ready, point to the code that counts the **209 Million records**.
*   **Say this**: *"We are now processing **209 Million records**. In a standard SQL database, this would take significant time. In RAINWISE, we use **Apache Spark's parallel processing**. We also use **Apache Parquet**, which is a columnar storage format that reduces I/O by 10x compared to CSV."*

---

### 📡 Phase 3: The "Live Velocity" Demo (2 Minutes)
*   **Step 1**: Open a NEW Terminal tab and run: `./scripts/start_viva_turbo.sh`.
*   **Step 2**: Show the screen as it says *"Fetching fresh Big Data cycle..."*
*   **Say this**: *"Our system handles the **5 V's of Big Data**. Here you can see **Velocity** in action. Every couple of minutes, new weather and satellite data is ingested, audited for veracity, and automatically appended to our HDFS cluster in an append-only, immutable format."*

---

### 🎨 Phase 4: The "Visual Story" (3 Minutes)
*   **Step 1**: Open the folder `reports/viva_visuals/` on your Mac.
*   **Step 2**: Show **`gujarat_risk_map.png`** and **`veracity_audit_pie.png`**.
*   **Say this**: *"Finally, our AI layer translates this raw Big Data into actionable value. The heatmap shows real-time flood risk across Gujarat cities. Note the Veracity Audit chart—it proves that we successfully filtered out anomalies from the 200 Million record stream before making predictions."*

---

### 🏁 The Closing Statement
*"In conclusion, RAINWISE proves that with **commodity laptops** and **Spark/Hadoop**, we can build a climate monitoring system that handles massive volumes and provides real-time protective intelligence. Thank you."*
