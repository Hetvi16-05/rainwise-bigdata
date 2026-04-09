# 🏆 RAINWISE: The Ultimate Big Data Viva Reference
**Project Goal:** End-to-end distributed forecasting and real-time ingestion for Gujarat's flood protection.

---

## 🌩️ Section 1: The "100 Marks" Technical Summary

| Big Data Aspect | Implementation Detail |
| :--- | :--- |
| **Record Volume** | **209,693,760 records** (Approx 209 Million rows) processed in parallel. |
| **Data Velocity** | Real-time **Micro-batch Ingestion** using an **Append-mode HDFS Bridge**. |
| **Storage (HDFS)** | Simulated **Hadoop HDFS** with NameNode, DataNodes, and **Replication Factor: 3**. |
| **Processing Engine** | **Apache PySpark 3.x** utilizing **Apache Parquet** for columnar high-speed storage. |
| **Fault Tolerance** | Designed with **distributed check-pointing** and multi-node coordination. |

---

## 🦁 Section 2: Architecture Deep-Dive

### 1. The Medallion Data Strategy
*   **Bronze (Raw)**: Incoming simulated HDFS streams (CSVs from APIs/Historical logs).
*   **Silver (Interim)**: Data audited via **PySpark Veracity Engine** (Null filtering & Schema validation).
*   **Gold (Processed)**: Highly compressed **Apache Parquet** files used for sub-second ML inference.

### 2. Physical Multi-Node Cluster
*   **Master (Hetvi's Mac)**: Coordinates the **DAG (Directed Acyclic Graph)** and task distribution.
*   **Slaves (Friends' Laptops)**: Perform the heavy computation.
*   **Shared Storage**: Implemented via **SMB Network Mounts** to decouple storage from compute.

---

## ⚡ Section 3: Spark vs. Standard Databases (The "Why")

| Feature | Standard SQL (MySQL/PostgreSQL) | **RAINWISE (PySpark)** |
| :--- | :--- | :--- |
| **Scaling** | Vertical (Larger CPU) | **Horizontal (Add more laptops)** |
| **209M Records**| High latency, might crash. | **Distributed across nodes (Sub-minute audit)** |
| **Streaming** | Hard to scale real-time flows. | **Micro-batching supported natively.** |
| **Format** | Row-based (Slow for analytics) | **Columnar Parquet (Lightning fast analytics)** |

---

## 👄 Section 4: Key Viva Keywords (Memorize These!)

*   **"Partitioning"**: How we split the 209M records into small chunks so every core/laptop can work on one piece simultaneously.
*   **"Shuffling"**: The process where data moves across the network between laptops (Master ↔ Slave).
*   **"Columnar Storage"**: Parquet stores data by columns instead of rows, allowing us to skip reading data we don't need.
*   **"Lazy Evaluation"**: Spark doesn't process data until we ask for a result (like `.show()` or `.count()`). This optimizes the execution plan.

---

## 🚀 Final Talking Point
*"Professor, the core innovation of RAINWISE isn't just the AI model—it's the **Scalable Big Data Infrastructure**. We've proven that we can take a dataset of **200 million records** and turn it into **real-time intelligence** using a physically distributed cluster of commodity hardware."*
