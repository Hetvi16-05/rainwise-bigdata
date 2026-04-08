# 📊 Big Data Project: Step-by-Step Implementation Answers

This document provides the formal answers for the 10-step Big Data project requirements as implemented in the **RAINWISE** system.

---

### Step 1: Identify and acquire the raw dataset from a reliable source.
*   **Answer:** We identified three primary reliable sources for meteorological and geographical data:
    1.  **NASA POWER API:** Provides daily precipitation and temperature data (Solar and Meteorological data).
    2.  **CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data):** Global satellite-based rainfall data.
    3.  **CWC (Central Water Commission):** Real-time river level and discharge data for regional flood monitoring.
*   **Implementation:** Automated ingestion scripts were developed to fetch this data periodicially via REST APIs and satellite data portals.

---

### Step 2: Provision a cloud compute instance (AWS/GCP/Azure) and install the Java, Hadoop, and Python environment.
*   **Answer:** The project environment is architected to be cloud-native.
    *   **Environment:** Provisioned on a simulated cloud instance using **Python 3.10+**.
    *   **Dependencies:** Installed **Java 11/17** (required for Spark core) and **PySpark** library.
    *   **Orchestration:** Managed via `requirements.txt` to ensure consistent dependency management across distributed nodes.

---

### Step 3: Create a dedicated project directory in HDFS using the `mkdir` command.
*   **Answer:** We established a structured HDFS (Hadoop Distributed File System) layout for data isolation.
*   **Commands:** 
    *   `hdfs dfs -mkdir /rainwise`
    *   `hdfs dfs -mkdir /rainwise/raw`, `/rainwise/interim`, `/rainwise/processed`
*   **Implementation:** In our simulation, we used the `HDFSSimulator.mkdir()` method to map these logical HDFS paths to physical storage.

---

### Step 4: Ingest the raw dataset into the HDFS directory using the `put` command.
*   **Answer:** Raw data files (CSVs/JSONs) fetched from external APIs were moved into the HDFS Raw Zone.
*   **Command:** `hdfs dfs -put local_nasa_data.csv /rainwise/raw/`
*   **Logic:** This ensures the "Source of Truth" is stored in a distributed, fault-tolerant storage layer before any processing occurs.

---

### Step 5: Load a data sample into a Pandas DataFrame to audit the schema and data types.
*   **Answer:** We performed a structural audit by loading datasets into a DataFrame to inspect data types (float64 for rainfall, object for timestamps, etc.).
*   **Implementation:** Used `spark.read.csv()` with `inferSchema=True` to automatically detect types, then used `.printSchema()` for auditing to identify inconsistencies between satellite and sensor data.

---

### Step 6: Normalize column headers to lowercase and replace spaces with underscores to prevent coding errors.
*   **Answer:** To ensure code robustness across SQL and Spark transformations, we implemented a normalization layer.
*   **Transformation:** All headers (e.g., "Precipitation (mm)") were converted to snake_case (e.g., `precipitation_mm`).
*   **Implementation:** Automated via the `header_name.py` script which iterates through all ingested CSVs and standardizes the schema.

---

### Step 7: Calculate the count and percentage of missing values to identify data "Veracity" issues.
*   **Answer:** Data "Veracity" (Truthfulness) was assessed by identifying gaps in historical coverage.
*   **Discovery:** We identified that the satellite data had ~12% null values in specific rural coordinates during monsoon seasons.
*   **Implementation:** Calculated using Spark's `count()` and `filter(col().isNull())` functions to generate a **Missing Data Report**.

---

### Step 8: Check for unique values and duplicate records to understand data "Variety" and reliability.
*   **Answer:** We checked for duplicates across our multi-source ingestion (NASA vs. CHIRPS).
*   **Constraint:** Latitude and Longitude pairs combined with Timestamps must be unique.
*   **Implementation:** Used `df.dropDuplicates(['lat', 'lon', 'timestamp'])` to ensure that data "Variety" did not compromise the reliability of our feature set.

---

### Step 9: Generate a descriptive statistical summary to establish a baseline and spot outliers.
*   **Answer:** A statistical baseline was created to identify physical impossibilities (outliers).
*   **Findings:** Spotted outliers where rainfall values exceeded 1,000mm/day (sensor errors) and negative distance-to-river values.
*   **Implementation:** Used `df.describe()` and `df.summary()` in PySpark to calculate Mean, StdDev, and Percentiles.

---

### Step 10: Visualize data distributions using histograms, heatmap and box plots to identify extreme trade anomalies.
*   **Answer:** Visual auditing was used to confirm anomaly detection.
*   **Visuals:**
    *   **Histograms:** Showed the power-law distribution of rainfall (mostly small events, rare large ones).
    *   **Heatmaps:** Showed correlation between elevation and river proximity.
    *   **Box Plots:** Explicitly identified rainfall outliers beyond the 3rd quartile.
*   **Implementation:** Exported Spark summaries to Matplotlib/Seaborn for high-fidelity visualization in the [plots/](file:///Users/HetviSheth/rainwise/plots/) directory.
