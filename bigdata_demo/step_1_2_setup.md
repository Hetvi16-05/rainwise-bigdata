# Big Data Demo: Steps 1 & 2

## Step 1: Identify and Acquire Raw Dataset
For this demonstration, we are using a curated meteorological grid dataset for India. 
- **Source:** NASA POWER (Solar and Meteorological data).
- **Format:** CSV (Comma Separated Values).
- **Landing Zone:** `bigdata_demo/raw/india_grid.csv`

## Step 2: Provision Cloud Instance & Environment
The project is architected for a distributed cloud environment.
- **Compute:** Provisioned AWS EC2 / Azure VM running Ubuntu 22.04 LTS.
- **Java:** Installed OpenJDK 11 (Critical for Hadoop/Spark).
- **Hadoop:** Installed Hadoop 3.3.x for distributed storage (HDFS).
- **Python:** Installed Python 3.10 with `pyspark`, `pandas`, `matplotlib`, and `seaborn`.

---

### Prerequisites Command (Simulated):
```bash
# Update system
sudo apt-get update

# Install Java 11
sudo apt-get install openjdk-11-jdk -y

# Install Python requirements
pip install pyspark pandas matplotlib seaborn joblib
```
