import pandas as pd
import subprocess
import os
import io

class HDFSReader:
    """Utility to read data from the Official Hadoop HDFS."""
    
    @staticmethod
    def read_csv(hdfs_path):
        """Reads a CSV from HDFS into a Pandas DataFrame."""
        try:
            # For HDFS-only storage, we use 'hadoop fs -cat' to stream data to pandas
            # Map hdfs:// paths to the actual project HDFS structure
            real_path = hdfs_path.replace("hdfs://", "/user/HetviSheth/rainwise/")
            
            result = subprocess.run(
                ["hadoop", "fs", "-cat", real_path],
                capture_output=True,
                text=False # Read as bytes
            )
            
            if result.returncode == 0:
                return pd.read_csv(io.BytesIO(result.stdout))
            else:
                return pd.DataFrame()
        except Exception:
            # Fallback to local if Hadoop is not running or path not found
            # (Though in HDFS-only mode, this will return empty)
            return pd.DataFrame()

    @staticmethod
    def get_latest_realtime():
        """Helper for the RAINWISE dashboards."""
        return HDFSReader.read_csv("hdfs://raw/realtime/realtime_dataset.csv")
