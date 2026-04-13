import os
import shutil

class HDFSSimulator:
    """Simulates HDFS CLI operations for demonstration."""
    def __init__(self, root):
        self.root = root
        print(f"📡 [HDFS] Connected to NameNode at {root}")

    def mkdir(self, hdfs_path):
        """Simulates 'hdfs dfs -mkdir -p <path>'"""
        # Strip hdfs:// prefix if exists
        clean_path = hdfs_path.replace("hdfs://", "").strip("/")
        full_path = os.path.join(self.root, clean_path)
        os.makedirs(full_path, exist_ok=True)
        print(f"📁 [HDFS] Created directory: {hdfs_path}")

    def put(self, local_src, hdfs_dest):
        """Simulates 'hdfs dfs -put <src> <dest>'"""
        clean_dest = hdfs_dest.replace("hdfs://", "").strip("/")
        full_dest = os.path.join(self.root, clean_dest)
        
        # If dest is a directory, append filename
        if os.path.isdir(full_dest):
            full_dest = os.path.join(full_dest, os.path.basename(local_src))
            
        shutil.copy2(local_src, full_dest)
        print(f"📦 [HDFS] Uploded {os.path.basename(local_src)} to {hdfs_dest}")
        print(f"🔗 [HDFS] Block Replication Factor: 3. Distribution complete.")

if __name__ == "__main__":
    # Simulate Step 3
    hdfs = HDFSSimulator("bigdata_demo/hdfs_root")
    print("\n--- Step 3: Create Dedicated Project Directory ---")
    hdfs.mkdir("hdfs://rainwise/raw")
    
    # Simulate Step 4
    print("\n--- Step 4: Ingest Raw Dataset ---")
    hdfs.put("bigdata_demo/raw/india_grid.csv", "hdfs://rainwise/raw/")
    
    print("\n✅ Steps 3 & 4 Complete.")
