import os
import subprocess
import sys
import time
from datetime import datetime

# ----------------------
# paths
# ----------------------
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

PYTHON = "/Applications/miniconda3/bin/python"

LOCK_FILE = os.path.join(BASE_DIR, "pipeline.lock")
LAST_RUN_FILE = os.path.join(BASE_DIR, "pipeline_last_run.txt")


# ----------------------
# run script safely
# ----------------------
def run_script(script, name):

    path = os.path.join(BASE_DIR, script)

    try:
        print(f"[{datetime.now()}] 🚀 Running: {name}")

        result = subprocess.run(
            [PYTHON, path],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"❌ Failed: {name}")
            print(result.stderr)
            return False

        print(f"✅ Completed: {name}")
        return True

    except Exception as e:
        print(f"❌ Error in {name}: {e}")
        return False


# ----------------------
# pipeline
# ----------------------
def run_pipeline():

    if not run_script("data_collection/realtime_weather.py", "Weather"):
        return False

    if not run_script("data_collection/realtime_rainfall.py", "Rainfall"):
        return False

    if not run_script("data_collection/realtime_river.py", "River"):
        return False

    if not run_script("data_collection/build_dataset.py", "Dataset"):
        return False

    print("🎉 Pipeline completed successfully")
    return True


# ----------------------
# SMART LOCK SYSTEM
# ----------------------
def create_lock():

    if os.path.exists(LOCK_FILE):
        age = time.time() - os.path.getmtime(LOCK_FILE)

        # remove stale lock (older than 1 hour)
        if age > 3600:
            print("⚠️ Removing stale lock file...")
            os.remove(LOCK_FILE)
        else:
            print("⚠️ Pipeline already running. Exiting.")
            sys.exit()

    with open(LOCK_FILE, "w") as f:
        f.write(str(os.getpid()))


def remove_lock():
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)


# ----------------------
# avoid duplicate runs
# ----------------------
def already_ran_recently():

    if not os.path.exists(LAST_RUN_FILE):
        return False

    try:
        with open(LAST_RUN_FILE, "r") as f:
            last_time = f.read().strip()

        last_time = datetime.fromisoformat(last_time)

    except Exception:
        return False

    diff = (datetime.now() - last_time).total_seconds()

    # skip if ran in last 5 minutes
    return diff < 300


def update_last_run():
    with open(LAST_RUN_FILE, "w") as f:
        f.write(datetime.now().isoformat())


# ----------------------
# MAIN (RUN ONCE)
# ----------------------
if __name__ == "__main__":

    print(f"\n🔥 Running realtime pipeline at {datetime.now()}")

    # skip duplicate execution
    if already_ran_recently():
        # silent skip (no spam logs)
        sys.exit()

    create_lock()

    try:
        success = run_pipeline()

        if success:
            update_last_run()
        else:
            print("❌ Pipeline failed")

    finally:
        remove_lock()