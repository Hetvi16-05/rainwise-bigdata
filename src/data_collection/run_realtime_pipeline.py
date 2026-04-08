"""
🔥 RAINWISE Real-Time Pipeline
===============================
Orchestrates the complete data collection pipeline.
Designed to run via cron — executes ONCE per trigger, no infinite loop.

Steps:
  1. Weather    → realtime_weather.py
  2. Satellite  → gpm_fetcher.py (NEW — satellite rainfall)
  3. Rainfall   → realtime_rainfall.py (uses gpm_fetcher)
  4. River      → realtime_river.py
  5. Dataset    → build_dataset.py (merges all + geospatial features)

Safety:
  - Lock file prevents concurrent runs
  - Last-run timestamp prevents duplicate execution within 5 minutes
  - Stale lock auto-cleanup after 1 hour
  - All steps logged to pipeline.log
"""

import os
import subprocess
import sys
import time
import logging
from datetime import datetime

# ----------------------
# PATHS
# ----------------------
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

PROJECT_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..")
)

PYTHON = "/Applications/miniconda3/bin/python"
LOG_FILE = os.path.join(PROJECT_DIR, "pipeline.log")
LOCK_FILE = os.path.join(BASE_DIR, "pipeline.lock")
LAST_RUN_FILE = os.path.join(BASE_DIR, "pipeline_last_run.txt")


# ----------------------
# LOGGING SETUP
# ----------------------
logger = logging.getLogger("pipeline")
logger.setLevel(logging.INFO)

# File handler → pipeline.log
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
fh = logging.FileHandler(LOG_FILE, mode="a")
fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(fh)

# Console handler
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(ch)


# ----------------------
# DUPLICATE CHECK (runs FIRST — before any log spam)
# ----------------------
def already_ran_recently():
    """Check if pipeline ran in the last 30/60 minutes depending on season."""
    if not os.path.exists(LAST_RUN_FILE):
        return False

    try:
        with open(LAST_RUN_FILE, "r") as f:
            last_time = f.read().strip()
        last_time = datetime.fromisoformat(last_time)
    except Exception:
        return False

    diff = (datetime.now() - last_time).total_seconds()

    # Determine required interval based on month
    # Peak Monsoon (July-Sept): 30 mins
    # Rest of Year (Oct-June): 60 mins
    month = datetime.now().month
    if month in [7, 8, 9]:
        required_interval = 1800  # 30 mins
    else:
        required_interval = 3600  # 60 mins

    # Allow 60s grace period for launchd timing
    return diff < (required_interval - 60)


def update_last_run():
    with open(LAST_RUN_FILE, "w") as f:
        f.write(datetime.now().isoformat())


# ----------------------
# LOCK SYSTEM
# ----------------------
def create_lock():
    """Create lock file. Remove stale locks (>1 hour)."""
    if os.path.exists(LOCK_FILE):
        age = time.time() - os.path.getmtime(LOCK_FILE)
        if age > 3600:
            logger.warning("⚠️ Removing stale lock file (older than 1 hour)")
            os.remove(LOCK_FILE)
        else:
            logger.info("⚠️ Pipeline already running. Exiting.")
            sys.exit(0)

    with open(LOCK_FILE, "w") as f:
        f.write(str(os.getpid()))


def remove_lock():
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)


# ----------------------
# RUN SCRIPT SAFELY
# ----------------------
def run_script(script, name):
    """Run a Python script and log the result."""
    path = os.path.join(BASE_DIR, script)

    if not os.path.exists(path):
        logger.error(f"❌ Script not found: {path}")
        return False

    start_time = time.time()

    try:
        logger.info(f"🚀 Running: {name}")

        result = subprocess.run(
            [PYTHON, path],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": PROJECT_DIR},
            timeout=300  # 5 minute timeout per script
        )

        elapsed = time.time() - start_time

        if result.returncode != 0:
            logger.error(f"❌ Failed: {name} ({elapsed:.1f}s)")
            if result.stderr:
                logger.error(f"   Stderr: {result.stderr[:500]}")
            return False

        logger.info(f"✅ Completed: {name} ({elapsed:.1f}s)")

        # Log stdout if any
        if result.stdout:
            for line in result.stdout.strip().split("\n")[-3:]:
                logger.info(f"   → {line}")

        return True

    except subprocess.TimeoutExpired:
        logger.error(f"⏰ Timeout: {name} exceeded 5 minute limit")
        return False

    except Exception as e:
        logger.error(f"❌ Error in {name}: {e}")
        return False


# ----------------------
# PIPELINE
# ----------------------
def run_pipeline():
    """Execute all pipeline steps in order."""

    steps = [
        ("data_collection/realtime_weather.py", "1. Weather Collection"),
        ("data_collection/realtime_rainfall.py", "2. Satellite Rainfall (GPM)"),
        ("data_collection/realtime_river.py", "3. River Level Monitoring"),
        ("data_collection/build_dataset.py", "4. Dataset Builder"),
    ]

    results = {}

    for script, name in steps:
        success = run_script(script, name)
        results[name] = "✅" if success else "❌"

        if not success:
            logger.warning(f"⚠️ Step failed: {name} — continuing with remaining steps")
            # Continue instead of stopping — collect what we can

    # Summary
    logger.info("=" * 50)
    logger.info("📋 PIPELINE SUMMARY")
    for step, status in results.items():
        logger.info(f"   {status} {step}")
    logger.info("=" * 50)

    all_success = all(s == "✅" for s in results.values())

    if all_success:
        logger.info("🎉 Pipeline completed successfully!")
    else:
        logger.warning("⚠️ Pipeline completed with some failures")

    return all_success


# ----------------------
# MAIN (RUN ONCE)
# ----------------------
if __name__ == "__main__":

    # ⚠️ FIX: Duplicate check runs FIRST — before any logging
    if already_ran_recently():
        sys.exit(0)  # Silent exit, no spam

    # Now we know this is a fresh run — log it
    logger.info(f"\n{'='*50}")
    logger.info(f"🔥 Running RAINWISE pipeline at {datetime.now()}")
    logger.info(f"{'='*50}")

    create_lock()

    try:
        pipeline_start = time.time()

        success = run_pipeline()

        elapsed = time.time() - pipeline_start
        logger.info(f"⏱ Total pipeline time: {elapsed:.1f}s")

        if success:
            update_last_run()
        else:
            logger.warning("❌ Pipeline had failures — check logs above")

    finally:
        remove_lock()