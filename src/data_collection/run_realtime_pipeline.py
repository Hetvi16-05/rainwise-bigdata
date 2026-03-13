import os
import subprocess
import sys
from datetime import datetime


BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

PYTHON = sys.executable


STATE_FILE = os.path.join(
    BASE_DIR,
    "pipeline_state.txt"
)


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# -----------------------------
# schedule logic (stable)
# -----------------------------

def should_run():

    now_dt = datetime.now()
    month = now_dt.month

    print("Month:", month)

    # Monsoon months → every 30 min
    if month in [6, 7, 8, 9]:
        print("Monsoon mode")
        return True

    # Other months → every 60 min
    # launchd runs every 30 min
    # so run every 2nd call

    count = 0

    if os.path.exists(STATE_FILE):

        try:
            with open(STATE_FILE, "r") as f:
                count = int(f.read().strip())
        except:
            count = 0

    count += 1

    with open(STATE_FILE, "w") as f:
        f.write(str(count))

    if count % 2 == 1:
        print("Normal mode → run")
        return True
    else:
        print("Normal mode → skip")
        return False


# -----------------------------
# run script
# -----------------------------

def run_script(script_path, name):

    try:

        print(f"[{now()}] Running {name}")

        subprocess.run(
            [PYTHON, script_path],
            check=True
        )

        print(f"[{now()}] Done {name}")

    except Exception as e:

        print(f"Error {name}: {e}")


# -----------------------------
# pipeline
# -----------------------------

def run_pipeline():

    run_script(
        os.path.join(BASE_DIR, "data_collection", "realtime_weather.py"),
        "Weather"
    )

    run_script(
        os.path.join(BASE_DIR, "data_collection", "realtime_rainfall.py"),
        "Rainfall"
    )

    run_script(
        os.path.join(BASE_DIR, "data_collection", "realtime_river.py"),
        "River"
    )

    run_script(
        os.path.join(BASE_DIR, "data_collection", "build_dataset.py"),
        "Dataset"
    )

    print("Pipeline done", now())


# -----------------------------
# main
# -----------------------------

if __name__ == "__main__":

    if not should_run():
        print("Skip run")
        exit()

    run_pipeline()