import os
import subprocess
import sys

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

PYTHON = sys.executable


def run_script(script_path, name):
    try:
        print(f"🔄 Running {name}...")
        subprocess.run(
            [PYTHON, script_path],
            check=True
        )
        print(f"✅ {name} done")
    except Exception as e:
        print(f"❌ Error running {name}: {e}")


if __name__ == "__main__":

    # Weather
    run_script(
        os.path.join(BASE_DIR, "data_collection", "realtime_weather.py"),
        "Weather"
    )

    # Rainfall
    run_script(
        os.path.join(BASE_DIR, "data_collection", "realtime_rainfall.py"),
        "Rainfall"
    )

    # River
    run_script(
        os.path.join(BASE_DIR, "data_collection", "realtime_river.py"),
        "River"
    )

    # Dataset build (FIXED PATH)
    run_script(
        os.path.join(BASE_DIR, "data_collection", "build_dataset.py"),
        "Dataset Build"
    )

    print("===================================")
    print("✅ Realtime pipeline complete")
    print("===================================")
