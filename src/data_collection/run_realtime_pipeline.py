import os
from datetime import datetime

month = datetime.now().month

# monsoon months
monsoon = [6, 7, 8, 9, 10]

if month in monsoon:
    interval = 10
else:
    interval = 60

print("Month:", month)

# check last run time
log_file = "pipeline_last_run.txt"

run = True

if os.path.exists(log_file):

    with open(log_file) as f:
        last = f.read().strip()

    if last:
        last = datetime.fromisoformat(last)

        diff = (datetime.now() - last).total_seconds() / 60

        if diff < interval:
            run = False


if not run:
    print("Skipping run")
    exit()


# save time
with open(log_file, "w") as f:
    f.write(datetime.now().isoformat())


print("Running Weather...")
os.system("python src/data_collection/fetch_weather_realtime.py")

print("Running Rainfall...")
os.system("python src/data_collection/fetch_rainfall_realtime.py")

print("Running River...")
os.system("python src/data_collection/fetch_river_realtime.py")

print("Building Dataset...")
os.system("python src/data_collection/build_realtime_dataset.py")

print("Predicting Flood...")
os.system("python src/models/predict_realtime.py")

print("Done")