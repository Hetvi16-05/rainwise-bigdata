import os
import re
from datetime import datetime, timedelta

DATA_DIRS = ["data/raw/rainfall/chirps_india_daily"]

START_DATE = datetime(2000, 1, 1)
END_DATE = datetime(2025, 12, 31)

date_pattern = re.compile(r"(\d{4}_\d{2}_\d{2})")

dates_found = set()

for folder in DATA_DIRS:
    if not os.path.exists(folder):
        print("Folder not found:", folder)
        continue

    for filename in os.listdir(folder):
        match = date_pattern.search(filename)
        if match:
            d = datetime.strptime(match.group(1), "%Y_%m_%d")
            dates_found.add(d)

print("Total files found:", len(dates_found))

expected_dates = set()
d = START_DATE
while d <= END_DATE:
    expected_dates.add(d)
    d += timedelta(days=1)

missing = sorted(expected_dates - dates_found)

print("Expected days:", len(expected_dates))
print("Found days:", len(dates_found))
print("Missing days:", len(missing))

if missing:
    print("First 10 missing:")
    for m in missing[:10]:
        print(m.strftime("%Y-%m-%d"))

    # ✅ save file
    with open("missing_dates.txt", "w") as f:
        for m in missing:
            f.write(m.strftime("%Y-%m-%d") + "\n")

else:
    print("No missing dates 🎉")