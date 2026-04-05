import os
import re

FOLDER = "data/raw/rainfall/chirps_all"

date_pattern = re.compile(r"(\d{4}\.\d{2}\.\d{2})")

files = os.listdir(FOLDER)

seen = {}
removed = 0

for f in files:

    match = date_pattern.search(f)
    if not match:
        continue

    date = match.group(1)

    path = os.path.join(FOLDER, f)

    if date not in seen:
        seen[date] = f
    else:
        # duplicate found → remove
        print("Removing duplicate:", f)
        os.remove(path)
        removed += 1

print("Done")
print("Remaining:", len(seen))
print("Removed:", removed)