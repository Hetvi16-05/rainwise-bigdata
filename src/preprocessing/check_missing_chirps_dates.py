import os
import re
from datetime import datetime, timedelta

# Path to merged rainfall folder
DATA_DIR = "data/processed/rainfall/chirps_india_daily_full"

# Regex pattern to extract date (YYYY.MM.DD)
date_pattern = re.compile(r"(\d{4}\.\d{2}\.\d{2})")

dates_found = []

# Extract dates from filenames
for filename in os.listdir(DATA_DIR):
    match = date_pattern.search(filename)
    if match:
        date_str = match.group(1)
        date_obj = datetime.strptime(date_str, "%Y.%m.%d")
        dates_found.append(date_obj)

# Remove duplicates
unique_dates = sorted(set(dates_found))

print("Total files found:", len(dates_found))
print("Unique dates:", len(unique_dates))

if not unique_dates:
    print("No valid dates found!")
    exit()

start_date = unique_dates[0]
end_date = unique_dates[-1]

print("Start date:", start_date.date())
print("End date:", end_date.date())

# Generate complete date range
expected_dates = []
current = start_date
while current <= end_date:
    expected_dates.append(current)
    current += timedelta(days=1)

# Find missing dates
missing_dates = sorted(set(expected_dates) - set(unique_dates))

print("Total expected days:", len(expected_dates))
print("Missing days:", len(missing_dates))

if missing_dates:
    print("\nMissing Dates:")
    for d in missing_dates[:20]:  # show first 20
        print(d.date())
else:
    print("No missing dates 🎉")

# Check duplicates
if len(dates_found) != len(unique_dates):
    print("Duplicate files detected!")