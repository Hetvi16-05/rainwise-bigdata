import os
import re
from datetime import datetime, timedelta

# Folder containing merged CHIRPS data
DATA_DIR = "data/processed/rainfall/chirps_india_daily_full"

# Output file to store missing dates
OUTPUT_FILE = "missing_dates.txt"

# Regex pattern for YYYY.MM.DD
date_pattern = re.compile(r"(\d{4}\.\d{2}\.\d{2})")

dates_found = []

# Extract dates from filenames
for filename in os.listdir(DATA_DIR):
    match = date_pattern.search(filename)
    if match:
        date_str = match.group(1)
        date_obj = datetime.strptime(date_str, "%Y.%m.%d")
        dates_found.append(date_obj)

# Remove duplicates and sort
unique_dates = sorted(set(dates_found))

if not unique_dates:
    print("No valid dates found.")
    exit()

start_date = unique_dates[0]
end_date = unique_dates[-1]

print("Start date:", start_date.date())
print("End date:", end_date.date())
print("Total available files:", len(unique_dates))

# Generate full expected date range
expected_dates = []
current = start_date
while current <= end_date:
    expected_dates.append(current)
    current += timedelta(days=1)

# Find missing
missing_dates = sorted(set(expected_dates) - set(unique_dates))

print("Total expected days:", len(expected_dates))
print("Missing days:", len(missing_dates))

# Save missing dates to file
with open(OUTPUT_FILE, "w") as f:
    for d in missing_dates:
        f.write(d.strftime("%Y-%m-%d") + "\n")

print(f"\nMissing dates saved to {OUTPUT_FILE}")

if missing_dates:
    print("\nFirst 10 missing dates:")
    for d in missing_dates[:10]:
        print(d.date())
else:
    print("No missing dates 🎉")