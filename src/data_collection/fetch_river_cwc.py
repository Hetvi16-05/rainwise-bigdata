from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

import pandas as pd
from datetime import datetime
from pathlib import Path
import time

OUT_FILE = "data/raw/realtime/river/realtime_river_log.csv"

Path("data/raw/realtime/river").mkdir(parents=True, exist_ok=True)


url = "https://ffs.india-water.gov.in/"

print("Opening browser...")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

driver.get(url)

time.sleep(5)


tables = driver.find_elements(By.TAG_NAME, "table")

rows = []

for table in tables:

    trs = table.find_elements(By.TAG_NAME, "tr")

    for tr in trs:

        tds = tr.find_elements(By.TAG_NAME, "td")

        if len(tds) < 5:
            continue

        state = tds[1].text

        if "GUJARAT" not in state.upper():
            continue

        station = tds[2].text
        level = tds[4].text

        rows.append(
            {
                "station": station,
                "state": state,
                "river_level": level,
                "timestamp": datetime.now(),
            }
        )


driver.quit()


df = pd.DataFrame(rows)


if df.empty:
    print("No Gujarat data found")
else:
    try:
        old = pd.read_csv(OUT_FILE)
        df = pd.concat([old, df])
    except:
        pass

    df.to_csv(OUT_FILE, index=False)

    print("Saved:", OUT_FILE)