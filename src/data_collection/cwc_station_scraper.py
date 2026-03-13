from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

# Chrome options
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")

# Start driver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Open CWC Flood Forecast Portal
driver.get("https://ffs.indiawater.gov.in/")

time.sleep(10)

print("Portal opened successfully.")

input("Press Enter to close browser...")
driver.quit()