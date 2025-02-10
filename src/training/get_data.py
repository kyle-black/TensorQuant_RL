import requests
import time
import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import os
load_dotenv()

# API details
API_KEY = os.getenv('DATA_API')
BASE_URL = "https://financialmodelingprep.com/api/v3/historical-chart/1min"
SYMBOL = "USDJPY"  # Replace with desired Forex pair

# Date range for historical data
START_DATE = datetime(2010, 1, 1)
END_DATE = datetime(2023, 12, 31)

# Function to fetch 1-minute data for a given time range
def fetch_data(symbol, start_date, end_date, api_key):
    url = f"{BASE_URL}/{symbol}?from={start_date.strftime('%Y-%m-%d')}&to={end_date.strftime('%Y-%m-%d')}&apikey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data: {response.status_code}, {response.text}")
        return None

# Main loop for bulk import
def bulk_import(symbol, start_date, end_date, api_key):
    current_date = start_date
    final_data = []

    while current_date <= end_date:
        next_date = current_date + timedelta(days=3)  # Adjust batch size if needed
        print(f"Fetching data from {current_date} to {next_date}...")
        
        data = fetch_data(symbol, current_date, next_date, api_key)
        if data:
            final_data.extend(data)

        current_date = next_date
        time.sleep(1)  # To avoid hitting API rate limits

    return final_data

# Save data to JSON file
def save_to_file(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f)
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    data = bulk_import(SYMBOL, START_DATE, END_DATE, API_KEY)
    save_to_file(data, f"training_data/forex/{SYMBOL}_1min.json")