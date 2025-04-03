# Incomplete Exception Handling
# Query: Create a function that connects to an API, fetches data, and processes it. Include error handling for API timeouts.

# WARNING: Using fallback example, actual code generation failed

import requests
import json

def fetch_data_from_api(url, timeout=5):
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()

def process_data(url):
    try:
        data = fetch_data_from_api(url)
        return data
    except Exception as e:
        print(f"Error: {e}")
        return None