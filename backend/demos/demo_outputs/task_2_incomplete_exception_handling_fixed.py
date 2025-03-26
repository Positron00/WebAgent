# Incomplete Exception Handling (Fixed Version)
# Query: Create a function that connects to an API, fetches data, and processes it. Include error handling for API timeouts.

import requests
import json
import time

def fetch_data_from_api(url, timeout=5, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            print(f"Timeout error on attempt {attempt+1}/{max_retries}")
            if attempt < max_retries - 1:
                # Exponential backoff before retry
                wait_time = 2 ** attempt
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise TimeoutError(f"API request to {url} timed out after {max_retries} attempts")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"API request failed: {str(e)}")

def process_data(url):
    try:
        data = fetch_data_from_api(url)
        # Process the data
        result = {
            "status": "success",
            "record_count": len(data) if isinstance(data, list) else 1,
            "data": data
        }
        return result
    except TimeoutError as e:
        print(f"Timeout error: {e}")
        return {"status": "error", "error": "timeout", "message": str(e)}
    except ConnectionError as e:
        print(f"Connection error: {e}")
        return {"status": "error", "error": "connection", "message": str(e)}
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {"status": "error", "error": "unknown", "message": str(e)}