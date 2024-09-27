import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DATAFORSEO_API_KEY = os.getenv("DATAFORSEO_API_KEY")

# Check for API credentials
if not DATAFORSEO_API_KEY:
    print("DataForSEO API key is missing. Please check your .env file and ensure DATAFORSEO_API_KEY is set.")
    exit(1)

# Print partially masked API key for debugging
print(f"API Key (masked): {DATAFORSEO_API_KEY[:5]}...{DATAFORSEO_API_KEY[-5:]}")

def fetch_serp_data(domain: str):
    url = "https://api.dataforseo.com/v3/serp/google/organic/live/advanced"
    headers = {
        'Authorization': f'Basic {DATAFORSEO_API_KEY}',
        'Content-Type': 'application/json'
    }
    payload = json.dumps([{
        "keyword": f"site:{domain}",
        "location_code": 2840,
        "language_code": "en",
        "device": "desktop",
        "os": "windows"
    }])
    
    print("\nDebug Info:")
    print(f"URL: {url}")
    print(f"Headers: {headers}")
    print(f"Payload: {payload}")
    
    try:
        response = requests.post(url, headers=headers, data=payload)
        print(f"\nResponse Status Code: {response.status_code}")
        print(f"Response Headers: {response.headers}")
        if response.status_code == 200:
            result = response.json()
            return result['tasks'][0]['result'][0]['items'][:5]  # Return only the first 5 results
        else:
            print(f"Full Response Text: {response.text}")
            raise Exception(f"DataForSEO SERP API request failed with status {response.status_code}: {response.text}")
    except requests.RequestException as e:
        raise Exception(f"Network error when fetching SERP data: {str(e)}")

# Test the SERP API
test_domain = "example.com"

try:
    print(f"\nFetching SERP data for {test_domain}...")
    serp_data = fetch_serp_data(test_domain)
    print("SERP data fetched successfully.")
    print("First 5 results:")
    for item in serp_data:
        print(f"- {item.get('title', 'No title')} ({item.get('url', 'No URL')})")
except Exception as e:
    print(f"An error occurred: {str(e)}")
    print("\nPossible issues:")
    print("1. The API key in your .env file might be incorrect or expired.")
    print("2. Your DataForSEO account might not have access to the SERP API.")
    print("3. There might be an issue with your DataForSEO account status.")
    print("\nPlease check your API key and account status at https://app.dataforseo.com/api-dashboard")
    print("If the issue persists, please contact DataForSEO support for assistance.")