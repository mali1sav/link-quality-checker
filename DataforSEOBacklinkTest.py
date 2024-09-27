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

def fetch_backlink_data(domain: str):
    url = "https://api.dataforseo.com/v3/backlinks/summary/live"
    headers = {
        'Authorization': f'Basic {DATAFORSEO_API_KEY}',
        'Content-Type': 'application/json'
    }
    payload = json.dumps([{
        "target": domain,
        "internal_list_limit": 10,
        "backlinks_status_type": "all",
        "include_subdomains": True,
        "backlinks_filters": ["dofollow", "=", True]
    }]) 

    try:
        response = requests.post(url, headers=headers, data=payload)
        if response.status_code == 200:
            result = response.json()
            return result['tasks'][0]['result'][0]
        else:
            raise Exception(f"DataForSEO API request failed with status {response.status_code}: {response.text}")
    except requests.RequestException as e:
        raise Exception(f"Network error when fetching backlink data: {str(e)}")

# Test the backlink API
test_domain = "example.com"

try:
    print(f"\nFetching backlink data for {test_domain}...")
    backlink_data = fetch_backlink_data(test_domain)    
    print("Backlink data fetched successfully.")
    print(backlink_data)
except Exception as e:
    print(f"Error: {str(e)}")
    print("\nPossible issues:")
    print("1. The API key in your .env file might be incorrect or expired.")
    print("2. Your DataForSEO account might not have access to the backlink API.")
    print("3. There might be an issue with your DataForSEO account status.")
    print("\nPlease check your API key and account status at https://app.dataforseo.com/api-dashboard")
    print("If the issue persists, please contact DataForSEO support for assistance.")