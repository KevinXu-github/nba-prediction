import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_nba_api():
    nba_api_key = os.environ.get("NBA_API_KEY")
    url = "https://api.sportsdata.io/v3/nba/scores/json/Games/2024"
    headers = {
        "Ocp-Apim-Subscription-Key": nba_api_key
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        print("NBA API Test:")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()[:2]}")  # Show first 2 items
    except Exception as e:
        print(f"NBA API Error: {str(e)}")

def test_odds_api():
    odds_api_key = os.environ.get("ODDS_API_KEY")
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    params = {
        "apiKey": odds_api_key,
        "regions": "us",
        "markets": "totals",
        "oddsFormat": "american"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        print("\nOdds API Test:")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()[:2]}")  # Show first 2 items
    except Exception as e:
        print(f"Odds API Error: {str(e)}")

if __name__ == "__main__":
    test_nba_api()
    test_odds_api() 