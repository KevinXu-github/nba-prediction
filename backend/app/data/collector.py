import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NBADataCollector:
    """
    A class to collect NBA game data, player stats, and betting odds
    """
    
    def __init__(self, cache_dir=None):
        """Initialize the data collector with API keys and cache directory"""
        # API keys should be stored in environment variables
        self.nba_api_key = os.environ.get("NBA_API_KEY", "")
        self.odds_api_key = os.environ.get("ODDS_API_KEY", "")
        
        # Set up cache directory with absolute path
        if cache_dir is None:
            # Use the directory where this file is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.cache_dir = os.path.join(current_dir, "..", "..", "data", "cache")
        else:
            self.cache_dir = cache_dir
            
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"Cache directory set to: {self.cache_dir}")
        
        # Base URLs for the APIs
        self.nba_base_url = "https://api.sportsdata.io/v3/nba"
        self.odds_base_url = "https://api.the-odds-api.com/v4"
        
        logger.info("NBA Data Collector initialized")
    
    def get_nba_schedule(self, season_year=None):
        """Get the NBA schedule for a specific season"""
        if not season_year:
            # Default to current season
            current_date = datetime.now()
            if current_date.month < 7:  # NBA season typically ends in June
                season_year = current_date.year
            else:
                season_year = current_date.year + 1
                
        endpoint = f"{self.nba_base_url}/scores/json/Games/{season_year}"
        
        cache_file = os.path.join(self.cache_dir, f"nba_schedule_{season_year}.json")
        
        # Check if we have a cached version
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                logger.info(f"Loading NBA schedule for {season_year} from cache")
                return json.load(f)
        
        # Otherwise, fetch from API
        headers = {
            "Ocp-Apim-Subscription-Key": self.nba_api_key
        }
        
        try:
            logger.info(f"Fetching NBA schedule for {season_year}")
            response = requests.get(endpoint, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            # Cache the result
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching NBA schedule: {e}")
            return None
    
    def get_player_stats(self, season_year=None):
        """Get player statistics for a specific season"""
        if not season_year:
            current_date = datetime.now()
            if current_date.month < 7:
                season_year = current_date.year
            else:
                season_year = current_date.year + 1
        
        endpoint = f"{self.nba_base_url}/stats/json/PlayerSeasonStats/{season_year}"
        cache_file = os.path.join(self.cache_dir, f"player_stats_{season_year}.json")
        
        # Check cache
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                logger.info(f"Loading player stats for {season_year} from cache")
                return json.load(f)
        
        # Fetch from API
        headers = {
            "Ocp-Apim-Subscription-Key": self.nba_api_key
        }
        
        try:
            logger.info(f"Fetching player stats for {season_year}")
            response = requests.get(endpoint, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            # Cache the result
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching player stats: {e}")
            return None
    
    def get_team_stats(self, season_year=None):
        """Get team statistics for a specific season"""
        if not season_year:
            current_date = datetime.now()
            if current_date.month < 7:
                season_year = current_date.year
            else:
                season_year = current_date.year + 1
        
        endpoint = f"{self.nba_base_url}/stats/json/TeamSeasonStats/{season_year}"
        cache_file = os.path.join(self.cache_dir, f"team_stats_{season_year}.json")
        
        # Check cache
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                logger.info(f"Loading team stats for {season_year} from cache")
                return json.load(f)
        
        # Fetch from API
        headers = {
            "Ocp-Apim-Subscription-Key": self.nba_api_key
        }
        
        try:
            logger.info(f"Fetching team stats for {season_year}")
            response = requests.get(endpoint, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            # Cache the result
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching team stats: {e}")
            return None
    
    def get_betting_odds(self, sport="basketball_nba", regions="us", markets="totals"):
        """Get current betting odds for NBA games"""
        endpoint = f"{self.odds_base_url}/sports/{sport}/odds"
        
        params = {
            "apiKey": self.odds_api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": "american"
        }
        
        # Current odds should not be cached for long periods
        cache_file = os.path.join(self.cache_dir, f"betting_odds_{sport}_{datetime.now().strftime('%Y%m%d')}.json")
        
        # Check if cache is recent (less than 1 hour old)
        if os.path.exists(cache_file):
            file_time = os.path.getmtime(cache_file)
            if (time.time() - file_time) < 3600:  # 1 hour in seconds
                with open(cache_file, 'r') as f:
                    logger.info(f"Loading recent betting odds from cache")
                    return json.load(f)
        
        try:
            logger.info(f"Fetching current betting odds for {sport}")
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Cache the result
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching betting odds: {e}")
            return None
    
    def get_injuries(self):
        """Get current NBA injuries"""
        endpoint = f"{self.nba_base_url}/scores/json/Injuries"
        
        cache_file = os.path.join(self.cache_dir, f"injuries_{datetime.now().strftime('%Y%m%d')}.json")
        
        # Check if cache is recent (less than 12 hours old)
        if os.path.exists(cache_file):
            file_time = os.path.getmtime(cache_file)
            if (time.time() - file_time) < 43200:  # 12 hours in seconds
                with open(cache_file, 'r') as f:
                    logger.info(f"Loading recent injury data from cache")
                    return json.load(f)
        
        headers = {
            "Ocp-Apim-Subscription-Key": self.nba_api_key
        }
        
        try:
            logger.info("Fetching current NBA injuries")
            response = requests.get(endpoint, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            # Cache the result
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching injury data: {e}")
            return None
    
    def prepare_game_data_for_model(self, days_ahead=7):
        """
        Prepare a dataset with upcoming games and relevant statistics for model prediction
        """
        logger.info("Preparing game data for model prediction")
        
        # Get schedule data
        schedule = self.get_nba_schedule()
        if not schedule:
            return None
        
        # Get team stats
        team_stats = self.get_team_stats()
        if not team_stats:
            return None
        
        # Get current odds
        odds = self.get_betting_odds()
        if not odds:
            return None
        
        # Get current injuries (optional)
        injuries = self.get_injuries()
        # Don't fail if injuries data is not available
        
        # Convert schedule to DataFrame
        schedule_df = pd.DataFrame(schedule)
        
        # Filter for upcoming games within days_ahead
        current_date = datetime.now()
        schedule_df['DateTime'] = pd.to_datetime(schedule_df['Day'])
        schedule_df = schedule_df[
            (schedule_df['DateTime'] >= current_date) & 
            (schedule_df['DateTime'] <= current_date + timedelta(days=days_ahead))
        ]
        
        # Convert odds to DataFrame
        odds_df = pd.DataFrame(odds)
        
        # Convert team stats to DataFrame
        team_stats_df = pd.DataFrame(team_stats)
        
        # Merge data
        merged_data = schedule_df.merge(
            odds_df,
            left_on=['HomeTeam', 'AwayTeam'],
            right_on=['home_team', 'away_team'],
            how='left'
        )
        
        # Add team statistics
        merged_data = merged_data.merge(
            team_stats_df,
            left_on='HomeTeam',
            right_on='Team',
            how='left',
            suffixes=('', '_home')
        )
        
        merged_data = merged_data.merge(
            team_stats_df,
            left_on='AwayTeam',
            right_on='Team',
            how='left',
            suffixes=('', '_away')
        )
        
        # Add injury information if available
        if injuries:
            injuries_df = pd.DataFrame(injuries)
            # Process injuries here...
        
        return merged_data