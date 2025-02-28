import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2, teamgamelog
from nba_api.stats.static import teams
from datetime import datetime, timedelta
import time
import os
import json
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../data/nba_data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs('../data/historical', exist_ok=True)
os.makedirs('../data/training', exist_ok=True)

def get_team_mapping():
    """Get mapping of team IDs to abbreviations"""
    nba_teams = teams.get_teams()
    return {team['id']: team['abbreviation'] for team in nba_teams}

def fetch_season_games(season):
    """Fetch all games for a specific season"""
    logger.info(f"Fetching games for season {season}")
    
    team_id_to_abbr = get_team_mapping()
    
    # Get all games for the season
    gamefinder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00"  # NBA
    )
    games_df = gamefinder.get_data_frames()[0]
    
    # Filter for completed games only
    games_df = games_df[games_df['WL'].isin(['W', 'L'])]
    
    # Create a unique game ID by grouping home and away games
    games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
    
    # Separate home and away games
    home_games = games_df[games_df['MATCHUP'].str.contains(' vs. ')]
    away_games = games_df[games_df['MATCHUP'].str.contains(' @ ')]
    
    processed_games = []
    
    # Total points to use as proxy for over/under line
    total_points_by_game = {}
    
    for _, game in games_df.iterrows():
        game_id = game['GAME_ID']
        if game_id not in total_points_by_game:
            total_points_by_game[game_id] = game['PTS']
        else:
            total_points_by_game[game_id] += game['PTS']
    
    # Process each home game
    logger.info(f"Processing {len(home_games)} home games")
    for idx, home_game in tqdm(home_games.iterrows(), total=len(home_games)):
        game_id = home_game['GAME_ID']
        
        # Find corresponding away game
        away_game_matches = away_games[away_games['GAME_ID'] == game_id]
        if len(away_game_matches) == 0:
            continue
        
        away_game = away_game_matches.iloc[0]
        
        # Get teams
        home_team_id = home_game['TEAM_ID']
        away_team_id = away_game['TEAM_ID']
        
        home_team = team_id_to_abbr.get(home_team_id, "Unknown")
        away_team = team_id_to_abbr.get(away_team_id, "Unknown")
        
        # Get game date
        game_date = home_game['GAME_DATE']
        
        # Get total points scored
        total_points = total_points_by_game.get(game_id, 0)
        
        # Fetch previous games to calculate team stats
        home_team_prev_games = games_df[(games_df['TEAM_ID'] == home_team_id) & 
                                        (games_df['GAME_DATE'] < game_date)].tail(10)
        away_team_prev_games = games_df[(games_df['TEAM_ID'] == away_team_id) & 
                                         (games_df['GAME_DATE'] < game_date)].tail(10)
        
        # Skip if not enough previous games
        if len(home_team_prev_games) < 5 or len(away_team_prev_games) < 5:
            continue
        
        # Calculate team stats
        home_wins = len(home_team_prev_games[home_team_prev_games['WL'] == 'W'])
        home_losses = len(home_team_prev_games[home_team_prev_games['WL'] == 'L'])
        
        away_wins = len(away_team_prev_games[away_team_prev_games['WL'] == 'W'])
        away_losses = len(away_team_prev_games[away_team_prev_games['WL'] == 'L'])
        
        home_ppg = home_team_prev_games['PTS'].mean()
        home_papg = home_team_prev_games['PTS'].mean() - home_team_prev_games['PLUS_MINUS'].mean()
        
        away_ppg = away_team_prev_games['PTS'].mean()
        away_papg = away_team_prev_games['PTS'].mean() - away_team_prev_games['PLUS_MINUS'].mean()
        
        # Estimate a reasonable over/under line based on team averages
        # In a real scenario, this would come from betting odds data
        expected_total = (home_ppg + away_ppg + home_papg + away_papg) / 2
        ou_line = round(expected_total, 1)
        
        # Determine actual over/under result (1 for Over, 0 for Under)
        ou_result = 1 if total_points > ou_line else 0
        
        # Compile game record
        game_record = {
            'GameID': game_id,
            'Date': game_date.strftime('%Y-%m-%d'),
            'Season': season,
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'HomeTeamWins': home_wins,
            'HomeTeamLosses': home_losses,
            'AwayTeamWins': away_wins,
            'AwayTeamLosses': away_losses,
            'HomeTeamPointsPerGame': home_ppg,
            'HomeTeamPointsAllowedPerGame': home_papg,
            'AwayTeamPointsPerGame': away_ppg,
            'AwayTeamPointsAllowedPerGame': away_papg,
            'HomeTeamInjuries': 0,  # Would require separate injury data
            'AwayTeamInjuries': 0,  # Would require separate injury data
            'OverUnderLine': ou_line,
            'HomeScore': home_game['PTS'],
            'AwayScore': away_game['PTS'],
            'TotalPoints': total_points,
            'OverUnderResult': ou_result
        }
        
        processed_games.append(game_record)
        
        # Add a small delay to avoid API rate limits
        time.sleep(0.01)
    
    logger.info(f"Processed {len(processed_games)} games for season {season}")
    return processed_games

def main():
    """Main function to collect historical NBA data"""
    # Seasons to collect (can adjust as needed)
    seasons = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23"]
    
    all_games = []
    
    for season in seasons:
        try:
            season_games = fetch_season_games(season)
            all_games.extend(season_games)
            
            # Save each season separately as backup
            season_df = pd.DataFrame(season_games)
            season_df.to_csv(f'../data/historical/nba_games_{season.replace("-", "_")}.csv', index=False)
            
            logger.info(f"Saved {len(season_games)} games for season {season}")
            
            # Add a delay between seasons to avoid API rate limits
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"Error processing season {season}: {e}")
    
    # Combine all seasons
    all_games_df = pd.DataFrame(all_games)
    
    # Save combined dataset
    all_games_df.to_csv('../data/historical/nba_historical_games.csv', index=False)
    
    # Create training dataset
    training_df = all_games_df.sample(frac=0.8, random_state=42)
    testing_df = all_games_df.drop(training_df.index)
    
    training_df.to_csv('../data/training/nba_training_data.csv', index=False)
    testing_df.to_csv('../data/training/nba_testing_data.csv', index=False)
    
    logger.info(f"Data collection complete. Total games: {len(all_games_df)}")
    logger.info(f"Training set: {len(training_df)} games")
    logger.info(f"Testing set: {len(testing_df)} games")

if __name__ == "__main__":
    main()
