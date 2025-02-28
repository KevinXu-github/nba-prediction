from app.data.collector import NBADataCollector
from dotenv import load_dotenv
import os
import pandas as pd

# Load environment variables
load_dotenv()

def test_data_collection():
    # Initialize the collector
    collector = NBADataCollector()
    
    print("Testing data collection and caching...")
    
    # Test getting NBA schedule
    print("\n1. Testing NBA Schedule:")
    schedule = collector.get_nba_schedule()
    if schedule:
        print("✓ Successfully retrieved NBA schedule")
        print(f"✓ Number of games: {len(schedule)}")
    else:
        print("✗ Failed to retrieve NBA schedule")
    
    # Test getting betting odds
    print("\n2. Testing Betting Odds:")
    odds = collector.get_betting_odds()
    if odds:
        print("✓ Successfully retrieved betting odds")
        print(f"✓ Number of games with odds: {len(odds)}")
    else:
        print("✗ Failed to retrieve betting odds")
    
    # Test preparing game data for model
    print("\n3. Testing Game Data Preparation:")
    game_data = collector.prepare_game_data_for_model()
    if game_data is not None:
        print("✓ Successfully prepared game data")
        if isinstance(game_data, pd.DataFrame):
            print(f"✓ Number of prepared games: {len(game_data)}")
    else:
        print("✗ Failed to prepare game data")
    
    # Check cache directory
    print("\n4. Checking Cache Directory:")
    cache_files = os.listdir(collector.cache_dir)
    print(f"✓ Cache directory: {collector.cache_dir}")
    print(f"✓ Cached files: {cache_files}")

if __name__ == "__main__":
    test_data_collection() 