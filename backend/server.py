from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import os
import json
import random
from datetime import datetime, timedelta
import sys
from dotenv import load_dotenv


# Load .env file
load_dotenv()

# Print API keys to console (will be visible in the server logs)
print(f"NBA API Key: {'*' * (len(os.environ.get('NBA_API_KEY', '')) - 4)}...{os.environ.get('NBA_API_KEY', '')[-4:] if os.environ.get('NBA_API_KEY', '') else 'Not set'}")
print(f"Odds API Key: {'*' * (len(os.environ.get('ODDS_API_KEY', '')) - 4)}...{os.environ.get('ODDS_API_KEY', '')[-4:] if os.environ.get('ODDS_API_KEY', '') else 'Not set'}")


# Import your existing collector class
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.data.collector import NBADataCollector
from app.models.prediction import NBAPredictionModel

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the data collector
collector = NBADataCollector()
model = NBAPredictionModel()

# Create data folders
os.makedirs("./data/history", exist_ok=True)
os.makedirs("./data/models", exist_ok=True)

# Define request model
class ParlayRequest(BaseModel):
    parlay_size: int = 3
    min_confidence: Optional[float] = 0.6

# Generate mock games as fallback
def generate_mock_games(num_games=8):
    teams = [
        "LAL", "BOS", "GSW", "MIL", "PHX", "PHI", "BKN", "DEN",
        "LAC", "MIA", "DAL", "ATL", "CHI", "TOR", "CLE", "NYK"
    ]
    
    mock_games = []
    today = datetime.now()
    
    for i in range(num_games):
        # Get two random different teams
        home_idx = random.randint(0, len(teams) - 1)
        away_idx = random.randint(0, len(teams) - 1)
        while away_idx == home_idx:
            away_idx = random.randint(0, len(teams) - 1)
        
        home_team = teams[home_idx]
        away_team = teams[away_idx]
        
        # Random game date within the next week
        game_date = today + timedelta(days=random.randint(1, 7))
        
        mock_games.append({
            "GameID": f"mock-{i + 1}",
            "Date": game_date.isoformat(),
            "HomeTeam": home_team,
            "AwayTeam": away_team,
            "Stadium": f"{home_team} Arena",
            "HomeTeamWins": random.randint(10, 50),
            "HomeTeamLosses": random.randint(10, 50),
            "AwayTeamWins": random.randint(10, 50),
            "AwayTeamLosses": random.randint(10, 50),
            "HomeTeamPointsPerGame": 100 + random.random() * 20,
            "HomeTeamPointsAllowedPerGame": 100 + random.random() * 20,
            "AwayTeamPointsPerGame": 100 + random.random() * 20,
            "AwayTeamPointsAllowedPerGame": 100 + random.random() * 20,
            "HomeTeamInjuries": random.randint(0, 3),
            "AwayTeamInjuries": random.randint(0, 3),
            "OverUnderLine": 210 + random.randint(0, 30)
        })
    
    return mock_games

# API Endpoints
@app.get("/")
async def root():
    return {"message": "NBA Parlay Predictor API"}

@app.get("/api/games", response_model=List[Dict[str, Any]])
async def get_upcoming_games():
    """Get upcoming NBA games with betting odds"""
    try:
        print("Attempting to fetch game data...")
        
        # Use your existing collector to get game data
        games_df = collector.prepare_game_data_for_model()
        
        if games_df is None or len(games_df) == 0:
            print("No games data available, using mock data")
            return generate_mock_games()
        
        print(f"Successfully fetched {len(games_df)} games")
        
        # Convert DataFrame to list of dictionaries
        games = games_df.to_dict(orient='records')
        
        # Map SportsData.io fields to the expected field names for each game
        for game in games:
            # Map date fields
            if 'Day' in game and game['Day'] is not None:
                game['Date'] = game['Day']
            elif 'DateTime' in game and game['DateTime'] is not None:
                game['Date'] = game['DateTime']
            
            # Map team records
            # For team stats, the API provides 'Wins' and 'Losses' for home team
            # and 'Wins_away' and 'Losses_away' for away team
            if 'Wins' in game:
                game['HomeTeamWins'] = game['Wins']
            else:
                game['HomeTeamWins'] = random.randint(20, 50)
                
            if 'Losses' in game:
                game['HomeTeamLosses'] = game['Losses']
            else:
                game['HomeTeamLosses'] = random.randint(10, 40)
                
            if 'Wins_away' in game:
                game['AwayTeamWins'] = game['Wins_away']
            else:
                game['AwayTeamWins'] = random.randint(20, 50)
                
            if 'Losses_away' in game:
                game['AwayTeamLosses'] = game['Losses_away']
            else:
                game['AwayTeamLosses'] = random.randint(10, 40)
            
            # Map team points per game
            # In SportsData.io, the team stats have 'Points' and 'Points_away'
            if 'Points' in game:
                game['HomeTeamPointsPerGame'] = game['Points']
            else:
                game['HomeTeamPointsPerGame'] = round(100 + random.random() * 20, 1)
                
            if 'Points_away' in game:
                game['AwayTeamPointsPerGame'] = game['Points_away']
            else:
                game['AwayTeamPointsPerGame'] = round(100 + random.random() * 20, 1)
            
            # For points allowed, we might need to calculate or use mock data
            # Teams' defensive stats might be in different fields
            game['HomeTeamPointsAllowedPerGame'] = round(100 + random.random() * 20, 1)
            game['AwayTeamPointsAllowedPerGame'] = round(100 + random.random() * 20, 1)
            
            # Add injury data (SportsData.io provides an Injuries endpoint but it's separate)
            game['HomeTeamInjuries'] = random.randint(0, 3)
            game['AwayTeamInjuries'] = random.randint(0, 3)
            
            # Map over/under line
            if 'OverUnder' in game:
                game['OverUnderLine'] = game['OverUnder']
            else:
                game['OverUnderLine'] = 210 + random.randint(0, 30)
            
            # Handle stadium information
            if 'StadiumID' in game and not ('Stadium' in game and game['Stadium']):
                # If we have a StadiumID but no Stadium name, create a placeholder
                game['Stadium'] = f"{game['HomeTeam']} Arena"
        
        print(f"Returning {len(games)} games with all required fields added/mapped")
        return games
    except Exception as e:
        import traceback
        print(f"Error getting upcoming games: {str(e)}")
        print(traceback.format_exc())
        return generate_mock_games()

@app.post("/api/predict-parlay")
async def predict_parlay(request: ParlayRequest):
    """Predict optimal parlay based on specified number of legs"""
    try:
        # Get game data
        games_df = collector.prepare_game_data_for_model()
        
        if games_df is None or len(games_df) == 0:
            raise HTTPException(status_code=404, detail="No upcoming games found")
        
        # Ensure DataFrame has all required features
        required_features = [
            'HomeTeamWins', 'HomeTeamLosses', 'AwayTeamWins', 'AwayTeamLosses',
            'HomeTeamPointsPerGame', 'HomeTeamPointsAllowedPerGame',
            'AwayTeamPointsPerGame', 'AwayTeamPointsAllowedPerGame',
            'HomeTeamInjuries', 'AwayTeamInjuries', 'OverUnderLine'
        ]
        
        # Map known fields
        field_mapping = {
            'Wins': 'HomeTeamWins',
            'Losses': 'HomeTeamLosses',
            'Wins_away': 'AwayTeamWins',
            'Losses_away': 'AwayTeamLosses',
            'Points': 'HomeTeamPointsPerGame',
            'Points_away': 'AwayTeamPointsPerGame',
            'OverUnder': 'OverUnderLine'
        }
        
        # Apply mappings for existing fields
        for source, target in field_mapping.items():
            if source in games_df.columns and target not in games_df.columns:
                games_df[target] = games_df[source]
        
        # Add missing required fields
        for feature in required_features:
            if feature not in games_df.columns:
                if 'Wins' in feature:
                    games_df[feature] = [random.randint(20, 50) for _ in range(len(games_df))]
                elif 'Losses' in feature:
                    games_df[feature] = [random.randint(10, 40) for _ in range(len(games_df))]
                elif 'PerGame' in feature:
                    games_df[feature] = [round(100 + random.random() * 20, 1) for _ in range(len(games_df))]
                elif 'Injuries' in feature:
                    games_df[feature] = [random.randint(0, 3) for _ in range(len(games_df))]
                elif feature == 'OverUnderLine' and 'OverUnder' in games_df.columns:
                    games_df[feature] = games_df['OverUnder']
                elif feature == 'OverUnderLine':
                    games_df[feature] = [210 + random.randint(0, 30) for _ in range(len(games_df))]
        
        # Load the model if not already loaded
        if model.model is None:
            model.load_model()
        
        # Make predictions
        predictions = model.predict(games_df)
        
        if predictions is None:
            raise HTTPException(status_code=500, detail="Failed to make predictions")
        
        # Filter by minimum confidence
        if request.min_confidence:
            predictions = predictions[predictions['Confidence'] >= request.min_confidence]
        
        # Check if we have enough games
        if len(predictions) < request.parlay_size:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough games with confidence >= {request.min_confidence} to create a {request.parlay_size}-leg parlay"
            )
        
        # Optimize the parlay
        optimal_parlay = model.optimize_parlay(predictions, num_legs=request.parlay_size)
        
        # Calculate overall confidence
        overall_confidence = optimal_parlay['Confidence'].prod()
        
        # Determine risk level
        if overall_confidence >= 0.7:
            risk_level = "Low"
        elif overall_confidence >= 0.5:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        # Create game predictions
        games = []
        for _, game in optimal_parlay.iterrows():
            games.append({
                "game_id": str(game.get('GameID', '')),
                "home_team": game.get('HomeTeam', ''),
                "away_team": game.get('AwayTeam', ''),
                "game_date": game.get('Date', ''),
                "prediction": game.get('PredictionText', ''),
                "confidence": float(game.get('Confidence', 0)),
                "risk_level": game.get('RiskLevel', ''),
                "over_under_line": float(game.get('OverUnderLine', 0)) if game.get('OverUnderLine') else None
            })
        
        # Create parlay response
        parlay_id = f"parlay-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        parlay = {
            "parlay_id": parlay_id,
            "games": games,
            "overall_confidence": float(overall_confidence),
            "risk_level": risk_level,
            "parlay_size": request.parlay_size,
            "created_at": datetime.now().isoformat()
        }
        
        # Save to history
        try:
            history_path = f"./data/history/{parlay_id}.json"
            with open(history_path, "w") as f:
                json.dump(parlay, f, indent=2)
        except Exception as e:
            print(f"Error saving history: {e}")
        
        return parlay
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/parlay-history")
async def get_parlay_history():
    """Get history of predicted parlays"""
    try:
        history_dir = "./data/history"
        if not os.path.exists(history_dir):
            return []
        
        parlays = []
        for file_name in os.listdir(history_dir):
            if file_name.endswith(".json"):
                try:
                    with open(os.path.join(history_dir, file_name), 'r') as f:
                        parlay = json.load(f)
                        parlays.append(parlay)
                except Exception:
                    continue
        
        # Sort by creation date (newest first)
        parlays.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return parlays
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/refresh-data")
async def refresh_data():
    """Refresh NBA data in the background"""
    try:
        # Force refresh by clearing cache
        collector.prepare_game_data_for_model(force_refresh=True)
        return {"message": "Data refresh scheduled"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)