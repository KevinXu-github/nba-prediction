from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import logging
import os
import json
from datetime import datetime

# Import our custom modules
from data.collector import NBADataCollector
from models.prediction import NBAPredictionModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NBA Parlay Predictor API",
    description="API for predicting optimal NBA parlays",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize data collector and prediction model
data_collector = NBADataCollector()
prediction_model = NBAPredictionModel()

# Load the model on startup
@app.on_event("startup")
async def startup_event():
    try:
        prediction_model.load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")

# Request and response models
class ParlayRequest(BaseModel):
    parlay_size: int = 3
    max_games: Optional[int] = 10
    min_confidence: Optional[float] = 0.6

class GamePrediction(BaseModel):
    game_id: str
    home_team: str
    away_team: str
    game_date: str
    prediction: str  # "Over" or "Under"
    confidence: float
    risk_level: str
    over_under_line: Optional[float] = None

class ParlayResponse(BaseModel):
    parlay_id: str
    games: List[GamePrediction]
    overall_confidence: float
    risk_level: str
    parlay_size: int
    created_at: str

# Routes
@app.get("/")
async def root():
    return {"message": "NBA Parlay Predictor API"}

@app.get("/api/games", response_model=List[Dict[str, Any]])
async def get_upcoming_games():
    """Get upcoming NBA games with betting odds"""
    try:
        # Collect and prepare the latest game data
        games_df = data_collector.prepare_game_data_for_model()
        
        if games_df is None or len(games_df) == 0:
            raise HTTPException(status_code=404, detail="No upcoming games found")
        
        # Convert DataFrame to list of dictionaries
        games = games_df.to_dict(orient='records')
        return games
    except Exception as e:
        logger.error(f"Error getting upcoming games: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict-parlay", response_model=ParlayResponse)
async def predict_parlay(request: ParlayRequest):
    """Predict optimal parlay based on specified number of legs"""
    try:
        # Collect and prepare the latest game data
        games_df = data_collector.prepare_game_data_for_model()
        
        if games_df is None or len(games_df) == 0:
            raise HTTPException(status_code=404, detail="No upcoming games found")
        
        # Make predictions for all games
        predictions = prediction_model.predict(games_df)
        
        if predictions is None:
            raise HTTPException(status_code=500, detail="Failed to make predictions")
        
        # Filter by minimum confidence if specified
        if request.min_confidence:
            predictions = predictions[predictions['Confidence'] >= request.min_confidence]
        
        # Check if we have enough games after filtering
        if len(predictions) < request.parlay_size:
            raise HTTPException(
                status_code=400, 
                detail=f"Not enough games with confidence >= {request.min_confidence} to create a {request.parlay_size}-leg parlay"
            )
        
        # Optimize the parlay
        optimal_parlay = prediction_model.optimize_parlay(predictions, num_legs=request.parlay_size)
        
        # Calculate overall parlay confidence
        overall_confidence = optimal_parlay['Confidence'].prod()
        
        # Determine overall risk level
        if overall_confidence >= 0.7:
            risk_level = "Low"
        elif overall_confidence >= 0.5:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        # Create response object
        games = []
        for _, game in optimal_parlay.iterrows():
            games.append(GamePrediction(
                game_id=str(game.get('GameID', '')),
                home_team=game.get('HomeTeam', ''),
                away_team=game.get('AwayTeam', ''),
                game_date=game.get('Date', ''),
                prediction=game.get('PredictionText', ''),
                confidence=float(game.get('Confidence', 0)),
                risk_level=game.get('RiskLevel', ''),
                over_under_line=float(game.get('OverUnderLine', 0)) if game.get('OverUnderLine') else None
            ))
        
        # Generate a unique parlay ID based on timestamp
        parlay_id = f"parlay-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        response = ParlayResponse(
            parlay_id=parlay_id,
            games=games,
            overall_confidence=float(overall_confidence),
            risk_level=risk_level,
            parlay_size=request.parlay_size,
            created_at=datetime.now().isoformat()
        )
        
        # Save the parlay for future reference
        save_parlay(response)
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting parlay: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def save_parlay(parlay: ParlayResponse):
    """Save parlay to disk for history tracking"""
    try:
        history_dir = "./data/history"
        os.makedirs(history_dir, exist_ok=True)
        
        file_path = os.path.join(history_dir, f"{parlay.parlay_id}.json")
        
        with open(file_path, 'w') as f:
            json.dump(parlay.dict(), f, indent=4)
            
        logger.info(f"Saved parlay {parlay.parlay_id} to history")
    except Exception as e:
        logger.error(f"Error saving parlay: {e}")

@app.get("/api/parlay-history", response_model=List[ParlayResponse])
async def get_parlay_history():
    """Get history of predicted parlays"""
    try:
        history_dir = "./data/history"
        if not os.path.exists(history_dir):
            return []
        
        parlays = []
        for file_name in os.listdir(history_dir):
            if file_name.endswith(".json"):
                with open(os.path.join(history_dir, file_name), 'r') as f:
                    parlay = json.load(f)
                    parlays.append(parlay)
        
        # Sort by creation date (newest first)
        parlays.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return parlays
    except Exception as e:
        logger.error(f"Error getting parlay history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/refresh-data")
async def refresh_data(background_tasks: BackgroundTasks):
    """Refresh NBA data in the background"""
    try:
        # Schedule data refresh as a background task
        background_tasks.add_task(data_collector.prepare_game_data_for_model)
        return {"message": "Data refresh scheduled"}
    except Exception as e:
        logger.error(f"Error scheduling data refresh: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Webhooks for integrating with external services (future feature)
@app.post("/api/webhooks/odds-update")
async def odds_update_webhook(background_tasks: BackgroundTasks):
    """Webhook for receiving odds updates from external services"""
    try:
        # Process the update and refresh data in the background
        background_tasks.add_task(data_collector.prepare_game_data_for_model)
        return {"message": "Odds update received and processing started"}
    except Exception as e:
        logger.error(f"Error processing odds update: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)