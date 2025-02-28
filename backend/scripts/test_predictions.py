import pandas as pd
import numpy as np
import joblib
import os
import json
import logging
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../data/prediction_testing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_model_predictions():
    """Test the trained model on the test dataset"""
    logger.info("Testing model predictions")
    
    # Load the test data
    try:
        test_data = pd.read_csv('../data/training/nba_testing_data.csv')
        logger.info(f"Loaded {len(test_data)} test samples")
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        return False
    
    # Load the model and scaler
    try:
        model_path = '../data/models/nba_overunder_model_latest.joblib'
        scaler_path = '../data/models/scaler.joblib'
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        logger.info(f"Loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Error loading model or scaler: {e}")
        return False
    
    # Load the latest metrics to get the feature list
    try:
        metrics_files = [f for f in os.listdir('../data/models') if f.startswith('model_metrics_') and f.endswith('.json')]
        metrics_files.sort(reverse=True)
        
        if not metrics_files:
            logger.error("No metrics file found")
            return False
        
        with open(f'../data/models/{metrics_files[0]}', 'r') as f:
            metrics = json.load(f)
        
        features = metrics['features']
    except Exception as e:
        logger.error(f"Error loading model metrics: {e}")
        # Use default features if metrics file not found
        features = [
            'HomeTeamWins', 'HomeTeamLosses', 'AwayTeamWins', 'AwayTeamLosses',
            'HomeTeamPointsPerGame', 'HomeTeamPointsAllowedPerGame',
            'AwayTeamPointsPerGame', 'AwayTeamPointsAllowedPerGame',
            'HomeTeamInjuries', 'AwayTeamInjuries',
        ]
    
    # Prepare test data
    X_test = test_data[features]
    y_test = test_data['OverUnderResult']
    
    # Handle missing values
    X_test = X_test.fillna(X_test.mean())
    
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    logger.info(f"Test Results - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    # Add predictions to test data
    test_data['PredictedOverUnder'] = y_pred
    test_data['Confidence'] = [proba[pred] for pred, proba in zip(y_pred, y_pred_proba)]
    test_data['Correct'] = test_data['PredictedOverUnder'] == test_data['OverUnderResult']
    
    # Save predictions to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    predictions_path = f'../data/training/model_predictions_{timestamp}.csv'
    test_data.to_csv(predictions_path, index=False)
    
    logger.info(f"Saved predictions to {predictions_path}")
    
    # Analyze performance by confidence level
    logger.info("Performance by confidence level:")
    
    confidence_bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i in range(len(confidence_bins) - 1):
        lower = confidence_bins[i]
        upper = confidence_bins[i + 1]
        
        subset = test_data[(test_data['Confidence'] >= lower) & (test_data['Confidence'] < upper)]
        if len(subset) > 0:
            subset_accuracy = subset['Correct'].mean()
            logger.info(f"Confidence {lower:.1f}-{upper:.1f}: {len(subset)} predictions, Accuracy: {subset_accuracy:.4f}")
    
    return True

if __name__ == "__main__":
    test_model_predictions()
