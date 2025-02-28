import pandas as pd
import numpy as np
import os
import logging
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
import json
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../data/model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs('../data/models', exist_ok=True)

def train_model():
    """Train the NBA over/under prediction model"""
    logger.info("Starting model training process")
    
    # Load training data
    try:
        train_data = pd.read_csv('../data/training/nba_training_data.csv')
        test_data = pd.read_csv('../data/training/nba_testing_data.csv')
        logger.info(f"Loaded {len(train_data)} training samples and {len(test_data)} testing samples")
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        return False
    
    # Define features to use for training
    features = [
        'HomeTeamWins', 'HomeTeamLosses', 'AwayTeamWins', 'AwayTeamLosses',
        'HomeTeamPointsPerGame', 'HomeTeamPointsAllowedPerGame',
        'AwayTeamPointsPerGame', 'AwayTeamPointsAllowedPerGame',
        'HomeTeamInjuries', 'AwayTeamInjuries',
    ]
    
    # Make sure all features exist in the dataset
    for feature in features:
        if feature not in train_data.columns:
            logger.error(f"Missing required feature: {feature}")
            return False
    
    # Prepare training data
    X_train = train_data[features]
    y_train = train_data['OverUnderResult']
    
    X_test = test_data[features]
    y_test = test_data['OverUnderResult']
    
    # Handle missing values
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_test.mean())
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    scaler_path = '../data/models/scaler.joblib'
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved feature scaler to {scaler_path}")
    
    # Train multiple models to find the best performer
    logger.info("Training and evaluating multiple models")
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    }
    
    results = {}
    
    for name, model in tqdm(models.items(), desc="Training models"):
        logger.info(f"Training {name} model")
        
        # Train the model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        logger.info(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm.tolist()
        }
    
    # Find the best model based on accuracy
    best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    best_model = results[best_model_name]['model']
    best_metrics = results[best_model_name]
    
    logger.info(f"Best model: {best_model_name} with accuracy {best_metrics['accuracy']:.4f}")
    
    # Save the best model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f'../data/models/nba_overunder_model_{best_model_name}_{timestamp}.joblib'
    joblib.dump(best_model, model_path)
    logger.info(f"Saved best model to {model_path}")
    
    # Save metrics for the best model
    metrics_data = {
        'model_name': best_model_name,
        'timestamp': timestamp,
        'accuracy': float(best_metrics['accuracy']),
        'precision': float(best_metrics['precision']),
        'recall': float(best_metrics['recall']),
        'f1': float(best_metrics['f1']),
        'confusion_matrix': best_metrics['confusion_matrix'],
        'features': features
    }
    
    metrics_path = f'../data/models/model_metrics_{timestamp}.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)
    
    logger.info(f"Saved model metrics to {metrics_path}")
    
    # Create a symlink to the latest model
    latest_model_path = '../data/models/nba_overunder_model_latest.joblib'
    
    try:
        # Remove existing symlink if it exists
        if os.path.exists(latest_model_path):
            os.remove(latest_model_path)
        
        # This creates a hard copy for Windows (as symbolic links might not work properly)
        import shutil
        shutil.copy2(model_path, latest_model_path)
        logger.info(f"Created copy of latest model at {latest_model_path}")
    except Exception as e:
        logger.error(f"Error creating symlink to latest model: {e}")
    
    logger.info("Model training completed successfully")
    return True

if __name__ == "__main__":
    train_model()
