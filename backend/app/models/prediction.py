import pandas as pd
import numpy as np
import joblib
import os
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NBAPredictionModel:
    """
    ML model to predict NBA game outcomes, focusing on over/under bets
    """
    
    def __init__(self, model_dir="./data/models"):
        """Initialize the prediction model"""
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        
        # The model will be loaded or trained later
        self.model = None
        self.scaler = None
        
        # Features to use for prediction
        self.features = [
            'HomeTeamWins', 'HomeTeamLosses', 'HomeTeamPointsPerGame', 
            'HomeTeamPointsAllowedPerGame', 'AwayTeamWins', 'AwayTeamLosses', 
            'AwayTeamPointsPerGame', 'AwayTeamPointsAllowedPerGame',
            'HomeTeamInjuries', 'AwayTeamInjuries'
        ]
        
        logger.info("NBA Prediction Model initialized")
    
    def load_historical_data(self, data_path):
        """
        Load historical NBA game data for training
        Should include actual over/under results
        """
        try:
            logger.info(f"Loading historical data from {data_path}")
            data = pd.read_csv(data_path)
            return data
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return None
    
    def preprocess_data(self, data, training=True):
        """
        Preprocess data for model training or prediction
        """
        logger.info("Preprocessing data")
        
        # If we don't have essential columns, return None
        for feature in self.features:
            if feature not in data.columns:
                logger.error(f"Missing required feature: {feature}")
                return None, None
        
        # For training, we need target labels
        if training and 'OverUnderResult' not in data.columns:
            logger.error("Missing target column 'OverUnderResult' for training")
            return None, None
        
        # Handle missing values
        for feature in self.features:
            data[feature].fillna(data[feature].median(), inplace=True)
        
        # Extract features
        X = data[self.features]
        
        # For training, get target labels (0 = Under, 1 = Over)
        y = None
        if training:
            y = data['OverUnderResult']
        
        # Scale features
        if training or self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Save the scaler
            scaler_path = os.path.join(self.model_dir, "scaler.joblib")
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Saved feature scaler to {scaler_path}")
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, y
    
    def train(self, data_path):
        """
        Train the prediction model using historical data
        """
        logger.info("Starting model training")
        
        # Load and preprocess data
        data = self.load_historical_data(data_path)
        if data is None:
            return False
        
        X, y = self.preprocess_data(data, training=True)
        if X is None or y is None:
            return False
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Try different models and select the best one
        logger.info("Training multiple models to find the best performer")
        
        # 1. Random Forest
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train, y_train)
        rf_preds = rf_model.predict(X_val)
        rf_accuracy = accuracy_score(y_val, rf_preds)
        logger.info(f"Random Forest Accuracy: {rf_accuracy:.4f}")
        
        # 2. Gradient Boosting
        gb_model = GradientBoostingClassifier(random_state=42)
        gb_model.fit(X_train, y_train)
        gb_preds = gb_model.predict(X_val)
        gb_accuracy = accuracy_score(y_val, gb_preds)
        logger.info(f"Gradient Boosting Accuracy: {gb_accuracy:.4f}")
        
        # 3. XGBoost
        xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
        xgb_model.fit(X_train, y_train)
        xgb_preds = xgb_model.predict(X_val)
        xgb_accuracy = accuracy_score(y_val, xgb_preds)
        logger.info(f"XGBoost Accuracy: {xgb_accuracy:.4f}")
        
        # Select the best model
        models = {
            "RandomForest": (rf_model, rf_accuracy),
            "GradientBoosting": (gb_model, gb_accuracy),
            "XGBoost": (xgb_model, xgb_accuracy)
        }
        
        best_model_name = max(models.items(), key=lambda x: x[1][1])[0]
        self.model, best_accuracy = models[best_model_name]
        
        logger.info(f"Selected {best_model_name} as the best model with accuracy {best_accuracy:.4f}")
        
        # Now fine-tune the best model with grid search
        logger.info(f"Fine-tuning {best_model_name} with grid search")
        
        if best_model_name == "RandomForest":
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
            grid_model = RandomForestClassifier(random_state=42)
        
        elif best_model_name == "GradientBoosting":
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
            grid_model = GradientBoostingClassifier(random_state=42)
        
        elif best_model_name == "XGBoost":
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
            grid_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
        
        # Run grid search
        grid_search = GridSearchCV(
            estimator=grid_model,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get the best model
        self.model = grid_search.best_estimator_
        tuned_preds = self.model.predict(X_val)
        tuned_accuracy = accuracy_score(y_val, tuned_preds)
        
        logger.info(f"Fine-tuned {best_model_name} Accuracy: {tuned_accuracy:.4f}")
        logger.info(f"Best parameters: {grid_search.best_params_}")
        
        # Save the model
        model_path = os.path.join(self.model_dir, f"over_under_model_{datetime.now().strftime('%Y%m%d')}.joblib")
        joblib.dump(self.model, model_path)
        logger.info(f"Saved model to {model_path}")
        
        # Calculate additional metrics
        precision = precision_score(y_val, tuned_preds)
        recall = recall_score(y_val, tuned_preds)
        f1 = f1_score(y_val, tuned_preds)
        
        logger.info(f"Model Evaluation Metrics:")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        
        # Save the model metrics for reference
        metrics = {
            'accuracy': tuned_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'model_type': best_model_name,
            'best_params': grid_search.best_params_,
            'training_date': datetime.now().strftime('%Y-%m-%d'),
        }
        
        metrics_path = os.path.join(self.model_dir, f"model_metrics_{datetime.now().strftime('%Y%m%d')}.json")
        import json
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        return True
    
    def load_model(self, model_path=None):
        """
        Load a trained model from disk or create a mock model for testing
        """
        if model_path is None:
            # Find the latest model
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir, exist_ok=True)
            
            model_files = [f for f in os.listdir(self.model_dir) if f.startswith("over_under_model_") and f.endswith(".joblib")]
            
            if not model_files:
                logger.info("No trained model found, creating a mock model for testing")
                # Create a simple RandomForestClassifier as a mock model
                self.model = RandomForestClassifier(n_estimators=10, random_state=42)
                # Create mock data to fit the model
                X = np.random.rand(100, len(self.features))
                y = np.random.randint(0, 2, 100)
                self.model.fit(X, y)
                
                # Save the mock model
                model_path = os.path.join(self.model_dir, f"over_under_model_{datetime.now().strftime('%Y%m%d')}.joblib")
                joblib.dump(self.model, model_path)
                
                # Create a mock scaler
                self.scaler = StandardScaler()
                self.scaler.fit(X)
                scaler_path = os.path.join(self.model_dir, "scaler.joblib")
                joblib.dump(self.scaler, scaler_path)
                
                logger.info(f"Created and saved mock model to {model_path}")
                return True
            
            # Sort by date (newest first)
            model_files.sort(reverse=True)
            model_path = os.path.join(self.model_dir, model_files[0])
        
        try:
            logger.info(f"Loading model from {model_path}")
            self.model = joblib.load(model_path)
            
            # Also load the scaler
            scaler_path = os.path.join(self.model_dir, "scaler.joblib")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            else:
                logger.warning("No scaler found, will create one during prediction")
            
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict(self, game_data):
        """
        Make predictions for upcoming games
        
        Args:
            game_data: DataFrame containing upcoming game information
            
        Returns:
            DataFrame with original data plus predictions and confidence scores
        """
        if self.model is None:
            logger.error("Model not loaded or trained")
            if not self.load_model():
                return None
        
        logger.info("Making predictions for upcoming games")
        
        # Preprocess the data
        X, _ = self.preprocess_data(game_data, training=False)
        if X is None:
            return None
        
        # Make predictions (1 = Over, 0 = Under)
        predictions = self.model.predict(X)
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(X)
        
        # Add predictions to the original data
        result = game_data.copy()
        result['PredictedOverUnder'] = predictions
        
        # Add confidence scores (probability of the predicted class)
        confidence = []
        for i, pred in enumerate(predictions):
            confidence.append(probabilities[i][pred])
        
        result['Confidence'] = confidence
        
        # Add a column indicating Over or Under in text
        result['PredictionText'] = ['Over' if p == 1 else 'Under' for p in predictions]
        
        # Calculate risk level based on confidence
        def get_risk_level(conf):
            if conf >= 0.75:
                return "Low"
            elif conf >= 0.6:
                return "Medium"
            else:
                return "High"
        
        result['RiskLevel'] = result['Confidence'].apply(get_risk_level)
        
        return result
    
    def evaluate_past_predictions(self, predictions_path, results_path):
        """
        Evaluate the accuracy of past predictions
        
        Args:
            predictions_path: Path to saved predictions
            results_path: Path to actual game results
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            logger.info("Evaluating past predictions")
            
            # Load predictions and actual results
            predictions_df = pd.read_csv(predictions_path)
            results_df = pd.read_csv(results_path)
            
            # Merge on GameID
            merged = pd.merge(predictions_df, results_df, on='GameID')
            
            # Calculate accuracy
            correct = (merged['PredictedOverUnder'] == merged['ActualOverUnder']).sum()
            total = len(merged)
            accuracy = correct / total if total > 0 else 0
            
            # Calculate accuracy by risk level
            risk_levels = ['Low', 'Medium', 'High']
            risk_accuracy = {}
            
            for risk in risk_levels:
                risk_df = merged[merged['RiskLevel'] == risk]
                risk_correct = (risk_df['PredictedOverUnder'] == risk_df['ActualOverUnder']).sum()
                risk_total = len(risk_df)
                risk_accuracy[risk] = risk_correct / risk_total if risk_total > 0 else 0
            
            metrics = {
                'overall_accuracy': accuracy,
                'total_games': total,
                'correct_predictions': correct,
                'accuracy_by_risk': risk_accuracy,
                'evaluation_date': datetime.now().strftime('%Y-%m-%d')
            }
            
            logger.info(f"Prediction Accuracy: {accuracy:.4f}")
            for risk, acc in risk_accuracy.items():
                logger.info(f"{risk} Risk Accuracy: {acc:.4f}")
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error evaluating past predictions: {e}")
            return None
    
    def optimize_parlay(self, predictions, num_legs=3):
        """
        Optimize a parlay bet by selecting the best games
        
        Args:
            predictions: DataFrame with game predictions and confidence scores
            num_legs: Number of games to include in the parlay
            
        Returns:
            List of recommended games for the parlay
        """
        logger.info(f"Optimizing {num_legs}-leg parlay")
        
        if len(predictions) < num_legs:
            logger.warning(f"Not enough games to create a {num_legs}-leg parlay")
            return predictions
        
        # Sort by confidence (highest first)
        sorted_predictions = predictions.sort_values('Confidence', ascending=False)
        
        # Take the top N games
        optimal_parlay = sorted_predictions.head(num_legs)
        
        # Calculate overall parlay probability (product of individual probabilities)
        parlay_probability = optimal_parlay['Confidence'].prod()
        
        logger.info(f"Optimal {num_legs}-leg parlay has {parlay_probability:.4f} combined probability")
        
        return optimal_parlay