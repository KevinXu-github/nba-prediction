import os
import logging
import subprocess
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../data/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_pipeline():
    """Run the full data collection and model training pipeline"""
    start_time = time.time()
    logger.info("Starting NBA Parlay Prediction pipeline")
    
    # Create necessary directories
    os.makedirs('../data/historical', exist_ok=True)
    os.makedirs('../data/training', exist_ok=True)
    os.makedirs('../data/models', exist_ok=True)
    
    # Step 1: Fetch historical data
    logger.info("Step 1: Fetching historical NBA data")
    try:
        subprocess.run(['python', 'fetch_historical_data.py'], check=True)
        logger.info("Historical data collection completed")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during data collection: {e}")
        return False
    
    # Step 2: Train the model
    logger.info("Step 2: Training prediction model")
    try:
        subprocess.run(['python', 'train_model.py'], check=True)
        logger.info("Model training completed")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during model training: {e}")
        return False
    
    # Step 3: Test predictions
    logger.info("Step 3: Testing model predictions")
    try:
        subprocess.run(['python', 'test_predictions.py'], check=True)
        logger.info("Prediction testing completed")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during prediction testing: {e}")
        return False
    
    # Calculate total runtime
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"Pipeline completed successfully in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    return True

if __name__ == "__main__":
    run_pipeline()
