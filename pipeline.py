#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
End-to-end pipeline for oncology clinical trial prediction project.
This script orchestrates the entire workflow from data acquisition to model evaluation.
"""

import os
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)

logger = logging.getLogger(__name__)


def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        'data/raw',
        'data/processed',
        'reports/figures',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")


def run_data_acquisition():
    """Run the data acquisition process."""
    logger.info("Starting data acquisition...")
    try:
        from src.data.acquisition import fetch_clinical_trials_data
        fetch_clinical_trials_data()
        logger.info("Data acquisition completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error in data acquisition: {str(e)}")
        return False


def run_data_preprocessing():
    """Run the data preprocessing process."""
    logger.info("Starting data preprocessing...")
    try:
        from src.data.preprocessing import preprocess_clinical_trials_data
        preprocess_clinical_trials_data()
        logger.info("Data preprocessing completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        return False


def run_feature_engineering():
    """Run the feature engineering process."""
    logger.info("Starting feature engineering...")
    try:
        from src.features.build_features import build_structured_features
        from src.features.text_features import build_text_features
        
        build_structured_features()
        build_text_features()
        
        logger.info("Feature engineering completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        return False


def run_model_training():
    """Run the model training process."""
    logger.info("Starting model training...")
    try:
        from src.models.train_model import train_classification_model, train_regression_model
        
        # Train classification model for trial success prediction
        train_classification_model()
        
        # Train regression model for trial duration prediction
        train_regression_model()
        
        logger.info("Model training completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        return False


def run_model_evaluation():
    """Run the model evaluation process."""
    logger.info("Starting model evaluation...")
    try:
        from src.models.evaluate_model import evaluate_models, generate_insights
        
        evaluate_models()
        generate_insights()
        
        logger.info("Model evaluation completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}")
        return False


def run_visualization():
    """Generate visualizations for the results."""
    logger.info("Generating visualizations...")
    try:
        from src.visualization.visualize import create_visualizations
        
        create_visualizations()
        
        logger.info("Visualization generation completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error in visualization generation: {str(e)}")
        return False


def main():
    """Main function to run the entire pipeline."""
    start_time = time.time()
    logger.info("Starting the oncology clinical trial prediction pipeline")
    
    # Create necessary directories
    create_directories()
    
    # Run pipeline steps
    steps = [
        ("Data Acquisition", run_data_acquisition),
        ("Data Preprocessing", run_data_preprocessing),
        ("Feature Engineering", run_feature_engineering),
        ("Model Training", run_model_training),
        ("Model Evaluation", run_model_evaluation),
        ("Visualization", run_visualization)
    ]
    
    results = {}
    for step_name, step_func in steps:
        logger.info(f"\n{'=' * 50}\nExecuting step: {step_name}\n{'=' * 50}")
        step_start_time = time.time()
        success = step_func()
        step_duration = time.time() - step_start_time
        
        results[step_name] = {
            "success": success,
            "duration": step_duration
        }
        
        if not success:
            logger.error(f"Pipeline failed at step: {step_name}")
            break
    
    # Calculate total duration
    total_duration = time.time() - start_time
    
    # Log summary
    logger.info("\n" + "=" * 50)
    logger.info("Pipeline Execution Summary:")
    logger.info("=" * 50)
    
    all_success = True
    for step_name, result in results.items():
        status = "SUCCESS" if result["success"] else "FAILED"
        logger.info(f"{step_name}: {status} (Duration: {result['duration']:.2f} seconds)")
        if not result["success"]:
            all_success = False
    
    logger.info("=" * 50)
    logger.info(f"Total Duration: {total_duration:.2f} seconds")
    logger.info(f"Overall Status: {'SUCCESS' if all_success else 'FAILED'}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()