#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model training for clinical trial outcome prediction

This script trains and evaluates models to predict clinical trial outcomes:
1. Classification models for trial completion status (completed vs. terminated)
2. Regression models for trial duration
"""

import os
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime

# Modeling and evaluation
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import shap

# Import feature engineering pipeline
sys_path = str(Path(__file__).resolve().parents[1])
if sys_path not in sys.path:
    sys.path.append(sys_path)
from features.build_features import prepare_modeling_data, create_preprocessing_pipeline

# Define project directories
PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_DIR / 'data' / 'processed'
MODEL_DIR = PROJECT_DIR / 'models'

# Ensure model directory exists
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def load_modeling_data(filename=None):
    """
    Load modeling-ready data
    
    Args:
        filename: Specific filename to load, if None, loads the most recent file
        
    Returns:
        pandas.DataFrame: Modeling-ready data
    """
    if filename is None:
        # Find the most recent modeling-ready CSV file
        csv_files = list(PROCESSED_DATA_DIR.glob("oncology_trials_modeling_ready_*.csv"))
        if not csv_files:
            raise FileNotFoundError("No modeling-ready data files found")
        
        # Sort by modification time (most recent first)
        csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        filename = csv_files[0]
    else:
        filename = PROCESSED_DATA_DIR / filename
    
    print(f"Loading modeling data from {filename}")
    return pd.read_csv(filename)

def train_classification_models(X, y):
    """
    Train and evaluate classification models for trial completion status
    
    Args:
        X: Feature matrix
        y: Target vector (binary: completed vs. terminated)
        
    Returns:
        dict: Dictionary of trained models and their performance metrics
    """
    print("\nTraining classification models for trial completion status...")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(X_train)
    
    # Preprocess the data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Initialize models
    models = {
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'xgboost': xgb.XGBClassifier(n_estimators=100, random_state=42),
        'lightgbm': lgb.LGBMClassifier(n_estimators=100, random_state=42)
    }
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Train the model
        model.fit(X_train_processed, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_processed)
        y_pred_proba = model.predict_proba(X_test_processed)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        # Store results
        results[name] = {
            'model': model,
            'preprocessor': preprocessor,
            'metrics': metrics
        }
        
        print(f"{name} metrics: {metrics}")
    
    # Identify best model based on F1 score
    best_model_name = max(results.keys(), key=lambda k: results[k]['metrics']['f1'])
    print(f"\nBest classification model: {best_model_name}")
    print(f"F1 Score: {results[best_model_name]['metrics']['f1']:.4f}")
    
    # Feature importance for the best model (if applicable)
    if hasattr(results[best_model_name]['model'], 'feature_importances_'):
        print("\nTop 10 important features:")
        feature_names = X.columns.tolist()
        importances = results[best_model_name]['model'].feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        for i in indices:
            if i < len(feature_names):
                print(f"{feature_names[i]}: {importances[i]:.4f}")
    
    return results

def train_regression_models(X, y):
    """
    Train and evaluate regression models for trial duration prediction
    
    Args:
        X: Feature matrix
        y: Target vector (continuous: trial duration in days)
        
    Returns:
        dict: Dictionary of trained models and their performance metrics
    """
    print("\nTraining regression models for trial duration prediction...")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(X_train)
    
    # Preprocess the data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Initialize models
    models = {
        'ridge': Ridge(alpha=1.0, random_state=42),
        'lasso': Lasso(alpha=0.1, random_state=42),
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42),
        'lightgbm': lgb.LGBMRegressor(n_estimators=100, random_state=42)
    }
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Train the model
        model.fit(X_train_processed, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_processed)
        
        # Calculate metrics
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        
        # Store results
        results[name] = {
            'model': model,
            'preprocessor': preprocessor,
            'metrics': metrics
        }
        
        print(f"{name} metrics: {metrics}")
    
    # Identify best model based on R² score
    best_model_name = max(results.keys(), key=lambda k: results[k]['metrics']['r2'])
    print(f"\nBest regression model: {best_model_name}")
    print(f"R² Score: {results[best_model_name]['metrics']['r2']:.4f}")
    print(f"RMSE: {results[best_model_name]['metrics']['rmse']:.2f} days")
    
    # Feature importance for the best model (if applicable)
    if hasattr(results[best_model_name]['model'], 'feature_importances_'):
        print("\nTop 10 important features:")
        feature_names = X.columns.tolist()
        importances = results[best_model_name]['model'].feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        for i in indices:
            if i < len(feature_names):
                print(f"{feature_names[i]}: {importances[i]:.4f}")
    
    return results

def save_model(model_dict, model_name, target_type):
    """
    Save trained model and preprocessor to disk
    
    Args:
        model_dict: Dictionary containing model and preprocessor
        model_name: Name of the model
        target_type: Type of prediction (classification or regression)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{target_type}_{model_name}_{timestamp}.pkl"
    filepath = MODEL_DIR / filename
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_dict, f)
    
    print(f"Saved {target_type} model {model_name} to {filepath}")

def main():
    """
    Main function to execute the model training pipeline
    """
    try:
        # Load modeling data
        df = load_modeling_data()
        
        # Prepare data for classification (completion status prediction)
        print("\nPreparing data for completion status prediction...")
        X_status, y_status = prepare_modeling_data(df, target_column='completion_status')
        
        # Train classification models
        classification_results = train_classification_models(X_status, y_status)
        
        # Save best classification model
        best_class_model_name = max(classification_results.keys(), 
                                   key=lambda k: classification_results[k]['metrics']['f1'])
        save_model(classification_results[best_class_model_name], 
                  best_class_model_name, 'classification')
        
        # Prepare data for regression (duration prediction)
        print("\nPreparing data for trial duration prediction...")
        X_duration, y_duration = prepare_modeling_data(df, target_column='trial_duration_days')
        
        # Train regression models
        regression_results = train_regression_models(X_duration, y_duration)
        
        # Save best regression model
        best_reg_model_name = max(regression_results.keys(), 
                                key=lambda k: regression_results[k]['metrics']['r2'])
        save_model(regression_results[best_reg_model_name], 
                 best_reg_model_name, 'regression')
        
        print("\nModel training complete!")
        
        return {
            'classification': classification_results,
            'regression': regression_results
        }
    
    except Exception as e:
        print(f"Error in model training pipeline: {e}")
        return None

if __name__ == "__main__":
    import sys
    from datetime import datetime  # Import here to avoid circular import
    main()
