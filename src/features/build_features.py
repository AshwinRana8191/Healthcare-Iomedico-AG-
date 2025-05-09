#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature engineering for clinical trial data

This script combines structured features with text features and prepares the data for modeling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Import text feature extraction
from text_features import process_all_text_fields

# Define project directories
PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_DIR / 'data' / 'processed'

def load_processed_data(filename=None):
    """
    Load processed clinical trials data
    
    Args:
        filename: Specific filename to load, if None, loads the most recent file
        
    Returns:
        pandas.DataFrame: Processed clinical trials data
    """
    if filename is None:
        # Find the most recent processed CSV file
        csv_files = list(PROCESSED_DATA_DIR.glob("processed_oncology_trials_*.csv"))
        if not csv_files:
            raise FileNotFoundError("No processed data files found")
        
        # Sort by modification time (most recent first)
        csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        filename = csv_files[0]
    else:
        filename = PROCESSED_DATA_DIR / filename
    
    print(f"Loading processed data from {filename}")
    return pd.read_csv(filename)

def create_structured_features(df):
    """
    Create features from structured data fields
    
    Args:
        df: DataFrame containing the processed data
        
    Returns:
        pandas.DataFrame: DataFrame with additional structured features
    """
    # Make a copy to avoid modifying the original
    df_features = df.copy()
    
    # Create enrollment size categories
    if 'EnrollmentCount' in df_features.columns:
        df_features['enrollment_size'] = pd.cut(
            df_features['EnrollmentCount'],
            bins=[0, 50, 100, 200, 500, float('inf')],
            labels=['very_small', 'small', 'medium', 'large', 'very_large']
        )
    
    # Create trial duration categories
    if 'trial_duration_days' in df_features.columns:
        df_features['duration_category'] = pd.cut(
            df_features['trial_duration_days'],
            bins=[0, 365, 730, 1095, float('inf')],
            labels=['short', 'medium', 'long', 'very_long']
        )
    
    # Create features from study design
    if 'DesignAllocation' in df_features.columns:
        df_features['is_randomized'] = df_features['DesignAllocation'].apply(
            lambda x: 1 if isinstance(x, str) and 'Randomized' in x else 0
        )
    
    if 'DesignInterventionModel' in df_features.columns:
        df_features['is_parallel_design'] = df_features['DesignInterventionModel'].apply(
            lambda x: 1 if isinstance(x, str) and 'Parallel' in x else 0
        )
        df_features['is_crossover_design'] = df_features['DesignInterventionModel'].apply(
            lambda x: 1 if isinstance(x, str) and 'Crossover' in x else 0
        )
    
    if 'DesignMasking' in df_features.columns:
        df_features['is_double_blind'] = df_features['DesignMasking'].apply(
            lambda x: 1 if isinstance(x, str) and 'Double' in x else 0
        )
        df_features['is_single_blind'] = df_features['DesignMasking'].apply(
            lambda x: 1 if isinstance(x, str) and 'Single' in x else 0
        )
        df_features['is_open_label'] = df_features['DesignMasking'].apply(
            lambda x: 1 if isinstance(x, str) and 'None' in x else 0
        )
    
    # Create features from intervention type
    if 'InterventionType' in df_features.columns:
        intervention_types = ['Drug', 'Device', 'Biological', 'Procedure', 'Radiation', 'Behavioral']
        for intervention in intervention_types:
            df_features[f'intervention_{intervention.lower()}'] = df_features['InterventionType'].apply(
                lambda x: 1 if isinstance(x, list) and intervention in x else 0
            )
    
    return df_features

def prepare_modeling_data(df, target_column='completion_status'):
    """
    Prepare data for modeling by selecting relevant features and target
    
    Args:
        df: DataFrame containing all features
        target_column: Name of the target column for modeling
        
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target vector
    """
    # Ensure target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    # Define columns to exclude from features
    exclude_columns = [
        'NCTId', 'BriefTitle', 'OfficialTitle', 'OverallStatus', 
        'StartDate', 'PrimaryCompletionDate', 'CompletionDate',
        'EligibilityCriteria', 'InterventionName', 'InterventionType',
        'DesignAllocation', 'DesignInterventionModel', 'DesignMasking',
        'StartDate_std', 'PrimaryCompletionDate_std', 'CompletionDate_std',
        'WhyStopped', 'completion_status', 'trial_duration_days'
    ]
    
    # Remove target from exclude list if it's not 'completion_status'
    if target_column != 'completion_status':
        exclude_columns.append('completion_status')
    else:
        exclude_columns.append('trial_duration_days')  # Exclude duration if predicting status
    
    # Select feature columns (all columns except excluded ones)
    feature_columns = [col for col in df.columns if col not in exclude_columns]
    
    # Create feature matrix and target vector
    X = df[feature_columns]
    y = df[target_column]
    
    print(f"Prepared modeling data with {X.shape[1]} features and {X.shape[0]} samples")
    return X, y

def create_preprocessing_pipeline(X):
    """
    Create a preprocessing pipeline for numerical and categorical features
    
    Args:
        X: Feature matrix
        
    Returns:
        sklearn.pipeline.Pipeline: Preprocessing pipeline
    """
    # Identify numerical and categorical columns
    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create preprocessing pipelines
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine pipelines in a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_columns),
            ('cat', categorical_pipeline, categorical_columns)
        ],
        remainder='passthrough'
    )
    
    return preprocessor

def main():
    """
    Main function to execute the feature engineering pipeline
    """
    try:
        # Load processed data
        df = load_processed_data()
        
        # Create structured features
        print("Creating structured features...")
        df_with_structured = create_structured_features(df)
        
        # Add text features
        print("Adding text features...")
        df_with_all_features = process_all_text_fields(df_with_structured)
        
        # Save the enhanced dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = PROCESSED_DATA_DIR / f"oncology_trials_modeling_ready_{timestamp}.csv"
        df_with_all_features.to_csv(output_file, index=False)
        
        print(f"Saved modeling-ready data to {output_file}")
        print(f"Total features: {len(df_with_all_features.columns)}")
        
        # Prepare data for modeling (completion status prediction)
        X_status, y_status = prepare_modeling_data(df_with_all_features, target_column='completion_status')
        
        # Prepare data for modeling (duration prediction)
        X_duration, y_duration = prepare_modeling_data(df_with_all_features, target_column='trial_duration_days')
        
        return df_with_all_features
    
    except Exception as e:
        print(f"Error in feature engineering pipeline: {e}")
        return None

if __name__ == "__main__":
    from datetime import datetime  # Import here to avoid circular import
    main()