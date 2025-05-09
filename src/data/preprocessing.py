#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data preprocessing script for cleaning and preparing clinical trial data

This script processes raw data from ClinicalTrials.gov, handling missing values,
standardizing formats, and preparing the data for feature engineering and analysis.
"""

import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Define project directories
PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = PROJECT_DIR / 'data' / 'processed'

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

def load_raw_data(filename=None):
    """
    Load raw clinical trials data from CSV file
    
    Args:
        filename: Specific filename to load, if None, loads the most recent file
        
    Returns:
        pandas.DataFrame: Raw clinical trials data
    """
    if filename is None:
        # Find the most recent combined CSV file
        csv_files = list(RAW_DATA_DIR.glob("oncology_trials_combined_*.csv"))
        if not csv_files:
            raise FileNotFoundError("No raw data files found")
        
        # Sort by modification time (most recent first)
        csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        filename = csv_files[0]
    else:
        filename = RAW_DATA_DIR / filename
    
    print(f"Loading raw data from {filename}")
    return pd.read_csv(filename)

def clean_date_columns(df, date_columns):
    """
    Clean and standardize date columns
    
    Args:
        df: DataFrame containing the data
        date_columns: List of column names containing dates
        
    Returns:
        pandas.DataFrame: DataFrame with cleaned date columns
    """
    for col in date_columns:
        if col in df.columns:
            # Convert to datetime, handling various formats
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Create a new column with standardized date format
            df[f"{col}_std"] = df[col].dt.strftime('%Y-%m-%d')
    
    return df

def create_duration_features(df):
    """
    Create trial duration features from date columns
    
    Args:
        df: DataFrame containing the data
        
    Returns:
        pandas.DataFrame: DataFrame with added duration features
    """
    # Calculate trial duration (in days)
    if 'StartDate' in df.columns and 'CompletionDate' in df.columns:
        df['trial_duration_days'] = (df['CompletionDate'] - df['StartDate']).dt.days
    
    # Calculate time to primary completion (in days)
    if 'StartDate' in df.columns and 'PrimaryCompletionDate' in df.columns:
        df['primary_completion_days'] = (df['PrimaryCompletionDate'] - df['StartDate']).dt.days
    
    return df

def create_target_variables(df):
    """
    Create target variables for modeling
    
    Args:
        df: DataFrame containing the data
        
    Returns:
        pandas.DataFrame: DataFrame with added target variables
    """
    # Create binary target for trial completion status
    # 1 = Completed, 0 = Terminated/Withdrawn/Suspended
    status_mapping = {
        'Completed': 1,
        'Terminated': 0,
        'Withdrawn': 0,
        'Suspended': 0
    }
    
    # Create a new column with binary status
    df['completion_status'] = df['OverallStatus'].map(status_mapping)
    
    # Filter to only include trials with definitive completion status
    df_with_status = df[df['completion_status'].notna()].copy()
    
    return df_with_status

def clean_enrollment_data(df):
    """
    Clean enrollment count data
    
    Args:
        df: DataFrame containing the data
        
    Returns:
        pandas.DataFrame: DataFrame with cleaned enrollment data
    """
    if 'EnrollmentCount' in df.columns:
        # Convert to numeric, coercing errors to NaN
        df['EnrollmentCount'] = pd.to_numeric(df['EnrollmentCount'], errors='coerce')
    
    return df

def extract_phase_features(df):
    """
    Extract and standardize phase information
    
    Args:
        df: DataFrame containing the data
        
    Returns:
        pandas.DataFrame: DataFrame with standardized phase features
    """
    if 'Phase' in df.columns:
        # Create binary indicators for each phase
        df['is_phase_1'] = df['Phase'].str.contains('Phase 1', case=False, na=False).astype(int)
        df['is_phase_2'] = df['Phase'].str.contains('Phase 2', case=False, na=False).astype(int)
        df['is_phase_3'] = df['Phase'].str.contains('Phase 3', case=False, na=False).astype(int)
        df['is_phase_4'] = df['Phase'].str.contains('Phase 4', case=False, na=False).astype(int)
    
    return df

def extract_sponsor_features(df):
    """
    Extract and standardize sponsor information
    
    Args:
        df: DataFrame containing the data
        
    Returns:
        pandas.DataFrame: DataFrame with standardized sponsor features
    """
    # Create binary indicator for industry sponsorship
    if 'LeadSponsorClass' in df.columns:
        df['is_industry_sponsored'] = (df['LeadSponsorClass'] == 'INDUSTRY').astype(int)
    
    return df

def extract_location_features(df):
    """
    Extract features related to trial locations
    
    Args:
        df: DataFrame containing the data
        
    Returns:
        pandas.DataFrame: DataFrame with location-based features
    """
    if 'LocationCountry' in df.columns:
        # Count number of countries per trial
        df['country_count'] = df['LocationCountry'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        
        # Create binary indicator for multi-country trials
        df['is_multi_country'] = (df['country_count'] > 1).astype(int)
        
        # Check if trial includes US sites
        df['has_us_sites'] = df['LocationCountry'].apply(
            lambda x: 1 if isinstance(x, list) and 'United States' in x else 0
        )
    
    return df

def preprocess_data(df):
    """
    Main preprocessing function to clean and prepare the data
    
    Args:
        df: Raw DataFrame to process
        
    Returns:
        pandas.DataFrame: Processed DataFrame ready for analysis
    """
    print("Starting data preprocessing...")
    
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Clean date columns
    date_columns = ['StartDate', 'PrimaryCompletionDate', 'CompletionDate']
    processed_df = clean_date_columns(processed_df, date_columns)
    
    # Clean enrollment data
    processed_df = clean_enrollment_data(processed_df)
    
    # Extract features from structured data
    processed_df = extract_phase_features(processed_df)
    processed_df = extract_sponsor_features(processed_df)
    processed_df = extract_location_features(processed_df)
    
    # Create duration features
    processed_df = create_duration_features(processed_df)
    
    # Create target variables
    processed_df = create_target_variables(processed_df)
    
    # Filter to only include trials with definitive status and duration
    final_df = processed_df[
        processed_df['completion_status'].notna() & 
        processed_df['trial_duration_days'].notna()
    ].copy()
    
    print(f"Preprocessing complete. Rows remaining: {len(final_df)}")
    return final_df

def save_processed_data(df, filename=None):
    """
    Save processed data to CSV file
    
    Args:
        df: Processed DataFrame to save
        filename: Output filename, if None, generates a timestamped filename
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"processed_oncology_trials_{timestamp}.csv"
    
    filepath = PROCESSED_DATA_DIR / filename
    df.to_csv(filepath, index=False)
    print(f"Saved processed data to {filepath}")

def main():
    """
    Main function to execute the data preprocessing pipeline
    """
    try:
        # Load raw data
        raw_df = load_raw_data()
        
        # Preprocess the data
        processed_df = preprocess_data(raw_df)
        
        # Save processed data
        save_processed_data(processed_df)
        
        # Display data summary
        print("\nProcessed data summary:")
        print(f"Rows: {len(processed_df)}")
        print(f"Columns: {processed_df.columns.tolist()}")
        print(f"\nCompletion status counts:\n{processed_df['completion_status'].value_counts()}")
        print(f"\nTrial duration statistics:\n{processed_df['trial_duration_days'].describe()}")
        
        return processed_df
    
    except Exception as e:
        print(f"Error in preprocessing pipeline: {e}")
        return None

if __name__ == "__main__":
    main()