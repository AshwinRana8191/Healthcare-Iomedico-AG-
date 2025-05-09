#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data acquisition script for fetching oncology clinical trial data from ClinicalTrials.gov

This script uses the ClinicalTrials.gov API to fetch interventional oncology trials,
filtering for relevant phases and statuses. The data is saved as raw JSON and CSV files.
"""

import os
import json
import time
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

# Define project directories
PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = PROJECT_DIR / 'data' / 'processed'

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ClinicalTrials.gov API endpoints
BASE_URL = "https://clinicaltrials.gov/api/query/study_fields"

# Define search parameters for oncology trials
def get_search_params(min_rnk=1, max_rnk=1000):
    """
    Generate search parameters for the ClinicalTrials.gov API
    
    Args:
        min_rnk: Minimum rank for pagination
        max_rnk: Maximum rank for pagination
        
    Returns:
        dict: Parameters for the API request
    """
    return {
        "expr": "AREA[ConditionSearch] CANCER AND AREA[StudyType] INTERVENTIONAL AND AREA[Phase] PHASE2 OR PHASE3",
        "fields": ",".join([
            "NCTId", "BriefTitle", "OfficialTitle", "OverallStatus", "StartDate", "PrimaryCompletionDate", 
            "CompletionDate", "StudyType", "Phase", "EnrollmentCount", "ArmGroupDescription", 
            "InterventionType", "InterventionName", "EligibilityCriteria", "Gender", "MinimumAge", 
            "MaximumAge", "DesignAllocation", "DesignInterventionModel", "DesignPrimaryPurpose", 
            "DesignMasking", "LocationCountry", "LeadSponsorName", "LeadSponsorClass", "CollaboratorName", 
            "WhyStopped", "ConditionName"
        ]),
        "min_rnk": min_rnk,
        "max_rnk": max_rnk,
        "fmt": "json"
    }

def fetch_trials(params):
    """
    Fetch clinical trials data from ClinicalTrials.gov API
    
    Args:
        params: API request parameters
        
    Returns:
        dict: API response data
    """
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def save_raw_data(data, filename):
    """
    Save raw JSON data to file
    
    Args:
        data: JSON data to save
        filename: Output filename
    """
    filepath = RAW_DATA_DIR / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"Saved raw data to {filepath}")

def process_trials_to_dataframe(data):
    """
    Process raw API data into a pandas DataFrame
    
    Args:
        data: Raw API response data
        
    Returns:
        pandas.DataFrame: Processed clinical trials data
    """
    if not data or 'StudyFieldsResponse' not in data:
        return None
        
    studies = data['StudyFieldsResponse']['StudyFields']
    return pd.DataFrame(studies)

def fetch_all_trials():
    """
    Fetch all oncology trials using pagination
    
    Returns:
        pandas.DataFrame: Combined dataframe of all fetched trials
    """
    all_trials = []
    batch_size = 1000
    min_rank = 1
    max_rank = batch_size
    total_count = float('inf')
    batch_num = 1
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    while min_rank <= total_count:
        print(f"Fetching batch {batch_num}: records {min_rank} to {max_rank}")
        params = get_search_params(min_rank, max_rank)
        data = fetch_trials(params)
        
        if data and 'StudyFieldsResponse' in data:
            total_count = int(data['StudyFieldsResponse']['NStudiesFound'])
            print(f"Total studies found: {total_count}")
            
            # Save raw batch data
            batch_filename = f"oncology_trials_batch_{batch_num}_{timestamp}.json"
            save_raw_data(data, batch_filename)
            
            # Process batch to dataframe
            batch_df = process_trials_to_dataframe(data)
            if batch_df is not None and not batch_df.empty:
                all_trials.append(batch_df)
                print(f"Added {len(batch_df)} records from batch {batch_num}")
            
            # Update for next batch
            min_rank += batch_size
            max_rank += batch_size
            batch_num += 1
            
            # Respect API rate limits
            time.sleep(1)
        else:
            print("Failed to fetch data or no more data available")
            break
    
    # Combine all batches
    if all_trials:
        combined_df = pd.concat(all_trials, ignore_index=True)
        return combined_df
    else:
        return None

def main():
    """
    Main function to execute the data acquisition process
    """
    print("Starting data acquisition from ClinicalTrials.gov...")
    
    # Fetch all trials
    trials_df = fetch_all_trials()
    
    if trials_df is not None and not trials_df.empty:
        # Save combined data to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"oncology_trials_combined_{timestamp}.csv"
        csv_filepath = RAW_DATA_DIR / csv_filename
        trials_df.to_csv(csv_filepath, index=False)
        print(f"Saved combined data to {csv_filepath}")
        print(f"Total records: {len(trials_df)}")
        
        # Display sample of the data
        print("\nSample of acquired data:")
        print(trials_df.head())
        
        # Display data summary
        print("\nData summary:")
        print(f"Columns: {trials_df.columns.tolist()}")
        print(f"Trial status counts:\n{trials_df['OverallStatus'].value_counts()}")
        
        return trials_df
    else:
        print("No data acquired")
        return None

if __name__ == "__main__":
    main()