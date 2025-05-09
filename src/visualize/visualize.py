#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization utilities for clinical trial data analysis

This script provides functions for creating visualizations to explore
relationships in clinical trial data and visualize model results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Define project directories
PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_DIR / 'data' / 'processed'
FIGURE_DIR = PROJECT_DIR / 'reports' / 'figures'

# Ensure figure directory exists
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

def set_plotting_style():
    """
    Set consistent style for all plots
    """
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14

def plot_trial_status_distribution(df):
    """
    Plot distribution of trial statuses
    
    Args:
        df: DataFrame containing trial data with 'OverallStatus' column
    """
    if 'OverallStatus' not in df.columns:
        print("'OverallStatus' column not found in DataFrame")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Count trials by status
    status_counts = df['OverallStatus'].value_counts()
    
    # Plot horizontal bar chart
    ax = status_counts.plot(kind='barh', color=sns.color_palette("viridis", len(status_counts)))
    
    # Add count labels to bars
    for i, count in enumerate(status_counts):
        ax.text(count + 5, i, str(count), va='center')
    
    plt.title('Distribution of Trial Statuses')
    plt.xlabel('Number of Trials')
    plt.ylabel('Status')
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = FIGURE_DIR / f"trial_status_distribution_{timestamp}.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    print(f"Saved trial status distribution plot to {filepath}")
    return plt.gcf()

def plot_trial_phases(df):
    """
    Plot distribution of trial phases
    
    Args:
        df: DataFrame containing trial data with 'Phase' column
    """
    if 'Phase' not in df.columns:
        print("'Phase' column not found in DataFrame")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Count trials by phase
    phase_counts = df['Phase'].value_counts()
    
    # Plot horizontal bar chart
    ax = phase_counts.plot(kind='barh', color=sns.color_palette("mako", len(phase_counts)))
    
    # Add count labels to bars
    for i, count in enumerate(phase_counts):
        ax.text(count + 5, i, str(count), va='center')
    
    plt.title('Distribution of Trial Phases')
    plt.xlabel('Number of Trials')
    plt.ylabel('Phase')
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = FIGURE_DIR / f"trial_phase_distribution_{timestamp}.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    print(f"Saved trial phase distribution plot to {filepath}")
    return plt.gcf()

def plot_enrollment_distribution(df):
    """
    Plot distribution of trial enrollment sizes
    
    Args:
        df: DataFrame containing trial data with 'EnrollmentCount' column
    """
    if 'EnrollmentCount' not in df.columns:
        print("'EnrollmentCount' column not found in DataFrame")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Filter out missing values and convert to numeric
    enrollment_data = pd.to_numeric(df['EnrollmentCount'], errors='coerce')
    enrollment_data = enrollment_data.dropna()
    
    # Plot histogram with log scale for x-axis
    plt.hist(enrollment_data, bins=50, alpha=0.7, color='steelblue')
    plt.xscale('log')
    
    plt.title('Distribution of Trial Enrollment Sizes')
    plt.xlabel('Enrollment Count (log scale)')
    plt.ylabel('Number of Trials')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = FIGURE_DIR / f"enrollment_distribution_{timestamp}.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    print(f"Saved enrollment distribution plot to {filepath}")
    return plt.gcf()

def plot_duration_by_phase(df):
    """
    Plot trial duration by phase
    
    Args:
        df: DataFrame containing trial data with 'Phase' and 'trial_duration_days' columns
    """
    if 'Phase' not in df.columns or 'trial_duration_days' not in df.columns:
        print("Required columns not found in DataFrame")
        return
    
    plt.figure(figsize=(14, 8))
    
    # Filter out missing values
    plot_data = df[['Phase', 'trial_duration_days']].dropna()
    
    # Create box plot
    ax = sns.boxplot(x='Phase', y='trial_duration_days', data=plot_data, palette='viridis')
    
    plt.title('Trial Duration by Phase')
    plt.xlabel('Phase')
    plt.ylabel('Duration (days)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = FIGURE_DIR / f"duration_by_phase_{timestamp}.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    print(f"Saved duration by phase plot to {filepath}")
    return plt.gcf()

def plot_completion_rate_by_sponsor(df):
    """
    Plot trial completion rate by sponsor type
    
    Args:
        df: DataFrame containing trial data with 'LeadSponsorClass' and 'completion_status' columns
    """
    if 'LeadSponsorClass' not in df.columns or 'completion_status' not in df.columns:
        print("Required columns not found in DataFrame")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Filter out missing values
    plot_data = df[['LeadSponsorClass', 'completion_status']].dropna()
    
    # Calculate completion rate by sponsor type
    completion_rate = plot_data.groupby('LeadSponsorClass')['completion_status'].mean()
    
    # Plot bar chart
    ax = completion_rate.plot(kind='bar', color=sns.color_palette("mako", len(completion_rate)))
    
    # Add percentage labels to bars
    for i, rate in enumerate(completion_rate):
        ax.text(i, rate + 0.01, f"{rate:.1%}", ha='center')
    
    plt.title('Trial Completion Rate by Sponsor Type')
    plt.xlabel('Sponsor Type')
    plt.ylabel('Completion Rate')
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = FIGURE_DIR / f"completion_by_sponsor_{timestamp}.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    print(f"Saved completion rate by sponsor plot to {filepath}")
    return plt.gcf()

def plot_enrollment_vs_duration(df):
    """
    Plot relationship between enrollment size and trial duration
    
    Args:
        df: DataFrame containing trial data with 'EnrollmentCount' and 'trial_duration_days' columns
    """
    if 'EnrollmentCount' not in df.columns or 'trial_duration_days' not in df.columns:
        print("Required columns not found in DataFrame")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Filter out missing values and convert to numeric
    plot_data = df[['EnrollmentCount', 'trial_duration_days']].copy()
    plot_data['EnrollmentCount'] = pd.to_numeric(plot_data['EnrollmentCount'], errors='coerce')
    plot_data = plot_data.dropna()
    
    # Create scatter plot with regression line
    sns.regplot(x='EnrollmentCount', y='trial_duration_days', data=plot_data, 
                scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    
    plt.title('Relationship Between Enrollment Size and Trial Duration')
    plt.xlabel('Enrollment Count')
    plt.ylabel('Duration (days)')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = FIGURE_DIR / f"enrollment_vs_duration_{timestamp}.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    print(f"Saved enrollment vs duration plot to {filepath}")
    return plt.gcf()

def plot_completion_by_year(df):
    """
    Plot trial completion rate by start year
    
    Args:
        df: DataFrame containing trial data with 'StartDate' and 'completion_status' columns
    """
    if 'StartDate' not in df.columns or 'completion_status' not in df.columns:
        print("Required columns not found in DataFrame")
        return
    
    plt.figure(figsize=(14, 8))
    
    # Filter out missing values and extract year
    plot_data = df[['StartDate', 'completion_status']].copy()
    plot_data['StartDate'] = pd.to_datetime(plot_data['StartDate'], errors='coerce')
    plot_data = plot_data.dropna()
    plot_data['StartYear'] = plot_data['StartDate'].dt.year
    
    # Calculate completion rate by year
    yearly_data = plot_data.groupby('StartYear').agg(
        completion_rate=('completion_status', 'mean'),
        trial_count=('completion_status', 'count')
    ).reset_index()
    
    # Filter years with sufficient data
    yearly_data = yearly_data[yearly_data['trial_count'] >= 10]
    
    # Create line plot
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot completion rate
    ax1.plot(yearly_data['StartYear'], yearly_data['completion_rate'], 'o-', color='blue', linewidth=2)
    ax1.set_xlabel('Start Year')
    ax1.set_ylabel('Completion Rate', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(0, 1)
    
    # Create second y-axis for trial count
    ax2 = ax1.twinx()
    ax2.bar(yearly_data['StartYear'], yearly_data['trial_count'], alpha=0.3, color='gray')
    ax2.set_ylabel('Number of Trials', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    
    plt.title('Trial Completion Rate by Start Year')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = FIGURE_DIR / f"completion_by_year_{timestamp}.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    print(f"Saved completion by year plot to {filepath}")
    return plt.gcf()

def plot_correlation_heatmap(df, features=None):
    """
    Plot correlation heatmap for selected features
    
    Args:
        df: DataFrame containing trial data
        features: List of feature columns to include in heatmap (if None, uses numeric columns)
    """
    if features is None:
        # Use numeric columns if no features specified
        features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Filter to only include specified features that exist in the DataFrame
    valid_features = [f for f in features if f in df.columns]
    
    if len(valid_features) < 2:
        print("Not enough valid features for correlation heatmap")
        return
    
    plt.figure(figsize=(14, 12))
    
    # Calculate correlation matrix
    corr_matrix = df[valid_features].corr()
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=True, fmt='.2f',
                linewidths=0.5, vmin=-1, vmax=1)
    
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = FIGURE_DIR / f"correlation_heatmap_{timestamp}.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    print(f"Saved correlation heatmap to {filepath}")
    return plt.gcf()

def create_eda_dashboard(df):
    """
    Create a comprehensive EDA dashboard with multiple visualizations
    
    Args:
        df: DataFrame containing trial data
    """
    # Set plotting style
    set_plotting_style()
    
    print("Generating EDA dashboard...")
    
    # Create individual plots
    plot_trial_status_distribution(df)
    plot_trial_phases(df)
    plot_enrollment_distribution(df)
    plot_duration_by_phase(df)
    plot_completion_rate_by_sponsor(df)
    plot_enrollment_vs_duration(df)
    plot_completion_by_year(df)
    
    # Create correlation heatmap for key features
    key_features = [
        'trial_duration_days', 'EnrollmentCount', 'completion_status',
        'is_industry_sponsored', 'is_multi_country', 'has_us_sites',
        'is_phase_2', 'is_phase_3', 'is_randomized', 'is_double_blind'
    ]
    plot_correlation_heatmap(df, key_features)
    
    print("EDA dashboard generation complete!")

def main():
    """
    Main function to execute visualization pipeline
    """
    try:
        # Find the most recent processed data file
        csv_files = list(PROCESSED_DATA_DIR.glob("processed_oncology_trials_*.csv"))
        if not csv_files:
            raise FileNotFoundError("No processed data files found")
        
        # Sort by modification time (most recent first)
        csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_data_path = csv_files[0]
        
        print(f"Loading data from {latest_data_path}")
        df = pd.read_csv(latest_data_path)
        
        # Create EDA dashboard
        create_eda_dashboard(df)
        
    except Exception as e:
        print(f"Error in visualization pipeline: {e}")

if __name__ == "__main__":
    main()
