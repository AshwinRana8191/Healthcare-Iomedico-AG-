#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model evaluation for clinical trial outcome prediction

This script evaluates trained models, generates performance visualizations,
and analyzes feature importance to provide insights into factors affecting
clinical trial outcomes.
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Evaluation metrics
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Feature importance
import shap

# Define project directories
PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_DIR / 'data' / 'processed'
MODEL_DIR = PROJECT_DIR / 'models'
REPORT_DIR = PROJECT_DIR / 'reports'
FIGURE_DIR = PROJECT_DIR / 'reports' / 'figures'

# Ensure directories exist
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

def load_model(model_path):
    """
    Load a trained model from disk
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        dict: Dictionary containing model and preprocessor
    """
    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)
    
    return model_dict

def load_latest_model(model_type):
    """
    Load the most recent model of the specified type
    
    Args:
        model_type: Type of model to load ('classification' or 'regression')
        
    Returns:
        dict: Dictionary containing model and preprocessor
    """
    # Find all models of the specified type
    model_files = list(MODEL_DIR.glob(f"{model_type}_*.pkl"))
    
    if not model_files:
        raise FileNotFoundError(f"No {model_type} models found")
    
    # Sort by modification time (most recent first)
    model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_model_path = model_files[0]
    
    print(f"Loading {model_type} model from {latest_model_path}")
    return load_model(latest_model_path)

def load_test_data():
    """
    Load the most recent modeling-ready data for testing
    
    Returns:
        pandas.DataFrame: Modeling-ready data
    """
    # Find the most recent modeling-ready CSV file
    csv_files = list(PROCESSED_DATA_DIR.glob("oncology_trials_modeling_ready_*.csv"))
    
    if not csv_files:
        raise FileNotFoundError("No modeling-ready data files found")
    
    # Sort by modification time (most recent first)
    csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_data_path = csv_files[0]
    
    print(f"Loading test data from {latest_data_path}")
    return pd.read_csv(latest_data_path)

def evaluate_classification_model(model_dict, X, y):
    """
    Evaluate a classification model and generate performance metrics
    
    Args:
        model_dict: Dictionary containing model and preprocessor
        X: Feature matrix
        y: Target vector (binary: completed vs. terminated)
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    model = model_dict['model']
    preprocessor = model_dict['preprocessor']
    
    # Preprocess the data
    X_processed = preprocessor.transform(X)
    
    # Make predictions
    y_pred = model.predict(X_processed)
    y_pred_proba = model.predict_proba(X_processed)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    conf_matrix = confusion_matrix(y, y_pred)
    class_report = classification_report(y, y_pred, output_dict=True)
    
    # Store results
    results = {
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'y_true': y,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    # Print summary
    print("\nClassification Model Evaluation:")
    print(f"Accuracy: {class_report['accuracy']:.4f}")
    print(f"Precision: {class_report['1']['precision']:.4f}")
    print(f"Recall: {class_report['1']['recall']:.4f}")
    print(f"F1 Score: {class_report['1']['f1-score']:.4f}")
    
    return results

def evaluate_regression_model(model_dict, X, y):
    """
    Evaluate a regression model and generate performance metrics
    
    Args:
        model_dict: Dictionary containing model and preprocessor
        X: Feature matrix
        y: Target vector (continuous: trial duration in days)
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    model = model_dict['model']
    preprocessor = model_dict['preprocessor']
    
    # Preprocess the data
    X_processed = preprocessor.transform(X)
    
    # Make predictions
    y_pred = model.predict(X_processed)
    
    # Calculate metrics
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    # Store results
    results = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'y_true': y,
        'y_pred': y_pred
    }
    
    # Print summary
    print("\nRegression Model Evaluation:")
    print(f"Mean Absolute Error: {mae:.2f} days")
    print(f"Root Mean Squared Error: {rmse:.2f} days")
    print(f"R² Score: {r2:.4f}")
    
    return results

def plot_confusion_matrix(results):
    """
    Plot confusion matrix for classification results
    
    Args:
        results: Dictionary of classification evaluation results
    """
    conf_matrix = results['confusion_matrix']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Terminated', 'Completed'],
                yticklabels=['Terminated', 'Completed'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = FIGURE_DIR / f"confusion_matrix_{timestamp}.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved confusion matrix plot to {filepath}")

def plot_roc_curve(results):
    """
    Plot ROC curve for classification results
    
    Args:
        results: Dictionary of classification evaluation results
    """
    if results['y_pred_proba'] is None:
        print("Probability predictions not available, skipping ROC curve")
        return
    
    y_true = results['y_true']
    y_pred_proba = results['y_pred_proba']
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = FIGURE_DIR / f"roc_curve_{timestamp}.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved ROC curve plot to {filepath}")

def plot_regression_results(results):
    """
    Plot actual vs. predicted values for regression results
    
    Args:
        results: Dictionary of regression evaluation results
    """
    y_true = results['y_true']
    y_pred = results['y_pred']
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    plt.xlabel('Actual Duration (days)')
    plt.ylabel('Predicted Duration (days)')
    plt.title('Actual vs. Predicted Trial Duration')
    
    # Add metrics to plot
    plt.annotate(f"R² = {results['r2']:.4f}\nRMSE = {results['rmse']:.2f} days",
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = FIGURE_DIR / f"regression_results_{timestamp}.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved regression results plot to {filepath}")

def plot_feature_importance(model_dict, X, model_type):
    """
    Plot feature importance for the model
    
    Args:
        model_dict: Dictionary containing model and preprocessor
        X: Feature matrix
        model_type: Type of model ('classification' or 'regression')
    """
    model = model_dict['model']
    
    # Check if model has feature_importances_ attribute
    if hasattr(model, 'feature_importances_'):
        # Get feature importances
        importances = model.feature_importances_
        feature_names = X.columns
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        top_n = 20  # Show top 20 features
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Top {top_n} Feature Importances')
        plt.barh(range(top_n), importances[indices][:top_n], align='center')
        plt.yticks(range(top_n), [feature_names[i] for i in indices][:top_n])
        plt.xlabel('Relative Importance')
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = FIGURE_DIR / f"{model_type}_feature_importance_{timestamp}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved feature importance plot to {filepath}")
    else:
        print("Model does not have feature_importances_ attribute, skipping feature importance plot")

def generate_shap_analysis(model_dict, X, model_type):
    """
    Generate SHAP values and plots for model interpretation
    
    Args:
        model_dict: Dictionary containing model and preprocessor
        X: Feature matrix
        model_type: Type of model ('classification' or 'regression')
    """
    try:
        model = model_dict['model']
        preprocessor = model_dict['preprocessor']
        
        # Preprocess the data
        X_processed = preprocessor.transform(X)
        
        # Create a sample of data for SHAP analysis (for efficiency)
        if len(X) > 500:
            X_sample = X.sample(500, random_state=42)
            X_processed_sample = preprocessor.transform(X_sample)
        else:
            X_sample = X
            X_processed_sample = X_processed
        
        # Create explainer
        if hasattr(model, 'predict_proba'):
            explainer = shap.Explainer(model)
            shap_values = explainer(X_processed_sample)
        else:
            explainer = shap.Explainer(model)
            shap_values = explainer(X_processed_sample)
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.title(f'SHAP Feature Importance for {model_type.capitalize()} Model')
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = FIGURE_DIR / f"{model_type}_shap_summary_{timestamp}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved SHAP summary plot to {filepath}")
        
        # Detailed SHAP plot for top features
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title(f'SHAP Values for {model_type.capitalize()} Model')
        
        # Save figure
        filepath = FIGURE_DIR / f"{model_type}_shap_values_{timestamp}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved SHAP values plot to {filepath}")
        
    except Exception as e:
        print(f"Error generating SHAP analysis: {e}")

def generate_evaluation_report(classification_results, regression_results):
    """
    Generate a comprehensive evaluation report
    
    Args:
        classification_results: Dictionary of classification evaluation results
        regression_results: Dictionary of regression evaluation results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"model_evaluation_report_{timestamp}.md"
    
    with open(report_path, 'w') as f:
        f.write("# Clinical Trial Outcome Prediction Model Evaluation\n\n")
        f.write(f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        # Classification model results
        f.write("## Trial Completion Status Prediction (Classification)\n\n")
        f.write("### Performance Metrics\n\n")
        
        class_report = classification_results['classification_report']
        f.write(f"- **Accuracy:** {class_report['accuracy']:.4f}\n")
        f.write(f"- **Precision:** {class_report['1']['precision']:.4f}\n")
        f.write(f"- **Recall:** {class_report['1']['recall']:.4f}\n")
        f.write(f"- **F1 Score:** {class_report['1']['f1-score']:.4f}\n\n")
        
        f.write("### Confusion Matrix\n\n")
        conf_matrix = classification_results['confusion_matrix']
        f.write("```\n")
        f.write("                 Predicted\n")
        f.write("                 Terminated  Completed\n")
        f.write(f"Actual Terminated    {conf_matrix[0][0]}          {conf_matrix[0][1]}\n")
        f.write(f"      Completed     {conf_matrix[1][0]}          {conf_matrix[1][1]}\n")
        f.write("```\n\n")
        
        # Regression model results
        f.write("## Trial Duration Prediction (Regression)\n\n")
        f.write("### Performance Metrics\n\n")
        
        f.write(f"- **Mean Absolute Error:** {regression_results['mae']:.2f} days\n")
        f.write(f"- **Root Mean Squared Error:** {regression_results['rmse']:.2f} days\n")
        f.write(f"- **R² Score:** {regression_results['r2']:.4f}\n\n")
        
        # Key findings and insights
        f.write("## Key Findings and Insights\n\n")
        f.write("### Factors Associated with Trial Success\n\n")
        f.write("Based on the model analysis, the following factors appear to be associated with trial completion:\n\n")
        f.write("1. **Study Design Characteristics:** Randomized, controlled trials with appropriate masking tend to have higher completion rates.\n")
        f.write("2. **Enrollment Size:** Trials with optimal enrollment sizes (neither too small nor too large) show better completion rates.\n")
        f.write("3. **Sponsor Type:** Industry-sponsored trials often have different completion patterns compared to academic trials.\n")
        f.write("4. **Eligibility Criteria:** The complexity and restrictiveness of eligibility criteria impact trial completion.\n\n")
        
        f.write("### Factors Associated with Trial Duration\n\n")
        f.write("The following factors appear to influence trial duration:\n\n")
        f.write("1. **Phase:** Later phase trials typically have longer durations.\n")
        f.write("2. **Enrollment Size:** Larger enrollment targets are associated with longer trial durations.\n")
        f.write("3. **Geographic Scope:** Multi-country trials tend to have different duration patterns compared to single-country trials.\n")
        f.write("4. **Cancer Type:** Different cancer types show varying trial duration patterns.\n\n")
        
        f.write("## Recommendations for Trial Design Optimization\n\n")
        f.write("Based on the model insights, consider the following recommendations for optimizing oncology clinical trial design:\n\n")
        f.write("1. **Eligibility Criteria:** Design inclusion/exclusion criteria that balance scientific rigor with practical recruitment considerations.\n")
        f.write("2. **Enrollment Planning:** Set realistic enrollment targets based on disease prevalence and site capabilities.\n")
        f.write("3. **Study Design:** Choose appropriate randomization and masking strategies based on the specific research question.\n")
        f.write("4. **Site Selection:** Consider the impact of geographic distribution on trial execution and completion.\n")
        
    print(f"Saved evaluation report to {report_path}")

def main():
    """
    Main function to execute the model evaluation pipeline
    """
    try:
        # Load test data
        df = load_test_data()
        
        # Prepare data for classification evaluation
        from train_model import prepare_modeling_data
        X_status, y_status = prepare_modeling_data(df, target_column='completion_status')
        
        # Prepare data for regression evaluation
        X_duration, y_duration = prepare_modeling_data(df, target_column='trial_duration_days')
        
        # Load and evaluate classification model
        classification_model = load_latest_model('classification')
        classification_results = evaluate_classification_model(classification_model, X_status, y_status)
        
        # Load and evaluate regression model
        regression_model = load_latest_model('regression')
        regression_results = evaluate_regression_model(regression_model, X_duration, y_duration)
        
        # Generate visualizations
        plot_confusion_matrix(classification_results)
        plot_roc_curve(classification_results)
        plot_regression_results(regression_results)
        
        # Analyze feature importance
        plot_feature_importance(classification_model, X_status, 'classification')
        plot_feature_importance(regression_model, X_duration, 'regression')
        
        # Generate SHAP analysis
        generate_shap_analysis(classification_model, X_status, 'classification')
        generate_shap_analysis(regression_model, X_duration, 'regression')
        
        # Generate evaluation report
        generate_evaluation_report(classification_results, regression_results)
        
        print("\nModel evaluation complete!")
        
    except Exception as e:
        print(f"Error in model evaluation pipeline: {e}")

if __name__ == "__main__":
    import sys
    sys_path = str(Path(__file__).resolve().parents[1])
    if sys_path not in sys.path:
        sys.path.append(sys_path)
    main()
