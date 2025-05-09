# Clinical Trial Data Directory

This directory contains sample data files for the oncology clinical trial analysis project. The data is organized into two main subdirectories:

## Raw Data (`/raw`)

Contains the original, unprocessed data collected from clinical trial sources:

- `clinical_trials_sample.csv`: Sample dataset of oncology clinical trials with basic information including trial identifiers, conditions, interventions, enrollment figures, and timeline data.

## Processed Data (`/processed`)

Contains cleaned, transformed, and feature-engineered datasets ready for analysis and modeling:

- `clinical_trials_processed.csv`: Processed version of the raw data with additional derived features such as duration calculations, numeric encodings of categorical variables, and success indicators.

- `feature_importance.csv`: Dataset containing the importance scores of various features in predicting clinical trial outcomes, categorized by feature type.

## Data Dictionary

### Raw Data Fields

- `nct_id`: Unique identifier for the clinical trial
- `title`: Title of the clinical trial
- `status`: Current status of the trial (e.g., Completed, Active, Recruiting)
- `phase`: Trial phase (e.g., Phase 1, Phase 2, Phase 3)
- `condition`: Primary condition being studied
- `intervention_type`: Type of intervention being tested
- `start_date`: Date when the trial began
- `completion_date`: Actual or anticipated completion date
- `enrollment`: Number of participants enrolled
- `sponsor_type`: Type of organization sponsoring the trial

### Processed Data Additional Fields

- `phase_numeric`: Numerical representation of trial phase
- `condition_category`: Categorized cancer type
- `intervention_category`: Categorized intervention approach
- `duration_months`: Calculated trial duration in months
- `sponsor_category`: Categorized sponsor type
- `success_indicator`: Binary indicator of trial success (1=success, 0=not successful/ongoing)
- `time_to_completion`: Ratio of actual to planned completion time
- `text_complexity_score`: Measure of protocol text complexity
- `has_biomarker`: Whether the trial uses biomarkers for patient selection
- `patient_diversity_score`: Score representing diversity of patient population

## Note

These are sample files created for demonstration purposes. In a real project implementation, these would be replaced with actual clinical trial data from sources like ClinicalTrials.gov or proprietary databases.