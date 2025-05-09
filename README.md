# Predictive Modeling for Oncology Clinical Trial Success and Efficiency Analysis

## Project Overview
This project applies advanced data science techniques to analyze publicly available oncology clinical trial data from ClinicalTrials.gov. The goal is to identify factors associated with trial success and duration, build predictive models for trial outcomes, and develop a reproducible analysis pipeline.

## Objectives
- Identify factors associated with trial success (completion vs. termination) and duration
- Build predictive models for trial outcomes
- Develop a reproducible and efficient analysis pipeline
- Generate insights relevant to optimizing clinical trial design and execution in oncology
- Enable containerized deployment with Docker
- Implement CI/CD pipeline for automated testing and deployment

## Project Structure
```
├── data/                      # Data directory
│   ├── raw/                   # Raw data from ClinicalTrials.gov
│   └── processed/             # Cleaned and processed data
├── notebooks/                 # Jupyter notebooks for exploration and analysis
│   ├── 1_data_exploration.ipynb     # Exploratory data analysis
│   ├── 2_feature_engineering.ipynb  # Feature creation and preprocessing
│   ├── 3_model_training.ipynb       # Model training and initial evaluation
│   └── 4_model_evaluation.ipynb     # Comprehensive model evaluation and insights
├── src/                       # Source code
│   ├── data/                  # Data acquisition and processing scripts
│   │   ├── acquisition.py     # Scripts to fetch data from ClinicalTrials.gov
│   │   └── preprocessing.py   # Data cleaning and preprocessing
│   ├── features/              # Feature engineering
│   │   ├── build_features.py  # Feature creation from structured data
│   │   └── text_features.py   # NLP for text fields
│   ├── visualization/         # Visualization utilities
│   │   └── visualize.py       # Plotting functions
│   └── models/                # Model training and evaluation
│       ├── train_model.py     # Model training
│       └── evaluate_model.py  # Model evaluation
├── reports/                   # Generated analysis reports and figures
│   └── figures/               # Generated figures
├── .github/                   # GitHub configuration
│   └── workflows/             # GitHub Actions workflows
│       ├── python-tests.yml   # Workflow for running Python tests
│       └── deploy.yml         # Workflow for deployment
├── Dockerfile                 # Docker configuration for containerization
├── docker-compose.yml         # Docker Compose configuration for multi-container setup
├── .dockerignore              # Files to exclude from Docker context
├── pipeline.py               # End-to-end pipeline script
├── run_pipeline.sh           # Shell script to run the pipeline (Unix/Linux/Mac)
├── run_pipeline.bat          # Batch script to run the pipeline (Windows)
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Methodology

### 1. Data Exploration
- Analyze distributions of key variables in oncology trials
- Visualize relationships between features and outcomes
- Identify patterns and trends in trial characteristics

### 2. Feature Engineering
- Create target variables: Trial Status (Binary: Completed vs. Terminated), Trial Duration (Numerical)
- Extract features from structured data
- Apply NLP to text fields (Eligibility Criteria, Intervention Description)
- Handle missing values and standardize formats

### 3. Model Training
- Develop classification models for Trial Status prediction
- Develop regression models for Trial Duration prediction
- Compare performance of multiple algorithms
- Identify the most effective modeling approaches

### 4. Model Evaluation & Insights
- Evaluate models using appropriate metrics
- Analyze feature importance and SHAP values
- Visualize model performance and predictions
- Generate actionable insights for trial design optimization

### 5. Pipeline Development & Reproducibility
- Structure code logically
- Use version control
- Encapsulate processing and modeling steps into reproducible pipeline
- Implement containerization for environment consistency
- Set up CI/CD for automated testing and deployment

### 6. End-to-End Workflow
- Data acquisition from ClinicalTrials.gov
- Data preprocessing and cleaning
- Feature engineering and selection
- Model training and hyperparameter tuning
- Model evaluation and interpretation
- Results visualization and reporting
- Automated pipeline execution

## Setup and Installation

### Standard Setup
```bash
# Clone the repository
git clone [repository-url]
cd oncology-trial-prediction

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the data acquisition script
python src/data/acquisition.py
```

### Docker Setup
```bash
# Build and run using Docker Compose
docker-compose up --build

# Or build and run the Docker container directly
docker build -t oncology-trial-prediction .
docker run -p 8000:8000 oncology-trial-prediction
```

## Usage

### Running the Pipeline
The project includes scripts to run the complete analysis pipeline:

**On Linux/Mac:**
```bash
./run_pipeline.sh
```

**On Windows:**
```bash
run_pipeline.bat
```

**Using Python directly:**
```bash
python pipeline.py
```

### Notebooks
The Jupyter notebooks in the `notebooks/` directory provide detailed analysis steps:
1. `1_data_exploration.ipynb` - Initial data exploration and visualization
2. `2_feature_engineering.ipynb` - Feature creation and preprocessing
3. `3_model_training.ipynb` - Model training and initial evaluation
4. `4_model_evaluation.ipynb` - Comprehensive model evaluation and insights

## CI/CD Pipeline

This project uses GitHub Actions for continuous integration and deployment:

### Automated Testing
The `python-tests.yml` workflow automatically runs tests on every push and pull request to ensure code quality and functionality.

### Deployment
The `deploy.yml` workflow handles automated deployment when changes are merged to the main branch.

## Docker Implementation

The project is containerized using Docker for consistent environments and easy deployment:

- `Dockerfile` - Defines the container configuration
- `docker-compose.yml` - Orchestrates multi-container setup
- `.dockerignore` - Excludes unnecessary files from the Docker context

Refer to the `DOCKER_SETUP.md` file for detailed Docker implementation information.

## License
This project is licensed under the MIT License - see the LICENSE file for details.