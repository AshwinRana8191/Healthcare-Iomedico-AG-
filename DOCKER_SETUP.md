# Docker Setup for Oncology Clinical Trial Prediction Pipeline

This document provides instructions for setting up and running the oncology clinical trial prediction pipeline using Docker.

## Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop/) installed on your system
- [Docker Compose](https://docs.docker.com/compose/install/) installed on your system

## Project Structure

The Docker setup consists of the following files:

- `Dockerfile`: Defines the container environment with all required dependencies
- `docker-compose.yml`: Configures services for running the pipeline and Jupyter Lab
- `run_pipeline.sh` (Linux/Mac) or `run_pipeline.bat` (Windows): Scripts to automate the build and execution process
- `pipeline.py`: The main Python script that orchestrates the entire workflow

## Getting Started

### Building and Running the Pipeline

#### On Windows

```bash
# Run the batch script
.\run_pipeline.bat
```

#### On Linux/Mac

```bash
# Make the script executable
chmod +x run_pipeline.sh

# Run the shell script
./run_pipeline.sh
```

### Manual Execution

If you prefer to run commands manually:

```bash
# Build the Docker image
docker-compose build

# Run the pipeline
docker-compose up oncology-trial-pipeline

# Run Jupyter Lab for interactive analysis (optional)
docker-compose up jupyter
```

## Accessing Results

The pipeline mounts the following directories from your local machine to the Docker container:

- `./data`: Contains raw and processed data
- `./reports`: Contains generated reports and figures
- `./logs`: Contains pipeline execution logs

All results will be available in these directories after the pipeline completes.

## Customizing the Pipeline

To customize the pipeline execution:

1. Modify the `pipeline.py` script to change the workflow
2. Update the `Dockerfile` if additional dependencies are needed
3. Adjust volume mounts in `docker-compose.yml` if different directories need to be shared

## Troubleshooting

- If you encounter permission issues with mounted volumes, ensure your user has appropriate permissions
- Check the logs directory for detailed execution logs if the pipeline fails
- For Docker-related issues, refer to the [Docker documentation](https://docs.docker.com/)

## Additional Resources

- For more information on the project structure and methodology, refer to the main `README.md`
- For details on the data processing and modeling steps, explore the notebooks in the `notebooks/` directory