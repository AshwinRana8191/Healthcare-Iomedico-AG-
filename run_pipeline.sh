#!/bin/bash

# Script to build and run the oncology clinical trial prediction pipeline

echo "=== Building Docker image ==="
docker-compose build

echo "\n=== Running the pipeline ==="
docker-compose up oncology-trial-pipeline

echo "\n=== Pipeline execution completed ==="

# Uncomment the line below to start Jupyter Lab for interactive analysis
# echo "\n=== Starting Jupyter Lab ==="
# docker-compose up jupyter