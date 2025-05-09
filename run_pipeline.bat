@echo off

echo === Building Docker image ===
docker-compose build

echo.
echo === Running the pipeline ===
docker-compose up oncology-trial-pipeline

echo.
echo === Pipeline execution completed ===

REM Uncomment the line below to start Jupyter Lab for interactive analysis
REM echo.
REM echo === Starting Jupyter Lab ===
REM docker-compose up jupyter