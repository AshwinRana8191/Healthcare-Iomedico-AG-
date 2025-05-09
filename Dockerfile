FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install spaCy model
RUN python -m spacy download en_core_web_sm

# Copy the project files
COPY . .

# Create necessary directories if they don't exist
RUN mkdir -p data/raw data/processed reports/figures

# Set environment variables
ENV PYTHONPATH="${PYTHONPATH}:/app"
ENV PYTHONUNBUFFERED=1

# Run the pipeline script by default
CMD ["python", "pipeline.py"]