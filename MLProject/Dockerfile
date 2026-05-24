# Dockerfile untuk MLflow Model Serving
# Digunakan setelah model di-train dengan MLflow
# Build: docker build -t wine-classification-model .
# Run:   docker run -p 5001:8080 wine-classification-model

FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy MLflow project files
COPY modelling.py .
COPY MLproject .
COPY conda.yaml .
COPY dataset_preprocessing/ ./dataset_preprocessing/

# Expose port untuk model serving
EXPOSE 8080

# Default command: train model with MLflow
CMD ["python", "modelling.py"]
