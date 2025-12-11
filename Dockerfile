# Dockerfile for IEEE Fraud Detection Project
# Base image: Debian Linux
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY notebooks/ ./notebooks/

# Create necessary directories
RUN mkdir -p data models results reports

# Set Python path
ENV PYTHONPATH=/app

# Default command (can be overridden)
CMD ["python", "-c", "import sys; print('IEEE Fraud Detection Project - Docker Container Ready')"]

