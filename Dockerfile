# Use NVIDIA CUDA base image with Python
FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training script
COPY train.py .

# Copy any additional files
COPY config/ config/ 2>/dev/null || true
COPY data/ data/ 2>/dev/null || true

# Create directories for model output
RUN mkdir -p /app/results /app/fine_tuned_model

# Set permissions
RUN chmod +x train.py

# Default command
CMD ["python", "train.py", "--help"]