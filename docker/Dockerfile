# InfraMind Training Environment
# Base image with CUDA support for A100
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Create working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy inframind package
COPY inframind/ ./inframind/
COPY pyproject.toml .

# Install inframind
RUN pip install -e .

# Copy training scripts
COPY modal_app.py .
COPY configs/ ./configs/

# Default command
CMD ["python", "-c", "from inframind import create_dataset; print(f'Dataset: {len(create_dataset())} samples')"]
