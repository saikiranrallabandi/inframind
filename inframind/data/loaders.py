"""
InfraMind Data Loaders - Load training data from various sources.
"""

import json
import os
from typing import List, Dict, Optional


def load_training_data_from_json(paths: Optional[List[str]] = None) -> List[Dict]:
    """
    Load high-quality training data from JSON file.

    Args:
        paths: Optional list of paths to try. If None, uses default paths.

    Returns:
        List of training samples with instruction, input, output, category.
    """
    if paths is None:
        paths = [
            "/root/training_data.json",  # Modal container path
            "data/training_data.json",   # Local path
            "data/real_code_2k.json",    # Real GitHub code
        ]

    for path in paths:
        if os.path.exists(path):
            print(f"  Loading training data from: {path}")
            with open(path) as f:
                data = json.load(f)
            print(f"  Loaded {len(data)} samples")
            return data

    # Fallback: embedded training data
    print("  Using embedded training data")
    return EMBEDDED_TRAINING_DATA


# =============================================================================
# EMBEDDED HIGH-QUALITY TRAINING DATA
# Generated with actual code outputs for fallback
# =============================================================================
EMBEDDED_TRAINING_DATA = [
    {
        "instruction": "Create Kubernetes Service",
        "input": "LoadBalancer type for nginx on port 80",
        "output": """apiVersion: v1
kind: Service
metadata:
  name: nginx-loadbalancer
  labels:
    app: nginx
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 80
    protocol: TCP
    name: http
  selector:
    app: nginx""",
        "category": "kubernetes"
    },
    {
        "instruction": "Create Terraform for S3 bucket",
        "input": "with versioning enabled",
        "output": """terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

resource "aws_s3_bucket" "main" {
  bucket = var.bucket_name

  tags = {
    Name        = var.bucket_name
    Environment = var.environment
  }
}

resource "aws_s3_bucket_versioning" "main" {
  bucket = aws_s3_bucket.main.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "main" {
  bucket = aws_s3_bucket.main.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

variable "bucket_name" {
  type = string
}

variable "environment" {
  type    = string
  default = "dev"
}""",
        "category": "terraform"
    },
    {
        "instruction": "Create Dockerfile for Python application",
        "input": "with FastAPI and uvicorn",
        "output": """FROM python:3.11-slim

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]""",
        "category": "docker"
    },
    {
        "instruction": "Create GitHub Actions workflow",
        "input": "run tests on push to main",
        "output": """name: Run Tests

on:
  push:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Setup Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
        cache: 'pip'

    - name: Install dependencies
      run: pip install -r requirements.txt

    - name: Run tests
      run: pytest tests/ -v""",
        "category": "cicd"
    },
    {
        "instruction": "Create Kubernetes Deployment",
        "input": "nginx with 3 replicas",
        "output": """apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx
  labels:
    app: nginx
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.25-alpine
        ports:
        - containerPort: 80
        resources:
          requests:
            memory: "64Mi"
            cpu: "100m"
          limits:
            memory: "128Mi"
            cpu: "200m"
        livenessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 10
          periodSeconds: 10""",
        "category": "kubernetes"
    },
    {
        "instruction": "Create Terraform for AWS EC2 instance",
        "input": "t3.medium instance type",
        "output": """terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

resource "aws_instance" "main" {
  ami           = data.aws_ami.amazon_linux.id
  instance_type = "t3.medium"

  vpc_security_group_ids = [aws_security_group.instance.id]

  tags = {
    Name = var.instance_name
  }
}

data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
  }
}

resource "aws_security_group" "instance" {
  name_prefix = "instance-sg-"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

variable "instance_name" {
  type    = string
  default = "my-instance"
}

variable "allowed_ssh_cidr" {
  type    = string
  default = "0.0.0.0/0"
}""",
        "category": "terraform"
    },
    {
        "instruction": "Create docker-compose.yml",
        "input": "for web app with PostgreSQL and Redis",
        "output": """version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/app
      - REDIS_URL=redis://redis:6379
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: app
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
  redis_data:""",
        "category": "docker"
    },
    {
        "instruction": "Create Kubernetes ConfigMap",
        "input": "with database connection settings",
        "output": """apiVersion: v1
kind: ConfigMap
metadata:
  name: database-config
data:
  DB_HOST: "postgres-service"
  DB_PORT: "5432"
  DB_NAME: "myapp"
  DB_SSL_MODE: "require"
  CONNECTION_POOL_SIZE: "10"
  CONNECTION_TIMEOUT: "30"
---
apiVersion: v1
kind: Secret
metadata:
  name: database-credentials
type: Opaque
stringData:
  DB_USER: "myapp_user"
  DB_PASSWORD: "changeme"  # Use sealed-secrets in production""",
        "category": "kubernetes"
    },
]
