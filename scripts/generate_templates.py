#!/usr/bin/env python3
"""
Template-based IaC training data generator.
Zero cost - no API calls required.
"""

import json
import random
import argparse
from pathlib import Path

# =============================================================================
# TERRAFORM TEMPLATES
# =============================================================================

TERRAFORM_TEMPLATES = {
    "ec2": {
        "instruction": "Create Terraform for AWS EC2 instance with {instance_type} instance type",
        "output": '''resource "aws_instance" "{name}" {{
  ami           = "{ami_id}"
  instance_type = "{instance_type}"

  tags = {{
    Name        = "{name}"
    Environment = "{env}"
  }}
}}'''
    },
    "ec2_with_sg": {
        "instruction": "Create Terraform for EC2 instance with security group allowing port {port}",
        "output": '''resource "aws_security_group" "{name}_sg" {{
  name        = "{name}-sg"
  description = "Security group for {name}"

  ingress {{
    from_port   = {port}
    to_port     = {port}
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }}

  egress {{
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }}
}}

resource "aws_instance" "{name}" {{
  ami                    = "{ami_id}"
  instance_type          = "{instance_type}"
  vpc_security_group_ids = [aws_security_group.{name}_sg.id]

  tags = {{
    Name = "{name}"
  }}
}}'''
    },
    "s3": {
        "instruction": "Create Terraform for S3 bucket with {feature}",
        "output": '''resource "aws_s3_bucket" "{name}" {{
  bucket = "{bucket_name}"

  tags = {{
    Name        = "{name}"
    Environment = "{env}"
  }}
}}

resource "aws_s3_bucket_versioning" "{name}_versioning" {{
  bucket = aws_s3_bucket.{name}.id
  versioning_configuration {{
    status = "Enabled"
  }}
}}'''
    },
    "s3_website": {
        "instruction": "Create Terraform for S3 static website hosting",
        "output": '''resource "aws_s3_bucket" "{name}" {{
  bucket = "{bucket_name}"
}}

resource "aws_s3_bucket_website_configuration" "{name}_website" {{
  bucket = aws_s3_bucket.{name}.id

  index_document {{
    suffix = "index.html"
  }}

  error_document {{
    key = "error.html"
  }}
}}

resource "aws_s3_bucket_public_access_block" "{name}_public" {{
  bucket = aws_s3_bucket.{name}.id

  block_public_acls       = false
  block_public_policy     = false
  ignore_public_acls      = false
  restrict_public_buckets = false
}}'''
    },
    "vpc": {
        "instruction": "Create Terraform for VPC with CIDR {cidr}",
        "output": '''resource "aws_vpc" "{name}" {{
  cidr_block           = "{cidr}"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {{
    Name = "{name}"
  }}
}}

resource "aws_internet_gateway" "{name}_igw" {{
  vpc_id = aws_vpc.{name}.id

  tags = {{
    Name = "{name}-igw"
  }}
}}'''
    },
    "vpc_with_subnets": {
        "instruction": "Create Terraform for VPC with {num_public} public and {num_private} private subnets",
        "output": '''resource "aws_vpc" "{name}" {{
  cidr_block           = "{cidr}"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {{
    Name = "{name}"
  }}
}}

resource "aws_subnet" "{name}_public" {{
  vpc_id                  = aws_vpc.{name}.id
  cidr_block              = "{public_cidr}"
  map_public_ip_on_launch = true
  availability_zone       = "{az}a"

  tags = {{
    Name = "{name}-public"
    Type = "public"
  }}
}}

resource "aws_subnet" "{name}_private" {{
  vpc_id            = aws_vpc.{name}.id
  cidr_block        = "{private_cidr}"
  availability_zone = "{az}a"

  tags = {{
    Name = "{name}-private"
    Type = "private"
  }}
}}

resource "aws_internet_gateway" "{name}_igw" {{
  vpc_id = aws_vpc.{name}.id

  tags = {{
    Name = "{name}-igw"
  }}
}}'''
    },
    "rds": {
        "instruction": "Create Terraform for RDS {engine} database",
        "output": '''resource "aws_db_instance" "{name}" {{
  identifier           = "{name}"
  engine               = "{engine}"
  engine_version       = "{engine_version}"
  instance_class       = "{instance_class}"
  allocated_storage    = {storage}
  storage_type         = "gp3"
  db_name              = "{db_name}"
  username             = var.db_username
  password             = var.db_password
  skip_final_snapshot  = true

  tags = {{
    Name        = "{name}"
    Environment = "{env}"
  }}
}}'''
    },
    "lambda": {
        "instruction": "Create Terraform for Lambda function with {runtime} runtime",
        "output": '''resource "aws_lambda_function" "{name}" {{
  filename         = "{filename}"
  function_name    = "{name}"
  role             = aws_iam_role.{name}_role.arn
  handler          = "{handler}"
  runtime          = "{runtime}"
  timeout          = {timeout}
  memory_size      = {memory}

  environment {{
    variables = {{
      ENVIRONMENT = "{env}"
    }}
  }}

  tags = {{
    Name = "{name}"
  }}
}}

resource "aws_iam_role" "{name}_role" {{
  name = "{name}-role"

  assume_role_policy = jsonencode({{
    Version = "2012-10-17"
    Statement = [{{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {{
        Service = "lambda.amazonaws.com"
      }}
    }}]
  }})
}}'''
    },
    "alb": {
        "instruction": "Create Terraform for Application Load Balancer",
        "output": '''resource "aws_lb" "{name}" {{
  name               = "{name}"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.{name}_sg.id]
  subnets            = var.subnet_ids

  tags = {{
    Name = "{name}"
  }}
}}

resource "aws_lb_target_group" "{name}_tg" {{
  name     = "{name}-tg"
  port     = {port}
  protocol = "HTTP"
  vpc_id   = var.vpc_id

  health_check {{
    path                = "/health"
    healthy_threshold   = 2
    unhealthy_threshold = 10
  }}
}}

resource "aws_lb_listener" "{name}_listener" {{
  load_balancer_arn = aws_lb.{name}.arn
  port              = "80"
  protocol          = "HTTP"

  default_action {{
    type             = "forward"
    target_group_arn = aws_lb_target_group.{name}_tg.arn
  }}
}}'''
    },
    "dynamodb": {
        "instruction": "Create Terraform for DynamoDB table with partition key {pk}",
        "output": '''resource "aws_dynamodb_table" "{name}" {{
  name           = "{name}"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "{pk}"

  attribute {{
    name = "{pk}"
    type = "S"
  }}

  tags = {{
    Name        = "{name}"
    Environment = "{env}"
  }}
}}'''
    },
    "sqs": {
        "instruction": "Create Terraform for SQS queue with {feature}",
        "output": '''resource "aws_sqs_queue" "{name}" {{
  name                       = "{name}"
  delay_seconds              = 0
  max_message_size           = 262144
  message_retention_seconds  = 345600
  receive_wait_time_seconds  = 10
  visibility_timeout_seconds = 30

  tags = {{
    Name        = "{name}"
    Environment = "{env}"
  }}
}}'''
    },
    "sns": {
        "instruction": "Create Terraform for SNS topic",
        "output": '''resource "aws_sns_topic" "{name}" {{
  name = "{name}"

  tags = {{
    Name        = "{name}"
    Environment = "{env}"
  }}
}}'''
    },
    "iam_role": {
        "instruction": "Create Terraform for IAM role with {service} access",
        "output": '''resource "aws_iam_role" "{name}" {{
  name = "{name}"

  assume_role_policy = jsonencode({{
    Version = "2012-10-17"
    Statement = [{{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {{
        Service = "{service}.amazonaws.com"
      }}
    }}]
  }})

  tags = {{
    Name = "{name}"
  }}
}}

resource "aws_iam_role_policy_attachment" "{name}_policy" {{
  role       = aws_iam_role.{name}.name
  policy_arn = "arn:aws:iam::aws:policy/{policy}"
}}'''
    },
    "cloudwatch_alarm": {
        "instruction": "Create Terraform for CloudWatch alarm for {metric}",
        "output": '''resource "aws_cloudwatch_metric_alarm" "{name}" {{
  alarm_name          = "{name}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "{metric}"
  namespace           = "{namespace}"
  period              = 300
  statistic           = "Average"
  threshold           = {threshold}
  alarm_description   = "Alarm when {metric} exceeds {threshold}"

  dimensions = {{
    InstanceId = var.instance_id
  }}

  alarm_actions = [var.sns_topic_arn]

  tags = {{
    Name = "{name}"
  }}
}}'''
    },
    "ecs_service": {
        "instruction": "Create Terraform for ECS Fargate service",
        "output": '''resource "aws_ecs_cluster" "{name}_cluster" {{
  name = "{name}-cluster"

  tags = {{
    Name = "{name}"
  }}
}}

resource "aws_ecs_task_definition" "{name}_task" {{
  family                   = "{name}"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "{cpu}"
  memory                   = "{memory}"
  execution_role_arn       = aws_iam_role.ecs_execution_role.arn

  container_definitions = jsonencode([{{
    name      = "{name}"
    image     = "{image}"
    essential = true
    portMappings = [{{
      containerPort = {port}
      protocol      = "tcp"
    }}]
  }}])
}}

resource "aws_ecs_service" "{name}" {{
  name            = "{name}"
  cluster         = aws_ecs_cluster.{name}_cluster.id
  task_definition = aws_ecs_task_definition.{name}_task.arn
  desired_count   = {replicas}
  launch_type     = "FARGATE"

  network_configuration {{
    subnets         = var.subnet_ids
    security_groups = [aws_security_group.{name}_sg.id]
  }}
}}'''
    },
}

# =============================================================================
# KUBERNETES TEMPLATES
# =============================================================================

KUBERNETES_TEMPLATES = {
    "deployment": {
        "instruction": "Create Kubernetes Deployment for {app} with {replicas} replicas",
        "output": '''apiVersion: apps/v1
kind: Deployment
metadata:
  name: {name}
  labels:
    app: {app}
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: {app}
  template:
    metadata:
      labels:
        app: {app}
    spec:
      containers:
      - name: {app}
        image: {image}
        ports:
        - containerPort: {port}
        resources:
          requests:
            memory: "{memory_request}"
            cpu: "{cpu_request}"
          limits:
            memory: "{memory_limit}"
            cpu: "{cpu_limit}"'''
    },
    "deployment_with_probes": {
        "instruction": "Create Kubernetes Deployment with health checks for {app}",
        "output": '''apiVersion: apps/v1
kind: Deployment
metadata:
  name: {name}
  labels:
    app: {app}
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: {app}
  template:
    metadata:
      labels:
        app: {app}
    spec:
      containers:
      - name: {app}
        image: {image}
        ports:
        - containerPort: {port}
        livenessProbe:
          httpGet:
            path: /health
            port: {port}
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: {port}
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "{memory_request}"
            cpu: "{cpu_request}"
          limits:
            memory: "{memory_limit}"
            cpu: "{cpu_limit}"'''
    },
    "service_clusterip": {
        "instruction": "Create Kubernetes ClusterIP Service for {app}",
        "output": '''apiVersion: v1
kind: Service
metadata:
  name: {name}
  labels:
    app: {app}
spec:
  type: ClusterIP
  selector:
    app: {app}
  ports:
  - port: {port}
    targetPort: {target_port}
    protocol: TCP'''
    },
    "service_loadbalancer": {
        "instruction": "Create Kubernetes LoadBalancer Service for {app}",
        "output": '''apiVersion: v1
kind: Service
metadata:
  name: {name}
  labels:
    app: {app}
spec:
  type: LoadBalancer
  selector:
    app: {app}
  ports:
  - port: {port}
    targetPort: {target_port}
    protocol: TCP'''
    },
    "service_nodeport": {
        "instruction": "Create Kubernetes NodePort Service for {app}",
        "output": '''apiVersion: v1
kind: Service
metadata:
  name: {name}
  labels:
    app: {app}
spec:
  type: NodePort
  selector:
    app: {app}
  ports:
  - port: {port}
    targetPort: {target_port}
    nodePort: {node_port}
    protocol: TCP'''
    },
    "configmap": {
        "instruction": "Create Kubernetes ConfigMap for {app} configuration",
        "output": '''apiVersion: v1
kind: ConfigMap
metadata:
  name: {name}
data:
  {key1}: "{value1}"
  {key2}: "{value2}"
  config.yaml: |
    environment: {env}
    logLevel: {log_level}'''
    },
    "secret": {
        "instruction": "Create Kubernetes Secret for {app}",
        "output": '''apiVersion: v1
kind: Secret
metadata:
  name: {name}
type: Opaque
stringData:
  {key1}: "{value1}"
  {key2}: "{value2}"'''
    },
    "ingress": {
        "instruction": "Create Kubernetes Ingress for {host}",
        "output": '''apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {name}
  annotations:
    kubernetes.io/ingress.class: nginx
spec:
  rules:
  - host: {host}
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: {service}
            port:
              number: {port}'''
    },
    "pvc": {
        "instruction": "Create Kubernetes PersistentVolumeClaim with {storage} storage",
        "output": '''apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: {name}
spec:
  accessModes:
    - {access_mode}
  storageClassName: {storage_class}
  resources:
    requests:
      storage: {storage}'''
    },
    "statefulset": {
        "instruction": "Create Kubernetes StatefulSet for {app}",
        "output": '''apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: {name}
spec:
  serviceName: {name}
  replicas: {replicas}
  selector:
    matchLabels:
      app: {app}
  template:
    metadata:
      labels:
        app: {app}
    spec:
      containers:
      - name: {app}
        image: {image}
        ports:
        - containerPort: {port}
        volumeMounts:
        - name: data
          mountPath: /data
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: {storage_class}
      resources:
        requests:
          storage: {storage}'''
    },
    "cronjob": {
        "instruction": "Create Kubernetes CronJob running {schedule}",
        "output": '''apiVersion: batch/v1
kind: CronJob
metadata:
  name: {name}
spec:
  schedule: "{schedule}"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: {name}
            image: {image}
            command: {command}
          restartPolicy: OnFailure'''
    },
    "hpa": {
        "instruction": "Create Kubernetes HorizontalPodAutoscaler for {app}",
        "output": '''apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {name}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {deployment}
  minReplicas: {min_replicas}
  maxReplicas: {max_replicas}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: {cpu_target}'''
    },
    "networkpolicy": {
        "instruction": "Create Kubernetes NetworkPolicy for {app}",
        "output": '''apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: {name}
spec:
  podSelector:
    matchLabels:
      app: {app}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: {allowed_app}
    ports:
    - protocol: TCP
      port: {port}'''
    },
}

# =============================================================================
# DOCKER TEMPLATES
# =============================================================================

DOCKER_TEMPLATES = {
    "python": {
        "instruction": "Create Dockerfile for Python {framework} application",
        "output": '''FROM python:{python_version}-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE {port}

CMD ["{cmd}", "{entry}"]'''
    },
    "python_multistage": {
        "instruction": "Create multi-stage Dockerfile for Python application",
        "output": '''FROM python:{python_version} AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:{python_version}-slim

WORKDIR /app
COPY --from=builder /usr/local/lib/python{python_version}/site-packages /usr/local/lib/python{python_version}/site-packages
COPY . .

EXPOSE {port}

CMD ["{cmd}", "{entry}"]'''
    },
    "node": {
        "instruction": "Create Dockerfile for Node.js {framework} application",
        "output": '''FROM node:{node_version}-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY . .

EXPOSE {port}

CMD ["node", "{entry}"]'''
    },
    "node_multistage": {
        "instruction": "Create multi-stage Dockerfile for Node.js application",
        "output": '''FROM node:{node_version}-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:{node_version}-alpine

WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules

EXPOSE {port}

CMD ["node", "dist/{entry}"]'''
    },
    "go": {
        "instruction": "Create Dockerfile for Go application",
        "output": '''FROM golang:{go_version}-alpine AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o main .

FROM alpine:latest

RUN apk --no-cache add ca-certificates
WORKDIR /root/
COPY --from=builder /app/main .

EXPOSE {port}

CMD ["./main"]'''
    },
    "java": {
        "instruction": "Create Dockerfile for Java Spring Boot application",
        "output": '''FROM eclipse-temurin:{java_version}-jdk-alpine AS builder

WORKDIR /app
COPY . .
RUN ./mvnw package -DskipTests

FROM eclipse-temurin:{java_version}-jre-alpine

WORKDIR /app
COPY --from=builder /app/target/*.jar app.jar

EXPOSE {port}

ENTRYPOINT ["java", "-jar", "app.jar"]'''
    },
    "nginx": {
        "instruction": "Create Dockerfile for Nginx static site",
        "output": '''FROM nginx:alpine

COPY {source_dir} /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]'''
    },
    "compose_web_db": {
        "instruction": "Create docker-compose.yml for {framework} with {database}",
        "output": '''version: '3.8'

services:
  app:
    build: .
    ports:
      - "{port}:{port}"
    environment:
      - DATABASE_URL={db_type}://user:password@db:5432/{db_name}
    depends_on:
      - db

  db:
    image: {db_image}
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB={db_name}
    volumes:
      - db_data:/var/lib/postgresql/data

volumes:
  db_data:'''
    },
    "compose_full_stack": {
        "instruction": "Create docker-compose.yml for full stack with {frontend}, {backend}, and {database}",
        "output": '''version: '3.8'

services:
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend

  backend:
    build: ./backend
    ports:
      - "{port}:{port}"
    environment:
      - DATABASE_URL={db_type}://user:password@db:5432/{db_name}
    depends_on:
      - db

  db:
    image: {db_image}
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB={db_name}
    volumes:
      - db_data:/var/lib/postgresql/data

volumes:
  db_data:'''
    },
}

# =============================================================================
# CI/CD TEMPLATES
# =============================================================================

CICD_TEMPLATES = {
    "github_test": {
        "instruction": "Create GitHub Actions workflow for running tests on {trigger}",
        "output": '''name: Tests

on:
  {trigger}:
    branches: [{branch}]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up {language}
      uses: {setup_action}
      with:
        {version_key}: '{version}'

    - name: Install dependencies
      run: {install_cmd}

    - name: Run tests
      run: {test_cmd}'''
    },
    "github_build_push": {
        "instruction": "Create GitHub Actions workflow for building and pushing Docker image",
        "output": '''name: Build and Push

on:
  push:
    branches: [{branch}]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3

    - name: Login to Container Registry
      uses: docker/login-action@v3
      with:
        registry: {registry}
        username: ${{{{ secrets.REGISTRY_USERNAME }}}}
        password: ${{{{ secrets.REGISTRY_PASSWORD }}}}

    - name: Build and push
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: {registry}/{image}:${{{{ github.sha }}}}'''
    },
    "github_deploy": {
        "instruction": "Create GitHub Actions workflow for deploying to {target}",
        "output": '''name: Deploy

on:
  push:
    branches: [{branch}]

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Configure credentials
      uses: {credentials_action}
      with:
        {credentials_key}: ${{{{ secrets.{credentials_secret} }}}}

    - name: Deploy to {target}
      run: |
        {deploy_cmd}'''
    },
    "gitlab_test": {
        "instruction": "Create GitLab CI pipeline for running tests",
        "output": '''stages:
  - test

test:
  stage: test
  image: {image}
  script:
    - {install_cmd}
    - {test_cmd}
  only:
    - {branch}'''
    },
    "gitlab_build_deploy": {
        "instruction": "Create GitLab CI pipeline for build and deploy",
        "output": '''stages:
  - build
  - deploy

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  only:
    - {branch}

deploy:
  stage: deploy
  image: {deploy_image}
  script:
    - {deploy_cmd}
  only:
    - {branch}
  environment:
    name: {env}'''
    },
}

# =============================================================================
# MLOPS TEMPLATES
# =============================================================================

MLOPS_TEMPLATES = {
    "sagemaker_endpoint": {
        "instruction": "Create Terraform for SageMaker endpoint with {instance_type}",
        "output": '''resource "aws_sagemaker_model" "{name}" {{
  name               = "{name}"
  execution_role_arn = aws_iam_role.sagemaker_role.arn

  primary_container {{
    image          = "{image}"
    model_data_url = "s3://{bucket}/{model_path}"
  }}
}}

resource "aws_sagemaker_endpoint_configuration" "{name}_config" {{
  name = "{name}-config"

  production_variants {{
    variant_name           = "primary"
    model_name             = aws_sagemaker_model.{name}.name
    initial_instance_count = {instance_count}
    instance_type          = "{instance_type}"
  }}
}}

resource "aws_sagemaker_endpoint" "{name}" {{
  name                 = "{name}"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.{name}_config.name
}}'''
    },
    "ml_dockerfile": {
        "instruction": "Create Dockerfile for ML training with {framework}",
        "output": '''FROM {base_image}

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "{entry}"]'''
    },
    "k8s_training_job": {
        "instruction": "Create Kubernetes Job for ML training",
        "output": '''apiVersion: batch/v1
kind: Job
metadata:
  name: {name}
spec:
  template:
    spec:
      containers:
      - name: training
        image: {image}
        resources:
          limits:
            nvidia.com/gpu: {gpus}
            memory: "{memory}"
            cpu: "{cpu}"
        env:
        - name: MODEL_NAME
          value: "{model_name}"
        - name: EPOCHS
          value: "{epochs}"
        volumeMounts:
        - name: data
          mountPath: /data
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: {pvc_name}
      restartPolicy: Never
  backoffLimit: 3'''
    },
    "mlflow_deployment": {
        "instruction": "Create Kubernetes Deployment for MLflow server",
        "output": '''apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mlflow
  template:
    metadata:
      labels:
        app: mlflow
    spec:
      containers:
      - name: mlflow
        image: {image}
        ports:
        - containerPort: 5000
        env:
        - name: MLFLOW_BACKEND_STORE_URI
          value: "{backend_uri}"
        - name: MLFLOW_ARTIFACT_ROOT
          value: "{artifact_root}"
        command: ["mlflow", "server", "--host", "0.0.0.0"]
---
apiVersion: v1
kind: Service
metadata:
  name: mlflow
spec:
  type: ClusterIP
  selector:
    app: mlflow
  ports:
  - port: 5000
    targetPort: 5000'''
    },
    "s3_ml_bucket": {
        "instruction": "Create Terraform for S3 bucket for ML artifacts",
        "output": '''resource "aws_s3_bucket" "{name}" {{
  bucket = "{bucket_name}"

  tags = {{
    Name    = "{name}"
    Purpose = "ML Artifacts"
  }}
}}

resource "aws_s3_bucket_versioning" "{name}_versioning" {{
  bucket = aws_s3_bucket.{name}.id
  versioning_configuration {{
    status = "Enabled"
  }}
}}

resource "aws_s3_bucket_lifecycle_configuration" "{name}_lifecycle" {{
  bucket = aws_s3_bucket.{name}.id

  rule {{
    id     = "archive-old-models"
    status = "Enabled"

    transition {{
      days          = 90
      storage_class = "GLACIER"
    }}
  }}
}}'''
    },
}

# =============================================================================
# RANDOM VALUE GENERATORS
# =============================================================================

def random_name():
    prefixes = ["app", "web", "api", "service", "backend", "frontend", "worker", "queue", "cache"]
    suffixes = ["prod", "dev", "staging", "main", "primary", "v1", "v2"]
    return f"{random.choice(prefixes)}-{random.choice(suffixes)}"

def random_app():
    return random.choice(["nginx", "redis", "postgres", "mysql", "mongodb", "api", "web", "worker"])

def random_instance_type():
    return random.choice(["t3.micro", "t3.small", "t3.medium", "t3.large", "m5.large", "m5.xlarge"])

def random_port():
    return random.choice([80, 443, 3000, 5000, 8000, 8080, 8443, 9000])

def random_replicas():
    return random.choice([1, 2, 3, 5])

def random_env():
    return random.choice(["production", "staging", "development"])

def random_cidr():
    return f"10.{random.randint(0, 255)}.0.0/16"

def random_storage():
    return random.choice(["1Gi", "5Gi", "10Gi", "20Gi", "50Gi", "100Gi"])

# =============================================================================
# SAMPLE GENERATORS
# =============================================================================

def generate_terraform_sample():
    template_name = random.choice(list(TERRAFORM_TEMPLATES.keys()))
    template = TERRAFORM_TEMPLATES[template_name]

    name = random_name().replace("-", "_")

    params = {
        "name": name,
        "env": random_env(),
        "instance_type": random_instance_type(),
        "ami_id": f"ami-{random.randint(10000000, 99999999):08x}",
        "port": random_port(),
        "cidr": random_cidr(),
        "public_cidr": f"10.0.{random.randint(1, 10)}.0/24",
        "private_cidr": f"10.0.{random.randint(11, 20)}.0/24",
        "az": random.choice(["us-east-1", "us-west-2", "eu-west-1"]),
        "bucket_name": f"my-bucket-{random.randint(1000, 9999)}",
        "feature": random.choice(["versioning", "encryption", "lifecycle rules"]),
        "engine": random.choice(["mysql", "postgres", "mariadb"]),
        "engine_version": random.choice(["8.0", "14.0", "13.0"]),
        "instance_class": random.choice(["db.t3.micro", "db.t3.small", "db.r5.large"]),
        "storage": random.choice([20, 50, 100]),
        "db_name": f"mydb_{random.randint(1, 100)}",
        "runtime": random.choice(["python3.9", "python3.10", "nodejs18.x", "go1.x"]),
        "filename": "function.zip",
        "handler": random.choice(["index.handler", "main.handler", "lambda_function.lambda_handler"]),
        "timeout": random.choice([30, 60, 120, 300]),
        "memory": random.choice([128, 256, 512, 1024]),
        "pk": random.choice(["id", "user_id", "order_id", "item_id"]),
        "service": random.choice(["ec2", "lambda", "ecs", "s3"]),
        "policy": random.choice(["AmazonEC2ReadOnlyAccess", "AmazonS3ReadOnlyAccess", "AWSLambdaBasicExecutionRole"]),
        "metric": random.choice(["CPUUtilization", "MemoryUtilization", "DiskReadOps"]),
        "namespace": random.choice(["AWS/EC2", "AWS/RDS", "AWS/Lambda"]),
        "threshold": random.choice([70, 80, 90]),
        "cpu": random.choice(["256", "512", "1024", "2048"]),
        "image": f"{random.choice(['nginx', 'python', 'node'])}:{random.choice(['latest', 'alpine', 'slim'])}",
        "replicas": random_replicas(),
        "num_public": random.choice([1, 2, 3]),
        "num_private": random.choice([1, 2, 3]),
    }

    instruction = template["instruction"].format(**params)
    output = template["output"].format(**params)

    return {
        "instruction": instruction,
        "input": "",
        "output": output,
        "category": "terraform"
    }

def generate_kubernetes_sample():
    template_name = random.choice(list(KUBERNETES_TEMPLATES.keys()))
    template = KUBERNETES_TEMPLATES[template_name]

    name = random_name()
    app = random_app()

    params = {
        "name": name,
        "app": app,
        "image": f"{app}:{random.choice(['latest', 'alpine', '1.0', '2.0'])}",
        "replicas": random_replicas(),
        "port": random_port(),
        "target_port": random_port(),
        "node_port": random.randint(30000, 32767),
        "memory_request": random.choice(["64Mi", "128Mi", "256Mi"]),
        "memory_limit": random.choice(["128Mi", "256Mi", "512Mi"]),
        "cpu_request": random.choice(["100m", "250m", "500m"]),
        "cpu_limit": random.choice(["250m", "500m", "1000m"]),
        "storage": random_storage(),
        "storage_class": random.choice(["standard", "gp2", "fast"]),
        "access_mode": random.choice(["ReadWriteOnce", "ReadOnlyMany", "ReadWriteMany"]),
        "host": f"{app}.example.com",
        "service": f"{name}-service",
        "key1": random.choice(["DATABASE_HOST", "API_URL", "LOG_LEVEL"]),
        "value1": random.choice(["localhost", "https://api.example.com", "info"]),
        "key2": random.choice(["CACHE_TTL", "MAX_CONNECTIONS", "TIMEOUT"]),
        "value2": random.choice(["3600", "100", "30"]),
        "log_level": random.choice(["debug", "info", "warn", "error"]),
        "env": random_env(),
        "schedule": random.choice(["0 0 * * *", "0 */6 * * *", "*/15 * * * *"]),
        "command": f'["{random.choice(["python", "node", "bash"])}", "{random.choice(["job.py", "task.js", "script.sh"])}"]',
        "deployment": f"{name}-deployment",
        "min_replicas": random.choice([1, 2]),
        "max_replicas": random.choice([5, 10, 20]),
        "cpu_target": random.choice([50, 70, 80]),
        "allowed_app": random.choice(["frontend", "backend", "api"]),
    }

    instruction = template["instruction"].format(**params)
    output = template["output"].format(**params)

    return {
        "instruction": instruction,
        "input": "",
        "output": output,
        "category": "kubernetes"
    }

def generate_docker_sample():
    template_name = random.choice(list(DOCKER_TEMPLATES.keys()))
    template = DOCKER_TEMPLATES[template_name]

    params = {
        "python_version": random.choice(["3.9", "3.10", "3.11", "3.12"]),
        "node_version": random.choice(["18", "20", "21"]),
        "go_version": random.choice(["1.21", "1.22"]),
        "java_version": random.choice(["17", "21"]),
        "port": random_port(),
        "framework": random.choice(["FastAPI", "Flask", "Django", "Express", "NestJS"]),
        "cmd": random.choice(["python", "gunicorn", "uvicorn"]),
        "entry": random.choice(["app.py", "main.py", "server.py", "index.js"]),
        "source_dir": random.choice(["./dist", "./build", "./public"]),
        "database": random.choice(["PostgreSQL", "MySQL", "MongoDB"]),
        "db_type": random.choice(["postgresql", "mysql", "mongodb"]),
        "db_image": random.choice(["postgres:15", "mysql:8", "mongo:6"]),
        "db_name": f"app_db_{random.randint(1, 100)}",
        "frontend": random.choice(["React", "Vue", "Angular"]),
        "backend": random.choice(["Node.js", "Python", "Go"]),
    }

    instruction = template["instruction"].format(**params)
    output = template["output"].format(**params)

    return {
        "instruction": instruction,
        "input": "",
        "output": output,
        "category": "docker"
    }

def generate_cicd_sample():
    template_name = random.choice(list(CICD_TEMPLATES.keys()))
    template = CICD_TEMPLATES[template_name]

    language_configs = {
        "Python": {
            "setup_action": "actions/setup-python@v5",
            "version_key": "python-version",
            "version": random.choice(["3.9", "3.10", "3.11"]),
            "install_cmd": "pip install -r requirements.txt",
            "test_cmd": "pytest",
            "image": f"python:{random.choice(['3.9', '3.10', '3.11'])}"
        },
        "Node.js": {
            "setup_action": "actions/setup-node@v4",
            "version_key": "node-version",
            "version": random.choice(["18", "20"]),
            "install_cmd": "npm ci",
            "test_cmd": "npm test",
            "image": f"node:{random.choice(['18', '20'])}"
        },
        "Go": {
            "setup_action": "actions/setup-go@v5",
            "version_key": "go-version",
            "version": random.choice(["1.21", "1.22"]),
            "install_cmd": "go mod download",
            "test_cmd": "go test ./...",
            "image": f"golang:{random.choice(['1.21', '1.22'])}"
        }
    }

    lang = random.choice(list(language_configs.keys()))
    config = language_configs[lang]

    params = {
        "trigger": random.choice(["push", "pull_request"]),
        "branch": random.choice(["main", "master", "develop"]),
        "language": lang,
        "registry": random.choice(["ghcr.io", "docker.io", "ecr.aws"]),
        "image": f"myapp-{random.randint(1, 100)}",
        "target": random.choice(["Kubernetes", "ECS", "Lambda"]),
        "credentials_action": random.choice(["aws-actions/configure-aws-credentials@v4", "azure/login@v1"]),
        "credentials_key": random.choice(["aws-access-key-id", "creds"]),
        "credentials_secret": random.choice(["AWS_ACCESS_KEY_ID", "AZURE_CREDENTIALS"]),
        "deploy_cmd": random.choice(["kubectl apply -f k8s/", "aws ecs update-service --cluster my-cluster"]),
        "deploy_image": random.choice(["bitnami/kubectl:latest", "amazon/aws-cli:latest"]),
        "env": random_env(),
        **config
    }

    instruction = template["instruction"].format(**params)
    output = template["output"].format(**params)

    return {
        "instruction": instruction,
        "input": "",
        "output": output,
        "category": "cicd"
    }

def generate_mlops_sample():
    template_name = random.choice(list(MLOPS_TEMPLATES.keys()))
    template = MLOPS_TEMPLATES[template_name]

    name = random_name().replace("-", "_")

    params = {
        "name": name,
        "instance_type": random.choice(["ml.t3.medium", "ml.m5.large", "ml.g4dn.xlarge"]),
        "instance_count": random.choice([1, 2]),
        "image": random.choice([
            "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.0.0-gpu-py310",
            "763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.12.0-gpu-py310"
        ]),
        "bucket": f"ml-bucket-{random.randint(1000, 9999)}",
        "bucket_name": f"ml-artifacts-{random.randint(1000, 9999)}",
        "model_path": f"models/{name}/model.tar.gz",
        "base_image": random.choice([
            "pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime",
            "tensorflow/tensorflow:2.12.0-gpu"
        ]),
        "framework": random.choice(["PyTorch", "TensorFlow", "scikit-learn"]),
        "entry": random.choice(["train.py", "main.py"]),
        "gpus": random.choice([1, 2, 4]),
        "memory": random.choice(["16Gi", "32Gi", "64Gi"]),
        "cpu": random.choice(["4", "8", "16"]),
        "model_name": random.choice(["bert-base", "gpt2", "resnet50"]),
        "epochs": random.choice(["10", "50", "100"]),
        "pvc_name": f"{name}-data-pvc",
        "backend_uri": random.choice(["postgresql://...", "mysql://...", "sqlite:///mlflow.db"]),
        "artifact_root": f"s3://ml-artifacts-{random.randint(1000, 9999)}/mlflow",
    }

    instruction = template["instruction"].format(**params)
    output = template["output"].format(**params)

    return {
        "instruction": instruction,
        "input": "",
        "output": output,
        "category": "mlops"
    }

# =============================================================================
# MAIN GENERATOR
# =============================================================================

def generate_samples(count: int, category: str = None) -> list:
    """Generate training samples."""
    generators = {
        "terraform": generate_terraform_sample,
        "kubernetes": generate_kubernetes_sample,
        "docker": generate_docker_sample,
        "cicd": generate_cicd_sample,
        "mlops": generate_mlops_sample,
    }

    samples = []
    categories = [category] if category else list(generators.keys())

    for i in range(count):
        cat = random.choice(categories)
        sample = generators[cat]()
        samples.append(sample)

        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{count} samples...")

    return samples

def main():
    parser = argparse.ArgumentParser(description="Generate IaC training data from templates")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="data/training_data_templates.json", help="Output file")
    parser.add_argument("--category", type=str, help="Specific category (terraform, kubernetes, docker, cicd, mlops)")
    parser.add_argument("--merge", type=str, help="Merge with existing file")

    args = parser.parse_args()

    print(f"Generating {args.samples} samples...")
    samples = generate_samples(args.samples, args.category)

    # Merge with existing file if specified
    if args.merge and Path(args.merge).exists():
        print(f"Merging with {args.merge}...")
        with open(args.merge) as f:
            existing = json.load(f)
        samples = existing + samples
        print(f"Total samples after merge: {len(samples)}")

    # Count by category
    categories = {}
    for s in samples:
        cat = s.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\nCategory breakdown:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"\nSaved {len(samples)} samples to {args.output}")

if __name__ == "__main__":
    main()
