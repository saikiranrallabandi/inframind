#!/usr/bin/env python3
"""
Generate code outputs for InfraMind training data using Claude.

Usage:
    python scripts/generate_outputs.py --samples 100 --output data/test_data.json
    python scripts/generate_outputs.py --samples 1500 --output data/training_data.json
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from collections import defaultdict

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


# Category-specific guidance for better output quality
CATEGORY_HINTS = {
    "terraform": "Use current Terraform syntax (1.5+). Include required providers block if needed. Use proper resource naming.",
    "kubernetes": "Use current stable APIs (apps/v1, batch/v1, networking.k8s.io/v1). Include all required fields like metadata, spec.",
    "docker": "Use multi-stage builds where appropriate. Run as non-root user. Use specific image tags, not 'latest'.",
    "cicd": "Include proper authentication setup. Use current action versions (actions/checkout@v4). Add error handling.",
    "mlops": "Include GPU resource limits where applicable. Use appropriate base images for ML frameworks.",
    "cloudformation": "Use current CloudFormation syntax. Include proper Type and Properties for each resource.",
    "pulumi": "Use proper Pulumi SDK syntax. Include necessary imports and exports.",
    "ansible": "Use proper YAML syntax for playbooks. Include become: yes where needed. Use modules not shell where possible.",
    "helm": "Use proper Helm template syntax with {{ }}. Include _helpers.tpl patterns.",
    "azure": "Use current Bicep syntax. Include proper resource dependencies.",
    "crossplane": "Use proper Crossplane CRD syntax. Include providerConfigRef.",
    "monitoring": "Use proper configuration syntax for the monitoring tool. Include necessary labels and annotations.",
    "debugging": "Provide specific commands and steps. Include expected outputs or indicators of success.",
}

GENERATION_PROMPT = """You are an expert DevOps engineer. Generate ONLY the requested code/configuration.

Task: {instruction}
{input_section}

Category guidance: {category_hint}

Requirements:
- Output ONLY the code/configuration, no explanations before or after
- Use current/stable API versions (not deprecated)
- Follow security best practices
- Make it production-ready
- Include brief inline comments only where logic is complex

Output:"""


def validate_output(output: str, category: str) -> bool:
    """Basic validation that output looks like correct IaC code."""
    if len(output) < 50:
        return False

    # Strip markdown code blocks if present
    if output.startswith("```"):
        lines = output.split("\n")
        output = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    checks = {
        "terraform": lambda o: "resource" in o or "provider" in o or "variable" in o or "output" in o,
        "kubernetes": lambda o: "apiVersion" in o or "kind:" in o,
        "docker": lambda o: "FROM" in o or "services:" in o or "version:" in o,
        "cicd": lambda o: "name:" in o or "jobs:" in o or "stages:" in o or "pipeline:" in o,
        "mlops": lambda o: "apiVersion" in o or "FROM" in o or "resource" in o or "name:" in o,
        "cloudformation": lambda o: "AWSTemplateFormatVersion" in o or "Resources:" in o or "Type:" in o,
        "pulumi": lambda o: "import" in o or "pulumi" in o.lower() or "export" in o,
        "ansible": lambda o: "hosts:" in o or "tasks:" in o or "name:" in o,
        "helm": lambda o: "{{" in o or "apiVersion" in o or "name:" in o,
        "azure": lambda o: "resource" in o or "param" in o or "targetScope" in o,
        "crossplane": lambda o: "apiVersion" in o or "kind:" in o,
        "monitoring": lambda o: any(x in o for x in ["scrape_configs", "alerting", "groups:", "datasources", "dashboard"]),
        "debugging": lambda o: len(o) > 100,  # Debugging outputs are more free-form
    }

    validator = checks.get(category, lambda o: True)
    return validator(output)


def clean_output(output: str) -> str:
    """Clean markdown code blocks from output."""
    output = output.strip()

    # Remove markdown code blocks
    if output.startswith("```"):
        lines = output.split("\n")
        # Remove first line (```language)
        lines = lines[1:]
        # Remove last line if it's just ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        output = "\n".join(lines)

    return output.strip()


def generate_output(client, instruction: str, input_text: str, category: str, model: str) -> str:
    """Generate code output using Claude."""

    input_section = f"Details: {input_text}" if input_text else ""
    category_hint = CATEGORY_HINTS.get(category, "Follow best practices for this technology.")

    prompt = GENERATION_PROMPT.format(
        instruction=instruction,
        input_section=input_section,
        category_hint=category_hint
    )

    try:
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        output = message.content[0].text.strip()
        return clean_output(output)
    except Exception as e:
        print(f"  Error: {e}")
        return ""


def generate_tasks_from_templates():
    """Generate 300+ tasks from expanded templates."""

    # Expanded Terraform templates (65 tasks)
    TERRAFORM_TASKS = [
        # EC2 variations
        ("Create Terraform for AWS EC2 instance", "t3.medium instance type"),
        ("Create Terraform for AWS EC2 instance", "t3.small with EBS volume attached"),
        ("Create Terraform for AWS EC2 Auto Scaling Group", "min 2, max 10 instances"),
        ("Create Terraform for AWS EC2 instance", "with user data script to install nginx"),
        ("Create Terraform for AWS Launch Template", "for Auto Scaling with spot instances"),
        # S3 variations
        ("Create Terraform for S3 bucket", "with versioning enabled"),
        ("Create Terraform for S3 bucket", "with lifecycle rules for cost optimization"),
        ("Create Terraform for S3 bucket", "for static website hosting"),
        ("Create Terraform for S3 bucket", "with cross-region replication"),
        ("Create Terraform for S3 bucket", "with server-side encryption using KMS"),
        # VPC variations
        ("Create Terraform for VPC", "CIDR 10.0.0.0/16 with 2 public subnets"),
        ("Create Terraform for VPC", "with public and private subnets and NAT gateway"),
        ("Create Terraform for VPC peering", "between two VPCs in same region"),
        ("Create Terraform for VPC endpoint", "for S3 gateway endpoint"),
        ("Create Terraform for Transit Gateway", "connecting 3 VPCs"),
        # Database variations
        ("Create Terraform for RDS PostgreSQL", "db.t3.micro with Multi-AZ"),
        ("Create Terraform for RDS MySQL", "with read replica in different AZ"),
        ("Create Terraform for Aurora PostgreSQL", "serverless v2 cluster"),
        ("Create Terraform for ElastiCache Redis", "cluster mode enabled with 3 shards"),
        ("Create Terraform for DynamoDB table", "partition key: user_id, sort key: timestamp"),
        ("Create Terraform for DynamoDB table", "with global secondary index"),
        ("Create Terraform for DocumentDB cluster", "with 3 instances"),
        # Serverless variations
        ("Create Terraform for Lambda function", "Python 3.11 runtime"),
        ("Create Terraform for Lambda function", "Node.js 20 with API Gateway trigger"),
        ("Create Terraform for Lambda function", "with VPC access and secrets manager"),
        ("Create Terraform for Lambda function", "with SQS event source mapping"),
        ("Create Terraform for Step Functions", "state machine for order processing"),
        ("Create Terraform for EventBridge rule", "trigger Lambda on schedule"),
        # IAM variations
        ("Create Terraform for IAM role", "for EC2 with S3 read access"),
        ("Create Terraform for IAM role", "for Lambda with DynamoDB full access"),
        ("Create Terraform for IAM policy", "least privilege for CI/CD deployment"),
        ("Create Terraform for IAM user", "with programmatic access only"),
        ("Create Terraform for IAM role", "for cross-account S3 access"),
        # Networking variations
        ("Create Terraform for Application Load Balancer", "with HTTPS listener"),
        ("Create Terraform for Network Load Balancer", "for TCP traffic on port 443"),
        ("Create Terraform for CloudFront distribution", "for S3 static website"),
        ("Create Terraform for Security Group", "allow HTTP/HTTPS inbound"),
        ("Create Terraform for Security Group", "for RDS allowing app tier access"),
        ("Create Terraform for WAF", "protect ALB from SQL injection and XSS"),
        ("Create Terraform for Route53 record", "A record pointing to ALB"),
        ("Create Terraform for Route53 health check", "with failover routing"),
        # Kubernetes on AWS
        ("Create Terraform for EKS cluster", "with 2 managed node groups"),
        ("Create Terraform for EKS cluster", "with Fargate profiles"),
        ("Create Terraform for EKS add-ons", "CoreDNS, kube-proxy, vpc-cni, ebs-csi"),
        # Monitoring & Logs
        ("Create Terraform for CloudWatch alarm", "CPU > 80% on EC2"),
        ("Create Terraform for CloudWatch Log Group", "with 30 day retention"),
        ("Create Terraform for CloudWatch dashboard", "for EC2 and RDS metrics"),
        ("Create Terraform for CloudWatch metric filter", "count ERROR in logs"),
        # Messaging
        ("Create Terraform for SNS topic", "with email subscription"),
        ("Create Terraform for SQS queue", "with dead letter queue"),
        ("Create Terraform for SQS queue", "FIFO queue with deduplication"),
        ("Create Terraform for EventBridge rule", "capture S3 object created events"),
        # API & Integration
        ("Create Terraform for API Gateway", "REST API with Lambda integration"),
        ("Create Terraform for API Gateway", "HTTP API with JWT authorizer"),
        ("Create Terraform for API Gateway", "WebSocket API for real-time"),
        # Security
        ("Create Terraform for KMS key", "for S3 bucket encryption"),
        ("Create Terraform for Secrets Manager secret", "for database credentials"),
        ("Create Terraform for ACM certificate", "with DNS validation for domain"),
        # Data & Analytics
        ("Create Terraform for Kinesis Data Stream", "for real-time data ingestion"),
        ("Create Terraform for Glue Crawler", "for S3 data catalog"),
        ("Create Terraform for Athena workgroup", "with query result location"),
        # Containers
        ("Create Terraform for ECR repository", "with lifecycle policy"),
        ("Create Terraform for ECS Fargate service", "with ALB integration"),
        ("Create Terraform for ECS task definition", "with container definitions"),
    ]

    # Expanded Kubernetes templates (65 tasks)
    K8S_TASKS = [
        # Deployments
        ("Create Kubernetes Deployment", "nginx with 3 replicas"),
        ("Create Kubernetes Deployment", "for Node.js app with environment variables"),
        ("Create Kubernetes Deployment", "with resource limits and requests"),
        ("Create Kubernetes Deployment", "with liveness and readiness probes"),
        ("Create Kubernetes Deployment", "with init container for setup"),
        ("Create Kubernetes Deployment", "with sidecar container for logging"),
        ("Create Kubernetes Deployment", "using rolling update strategy"),
        ("Create Kubernetes Deployment", "with pod anti-affinity for HA"),
        # Services
        ("Create Kubernetes Service", "LoadBalancer type for nginx on port 80"),
        ("Create Kubernetes Service", "ClusterIP for internal microservice"),
        ("Create Kubernetes Service", "NodePort for development access"),
        ("Create Kubernetes Service", "headless Service for StatefulSet"),
        ("Create Kubernetes Service", "ExternalName for external database"),
        # Config & Secrets
        ("Create Kubernetes ConfigMap", "with database connection settings"),
        ("Create Kubernetes ConfigMap", "from file with nginx.conf"),
        ("Create Kubernetes Secret", "for database password"),
        ("Create Kubernetes Secret", "TLS secret for Ingress"),
        ("Create Kubernetes Secret", "docker-registry type for private registry"),
        # Ingress
        ("Create Kubernetes Ingress", "with TLS termination"),
        ("Create Kubernetes Ingress", "with path-based routing to multiple services"),
        ("Create Kubernetes Ingress", "with nginx annotations for rate limiting"),
        ("Create Kubernetes Ingress", "with AWS ALB ingress controller"),
        # Storage
        ("Create Kubernetes PersistentVolumeClaim", "10Gi storage"),
        ("Create Kubernetes PersistentVolumeClaim", "with ReadWriteMany access mode"),
        ("Create Kubernetes PersistentVolume", "hostPath for development"),
        ("Create Kubernetes StorageClass", "for AWS EBS gp3 volumes"),
        # Autoscaling
        ("Create Kubernetes HorizontalPodAutoscaler", "scale on CPU 70%"),
        ("Create Kubernetes HorizontalPodAutoscaler", "scale on memory usage"),
        ("Create Kubernetes HorizontalPodAutoscaler", "with custom metrics"),
        ("Create Kubernetes VerticalPodAutoscaler", "for resource recommendations"),
        # Jobs & CronJobs
        ("Create Kubernetes CronJob", "run daily at midnight"),
        ("Create Kubernetes CronJob", "weekly database backup"),
        ("Create Kubernetes Job", "batch processing with 3 completions"),
        ("Create Kubernetes Job", "with parallelism for data processing"),
        # StatefulSets & DaemonSets
        ("Create Kubernetes StatefulSet", "for Redis with 3 replicas"),
        ("Create Kubernetes StatefulSet", "for PostgreSQL with persistent storage"),
        ("Create Kubernetes StatefulSet", "for Elasticsearch cluster"),
        ("Create Kubernetes DaemonSet", "for log collection"),
        ("Create Kubernetes DaemonSet", "for node monitoring agent"),
        ("Create Kubernetes DaemonSet", "for GPU driver installation"),
        # RBAC
        ("Create Kubernetes ServiceAccount", "with RBAC for pods"),
        ("Create Kubernetes Role", "read-only access to pods and services"),
        ("Create Kubernetes RoleBinding", "bind role to service account"),
        ("Create Kubernetes ClusterRole", "for cluster-wide read access"),
        ("Create Kubernetes ClusterRoleBinding", "for monitoring namespace"),
        # Network Policies
        ("Create Kubernetes NetworkPolicy", "allow only frontend to backend"),
        ("Create Kubernetes NetworkPolicy", "deny all ingress by default"),
        ("Create Kubernetes NetworkPolicy", "allow traffic from specific namespace"),
        # Resource Management
        ("Create Kubernetes ResourceQuota", "limit CPU and memory per namespace"),
        ("Create Kubernetes LimitRange", "default container resource limits"),
        ("Create Kubernetes PodDisruptionBudget", "minAvailable 2"),
        ("Create Kubernetes PriorityClass", "high priority for critical workloads"),
        # Operators & CRDs
        ("Create Kubernetes CRD", "for custom application resource"),
        # Pod Configuration
        ("Create Kubernetes Pod", "with multiple containers sharing volume"),
        ("Create Kubernetes Pod", "with security context for non-root"),
        ("Create Kubernetes Pod", "with node selector for specific nodes"),
        ("Create Kubernetes Pod", "with tolerations for tainted nodes"),
        # Service Mesh
        ("Create Kubernetes VirtualService", "Istio routing rules"),
        ("Create Kubernetes DestinationRule", "Istio traffic policy"),
        # Monitoring
        ("Create Kubernetes ServiceMonitor", "for Prometheus scraping"),
        ("Create Kubernetes PrometheusRule", "alerting rules for high CPU"),
        # Namespace
        ("Create Kubernetes Namespace", "with resource quotas and limits"),
    ]

    # Expanded Docker templates (50 tasks)
    DOCKER_TASKS = [
        # Python Dockerfiles
        ("Create Dockerfile for Python application", "with FastAPI and uvicorn"),
        ("Create Dockerfile for Python application", "with Django and gunicorn"),
        ("Create Dockerfile for Flask application", "with Gunicorn"),
        ("Create Dockerfile for Python CLI tool", "with pip install"),
        ("Create multi-stage Dockerfile for Python", "with poetry dependency management"),
        # Node.js Dockerfiles
        ("Create Dockerfile for Node.js application", "with npm build step"),
        ("Create Dockerfile for Node.js application", "with yarn and production build"),
        ("Create Dockerfile for React application", "with nginx for serving"),
        ("Create Dockerfile for Next.js application", "with standalone output"),
        ("Create Dockerfile for Express API", "with TypeScript compilation"),
        # Go Dockerfiles
        ("Create Dockerfile for Go application", "multi-stage build"),
        ("Create Dockerfile for Go application", "with scratch base image"),
        ("Create multi-stage Dockerfile for Go", "with CGO disabled"),
        # Java Dockerfiles
        ("Create Dockerfile for Java application", "with Maven build"),
        ("Create Dockerfile for Java application", "with Gradle build"),
        ("Create Dockerfile for Spring Boot application", "with layered jar"),
        # Other languages
        ("Create multi-stage Dockerfile", "for Rust application"),
        ("Create Dockerfile for .NET application", "with dotnet build"),
        ("Create Dockerfile for Ruby on Rails", "with asset precompilation"),
        ("Create Dockerfile for PHP Laravel", "with composer install"),
        # Security-focused
        ("Create Dockerfile", "with non-root user for security"),
        ("Create Dockerfile", "optimized for small image size using alpine"),
        ("Create Dockerfile", "using distroless base image"),
        ("Create secure Dockerfile", "with COPY --chown and USER directive"),
        # ML Dockerfiles
        ("Create Dockerfile for Python ML application", "with PyTorch"),
        ("Create Dockerfile for ML training", "with TensorFlow and GPU support"),
        ("Create Dockerfile for ML inference", "with ONNX runtime"),
        # docker-compose files
        ("Create docker-compose.yml", "for web app with PostgreSQL and Redis"),
        ("Create docker-compose.yml", "for MEAN stack application"),
        ("Create docker-compose.yml", "with healthchecks and restart policies"),
        ("Create docker-compose.yml", "for microservices with API gateway"),
        ("Create docker-compose.yml", "for development with hot reload"),
        ("Create docker-compose.yml", "for WordPress with MySQL"),
        ("Create docker-compose.yml", "for ELK stack logging"),
        ("Create docker-compose.yml", "for Prometheus and Grafana monitoring"),
        ("Create docker-compose.yml", "for RabbitMQ with management UI"),
        ("Create docker-compose.yml", "for Kafka with Zookeeper"),
        ("Create docker-compose.yml", "for MinIO object storage"),
        ("Create docker-compose.yml", "for Keycloak identity server"),
        # Multi-stage builds
        ("Create multi-stage Dockerfile", "with build and runtime stages"),
        ("Create multi-stage Dockerfile", "with test stage for CI"),
        # Specific configurations
        ("Create Dockerfile", "with HEALTHCHECK instruction"),
        ("Create Dockerfile", "with ARG for build-time variables"),
        ("Create Dockerfile", "with ENTRYPOINT and CMD combination"),
        ("Create Dockerfile", "optimized for Docker layer caching"),
        # Database containers
        ("Create docker-compose.yml", "for PostgreSQL with initialization scripts"),
        ("Create docker-compose.yml", "for MongoDB replica set"),
        ("Create docker-compose.yml", "for Redis Sentinel setup"),
        ("Create Dockerfile", "for custom PostgreSQL with extensions"),
    ]

    # Expanded CI/CD templates (60 tasks)
    CICD_TASKS = [
        # GitHub Actions - Testing
        ("Create GitHub Actions workflow", "run tests on push to main"),
        ("Create GitHub Actions workflow", "run on pull request with code review"),
        ("Create GitHub Actions workflow", "with matrix testing for multiple Python versions"),
        ("Create GitHub Actions workflow", "with matrix testing for Node.js versions"),
        ("Create GitHub Actions workflow", "for running integration tests"),
        ("Create GitHub Actions workflow", "with code coverage upload to Codecov"),
        # GitHub Actions - Docker
        ("Create GitHub Actions workflow", "build and push Docker image to ECR"),
        ("Create GitHub Actions workflow", "build and push to Docker Hub"),
        ("Create GitHub Actions workflow", "build and push to GitHub Container Registry"),
        ("Create GitHub Actions workflow", "with Docker layer caching"),
        ("Create GitHub Actions workflow", "multi-platform Docker build for arm64 and amd64"),
        # GitHub Actions - Deployment
        ("Create GitHub Actions workflow", "deploy to AWS ECS"),
        ("Create GitHub Actions workflow", "deploy to Kubernetes cluster"),
        ("Create GitHub Actions workflow", "deploy to AWS Lambda"),
        ("Create GitHub Actions workflow", "deploy to Vercel"),
        ("Create GitHub Actions workflow", "deploy to AWS S3 and CloudFront"),
        ("Create GitHub Actions workflow", "blue-green deployment to ECS"),
        # GitHub Actions - Infrastructure
        ("Create GitHub Actions workflow", "for Terraform plan and apply"),
        ("Create GitHub Actions workflow", "for Terraform with environment approval"),
        ("Create GitHub Actions workflow", "for Pulumi infrastructure deployment"),
        ("Create GitHub Actions workflow", "for AWS CDK deployment"),
        # GitHub Actions - Release
        ("Create GitHub Actions workflow", "for Python package release to PyPI"),
        ("Create GitHub Actions workflow", "for npm package publish"),
        ("Create GitHub Actions workflow", "for semantic versioning release"),
        ("Create GitHub Actions workflow", "create GitHub release with changelog"),
        ("Create GitHub Actions workflow", "for Go module release"),
        # GitHub Actions - Security
        ("Create GitHub Actions workflow", "with CodeQL security scanning"),
        ("Create GitHub Actions workflow", "with Snyk vulnerability scanning"),
        ("Create GitHub Actions workflow", "with Trivy container scanning"),
        ("Create GitHub Actions workflow", "with SAST using Semgrep"),
        ("Create GitHub Actions workflow", "for secret scanning"),
        # GitHub Actions - Notifications
        ("Create GitHub Actions workflow", "with Slack notification on failure"),
        ("Create GitHub Actions workflow", "with Teams notification"),
        ("Create GitHub Actions workflow", "comment PR with test results"),
        # GitHub Actions - Scheduled
        ("Create GitHub Actions workflow", "scheduled daily backup job"),
        ("Create GitHub Actions workflow", "weekly dependency update check"),
        ("Create GitHub Actions workflow", "nightly build and test"),
        # GitLab CI
        ("Create GitLab CI pipeline", "with test, build, deploy stages"),
        ("Create GitLab CI pipeline", "with security scanning"),
        ("Create GitLab CI pipeline", "with Docker layer caching"),
        ("Create GitLab CI pipeline", "with parallel test jobs"),
        ("Create GitLab CI pipeline", "with manual approval for production"),
        ("Create GitLab CI pipeline", "with environment-specific deployments"),
        ("Create GitLab CI pipeline", "for Kubernetes deployment with review apps"),
        ("Create GitLab CI pipeline", "with DAST security testing"),
        # Jenkins
        ("Create Jenkins pipeline", "declarative pipeline with parallel stages"),
        ("Create Jenkins pipeline", "with Docker agent"),
        ("Create Jenkins pipeline", "for Kubernetes deployment"),
        ("Create Jenkins pipeline", "with approval gates"),
        ("Create Jenkins shared library", "for common CI functions"),
        # Other CI/CD
        ("Create CircleCI config", "for Node.js application"),
        ("Create Azure DevOps pipeline", "for .NET application"),
        ("Create ArgoCD Application", "for GitOps deployment"),
        ("Create Flux HelmRelease", "for GitOps with Helm"),
        # Linting & Formatting
        ("Create GitHub Actions workflow", "for Python linting with ruff"),
        ("Create GitHub Actions workflow", "for ESLint and Prettier"),
        ("Create GitHub Actions workflow", "for Go linting with golangci-lint"),
    ]

    # Expanded MLOps templates (60 tasks)
    MLOPS_TASKS = [
        # Kubernetes for ML
        ("Create Kubernetes Deployment for ML model serving", "with GPU support"),
        ("Create Kubernetes Deployment", "for MLflow tracking server"),
        ("Create Kubernetes Deployment", "for Ray cluster head node"),
        ("Create Kubernetes Deployment", "for Ray worker nodes with GPU"),
        ("Create Kubernetes manifest for TensorFlow Serving", "with autoscaling"),
        ("Create Kubernetes manifest for TorchServe", "with GPU and multiple models"),
        ("Create Kubernetes manifest for Triton Inference Server", "multi-model serving"),
        ("Create Kubernetes manifest for vLLM", "serving Llama 2 model"),
        ("Create Kubernetes manifest for Text Generation Inference", "serving Mistral"),
        ("Create Kubernetes Job", "for batch ML inference"),
        ("Create Kubernetes Job", "for distributed training with PyTorch"),
        ("Create Kubernetes CronJob", "for model retraining schedule"),
        # Docker for ML
        ("Create Dockerfile for ML training", "with CUDA and PyTorch"),
        ("Create Dockerfile for ML training", "with TensorFlow and GPU"),
        ("Create Dockerfile for ML inference", "lightweight with ONNX runtime"),
        ("Create Dockerfile for Jupyter notebook", "with data science libraries"),
        ("Create docker-compose.yml", "for Jupyter notebook with GPU"),
        ("Create docker-compose for MLflow", "with PostgreSQL backend"),
        ("Create docker-compose.yml", "for Label Studio annotation tool"),
        ("Create docker-compose.yml", "for Weights & Biases server"),
        # Terraform for ML infrastructure
        ("Create Terraform for SageMaker endpoint", "for model inference"),
        ("Create Terraform for SageMaker training job", "with spot instances"),
        ("Create Terraform for SageMaker notebook instance", "with lifecycle config"),
        ("Create Terraform for S3 bucket", "for ML model artifacts with versioning"),
        ("Create Terraform for EC2 GPU instance", "p3.2xlarge for training"),
        ("Create Terraform for Bedrock", "Claude model access"),
        ("Create Terraform for Lambda", "for SageMaker endpoint invocation"),
        # GitHub Actions for ML
        ("Create GitHub Actions workflow", "for ML model training pipeline"),
        ("Create GitHub Actions workflow", "for model evaluation and comparison"),
        ("Create GitHub Actions workflow", "for model registration to registry"),
        ("Create GitHub Actions workflow", "for automated model testing"),
        ("Create GitHub Actions workflow", "for data validation pipeline"),
        ("Create GitHub Actions workflow", "train on GPU runner"),
        # Airflow DAGs
        ("Create Airflow DAG", "for ML training pipeline"),
        ("Create Airflow DAG", "for data preprocessing workflow"),
        ("Create Airflow DAG", "for model deployment pipeline"),
        ("Create Airflow DAG", "for feature engineering"),
        # Kubeflow
        ("Create Kubeflow Pipeline", "for end-to-end ML workflow"),
        ("Create Kubeflow Pipeline", "for hyperparameter tuning"),
        ("Create Kubeflow Pipeline", "for model serving deployment"),
        # Feature Stores
        ("Create Feast feature store config", "for online and offline features"),
        ("Create Terraform for Feature Store", "on AWS with DynamoDB and S3"),
        # Model Serving
        ("Create Kubernetes manifest for KServe", "InferenceService with transformer"),
        ("Create Kubernetes manifest for Seldon", "multi-model deployment"),
        ("Create BentoML service", "for model packaging and serving"),
        # Vector Databases
        ("Create Kubernetes manifest for Milvus", "vector database cluster"),
        ("Create docker-compose.yml", "for Qdrant vector database"),
        ("Create Kubernetes manifest for Weaviate", "with GPU support"),
        ("Create docker-compose.yml", "for Chroma vector database"),
        # LLM Infrastructure
        ("Create Kubernetes manifest for Ollama", "local LLM serving"),
        ("Create docker-compose.yml", "for LocalAI with models"),
        ("Create Kubernetes manifest", "for LangServe deployment"),
        # Monitoring for ML
        ("Create Kubernetes manifest", "for Prometheus ML metrics"),
        ("Create Grafana dashboard", "for model performance monitoring"),
        ("Create Kubernetes manifest", "for Evidently AI monitoring"),
        # Data Pipeline
        ("Create Kubernetes manifest", "for Apache Spark on K8s"),
        ("Create docker-compose.yml", "for Prefect workflow orchestration"),
        ("Create Kubernetes manifest", "for Dagster data pipeline"),
        # Experiment Tracking
        ("Create docker-compose.yml", "for Neptune.ai experiment tracking"),
        ("Create Kubernetes manifest", "for ClearML server"),
    ]

    tasks = []
    for instruction, input_text in TERRAFORM_TASKS:
        tasks.append({"instruction": instruction, "input": input_text, "category": "terraform"})
    for instruction, input_text in K8S_TASKS:
        tasks.append({"instruction": instruction, "input": input_text, "category": "kubernetes"})
    for instruction, input_text in DOCKER_TASKS:
        tasks.append({"instruction": instruction, "input": input_text, "category": "docker"})
    for instruction, input_text in CICD_TASKS:
        tasks.append({"instruction": instruction, "input": input_text, "category": "cicd"})
    for instruction, input_text in MLOPS_TASKS:
        tasks.append({"instruction": instruction, "input": input_text, "category": "mlops"})

    # Add IDs
    for i, task in enumerate(tasks):
        task["id"] = f"{task['category']}-{i:03d}"

    return tasks


def main():
    parser = argparse.ArgumentParser(description="Generate training data outputs")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--output", default="data/training_data.json", help="Output file path")
    parser.add_argument("--batch-size", type=int, default=25, help="Save checkpoint every N samples")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Claude model to use")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint file")
    parser.add_argument("--categories", nargs="+", help="Only generate for specific categories")
    parser.add_argument("--retry-failed", action="store_true", help="Retry failed validations")
    args = parser.parse_args()

    if not HAS_ANTHROPIC:
        print("Error: anthropic package not installed. Run: pip install anthropic")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        return

    client = anthropic.Anthropic(api_key=api_key)

    # Generate tasks
    tasks = generate_tasks_from_templates()

    if args.categories:
        tasks = [t for t in tasks if t["category"] in args.categories]
        print(f"Filtered to categories: {args.categories}")

    # Sample tasks
    random.shuffle(tasks)
    tasks = tasks[:args.samples]

    # Cost estimate
    # ~800 tokens per sample (input + output), Sonnet: $3/1M input, $15/1M output
    estimated_cost = len(tasks) * 0.012

    print(f"=" * 60)
    print(f"InfraMind Training Data Generator")
    print(f"=" * 60)
    print(f"Model: {args.model}")
    print(f"Samples: {len(tasks)}")
    print(f"Output: {args.output}")
    print(f"Estimated cost: ${estimated_cost:.2f}")

    # Category breakdown
    cat_counts = defaultdict(int)
    for t in tasks:
        cat_counts[t["category"]] += 1
    print(f"\nCategory breakdown:")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count}")

    # Load checkpoint if resuming
    completed_tasks = []
    if args.resume and Path(args.resume).exists():
        with open(args.resume) as f:
            checkpoint = json.load(f)
            completed_tasks = checkpoint.get("tasks", [])
            completed_ids = {t["id"] for t in completed_tasks}
            tasks = [t for t in tasks if t["id"] not in completed_ids]
            print(f"\nResuming from checkpoint: {len(completed_tasks)} completed, {len(tasks)} remaining")

    # Generate outputs
    print(f"\n[Generating outputs...]")
    generated = completed_tasks.copy()
    failed = []

    for i, task in enumerate(tasks):
        print(f"  [{i+1}/{len(tasks)}] {task['category']}: {task['instruction'][:50]}...", end=" ")

        output = generate_output(
            client,
            task["instruction"],
            task.get("input", ""),
            task["category"],
            args.model
        )

        if output and validate_output(output, task["category"]):
            task_with_output = {
                "id": task["id"],
                "instruction": task["instruction"],
                "input": task.get("input", ""),
                "output": output,
                "category": task["category"]
            }
            generated.append(task_with_output)
            print(f"OK ({len(output)} chars)")
        else:
            failed.append(task)
            if output:
                print(f"VALIDATION FAILED ({len(output)} chars)")
            else:
                print("FAILED")

        # Save checkpoint
        if (i + 1) % args.batch_size == 0:
            checkpoint_path = args.output.replace(".json", "_checkpoint.json")
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path, "w") as f:
                json.dump({"tasks": generated, "completed": len(generated)}, f, indent=2)
            print(f"  [Checkpoint saved: {len(generated)} tasks]")

        # Rate limiting
        time.sleep(0.3)

    # Retry failed tasks if requested
    if args.retry_failed and failed:
        print(f"\n[Retrying {len(failed)} failed tasks...]")
        retry_success = 0
        for i, task in enumerate(failed[:]):  # Copy list to avoid mutation issues
            print(f"  [Retry {i+1}/{len(failed)}] {task['category']}: {task['instruction'][:40]}...", end=" ")
            output = generate_output(
                client,
                task["instruction"],
                task.get("input", ""),
                task["category"],
                args.model
            )
            if output and validate_output(output, task["category"]):
                task_with_output = {
                    "id": task["id"],
                    "instruction": task["instruction"],
                    "input": task.get("input", ""),
                    "output": output,
                    "category": task["category"]
                }
                generated.append(task_with_output)
                failed.remove(task)
                retry_success += 1
                print(f"OK ({len(output)} chars)")
            else:
                print("FAILED again")
            time.sleep(0.3)
        print(f"  [Retry recovered {retry_success} tasks]")

    # Save final output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to Alpaca format (keep category for analysis)
    alpaca_data = [
        {
            "instruction": t["instruction"],
            "input": t.get("input", ""),
            "output": t["output"],
            "category": t["category"]
        }
        for t in generated
    ]

    with open(output_path, "w") as f:
        json.dump(alpaca_data, f, indent=2)

    print(f"\n" + "=" * 60)
    print(f"COMPLETE")
    print(f"=" * 60)
    print(f"Generated: {len(generated)} tasks")
    print(f"Failed: {len(failed)}")
    print(f"Saved to: {args.output}")

    # Show sample outputs
    if generated:
        print(f"\n[Sample outputs:]")
        for category in ["terraform", "kubernetes", "docker"]:
            samples = [t for t in generated if t["category"] == category]
            if samples:
                sample = random.choice(samples)
                print(f"\n--- {category.upper()} ---")
                print(f"Instruction: {sample['instruction']}")
                print(f"Input: {sample.get('input', 'N/A')}")
                print(f"Output preview:\n{sample['output'][:300]}...")


if __name__ == "__main__":
    main()
