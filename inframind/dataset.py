"""InfraMind-Bench Dataset - 500+ IaC tasks in Alpaca format"""
import json
from typing import List, Dict, Optional

# ==================== TASK TEMPLATES ====================
# Generate 500+ tasks programmatically from templates

TERRAFORM_TEMPLATES = [
    # AWS Compute
    ("Create Terraform for AWS EC2 instance", "{type} instance type"),
    ("Create Terraform for Auto Scaling Group", "min: {min}, max: {max}"),
    ("Create Terraform for EKS cluster", "{zones} availability zones"),
    ("Create Terraform for ECS Fargate service", "with {lb} load balancer"),
    ("Create Terraform for Lambda function", "{runtime} runtime"),
    # AWS Storage
    ("Create Terraform for S3 bucket", "{features}"),
    ("Create Terraform for EBS volume", "{size}GB, {type} type"),
    ("Create Terraform for EFS file system", "{mount_targets} mount targets"),
    # AWS Networking
    ("Create Terraform for VPC", "CIDR: {cidr}"),
    ("Create Terraform for VPC with subnets", "{public} public, {private} private subnets"),
    ("Create Terraform for Application Load Balancer", "{listener} listener"),
    ("Create Terraform for Network Load Balancer", ""),
    ("Create Terraform for security group", "allow {ports}"),
    ("Create Terraform for NAT Gateway", ""),
    ("Create Terraform for Internet Gateway", ""),
    ("Create Terraform for Route53 hosted zone", ""),
    ("Create Terraform for CloudFront distribution", ""),
    # AWS Database
    ("Create Terraform for RDS {engine}", "{features}"),
    ("Create Terraform for DynamoDB table", "partition key: {pk}, sort key: {sk}"),
    ("Create Terraform for ElastiCache {engine}", "{nodes} nodes"),
    ("Create Terraform for Aurora cluster", "{replicas} read replicas"),
    # AWS Serverless
    ("Create Terraform for API Gateway", "{type} API"),
    ("Create Terraform for SQS queue", "{features}"),
    ("Create Terraform for SNS topic", "with {subscription} subscription"),
    ("Create Terraform for Step Functions", ""),
    ("Create Terraform for EventBridge rule", ""),
    # AWS IAM
    ("Create Terraform for IAM role", "{access} access"),
    ("Create Terraform for IAM policy", "{permissions}"),
    ("Create Terraform for IAM user", ""),
    # AWS Monitoring
    ("Create Terraform for CloudWatch alarm", "{metric} > {threshold}"),
    ("Create Terraform for CloudWatch log group", "{retention} day retention"),
    ("Create Terraform for CloudWatch dashboard", ""),
    # AWS Security
    ("Create Terraform for KMS key", "{rotation}"),
    ("Create Terraform for WAF web ACL", "{rule}"),
    ("Create Terraform for Secrets Manager secret", ""),
    # GCP
    ("Create Terraform for GCE instance", "{machine_type}"),
    ("Create Terraform for GCS bucket", "{features}"),
    ("Create Terraform for GKE cluster", "{mode} mode"),
    ("Create Terraform for Cloud SQL {engine}", ""),
    ("Create Terraform for Cloud Function", "{runtime}"),
    # Azure
    ("Create Terraform for Azure VM", "{os}"),
    ("Create Terraform for Azure Storage Account", ""),
    ("Create Terraform for AKS cluster", "{node_pools} node pools"),
    ("Create Terraform for Azure SQL Database", ""),
    ("Create Terraform for Azure Function", ""),
]

K8S_TEMPLATES = [
    ("Create Kubernetes Deployment", "{image}, {replicas} replicas"),
    ("Create Kubernetes Deployment with resources", "CPU: {cpu}, Memory: {memory}"),
    ("Create Kubernetes Deployment with probes", "{probe_type} probe on port {port}"),
    ("Create Kubernetes Deployment with init container", ""),
    ("Create Kubernetes Deployment with sidecar", "{sidecar_type} sidecar"),
    ("Create Kubernetes Deployment with affinity", "{affinity_type} affinity"),
    ("Create Kubernetes Service", "{type} type, port {port}"),
    ("Create Kubernetes Ingress", "{features}"),
    ("Create Kubernetes ConfigMap", ""),
    ("Create Kubernetes Secret", "{type} type"),
    ("Create Kubernetes PersistentVolumeClaim", "{size} storage"),
    ("Create Kubernetes StatefulSet", "{app}, {replicas} replicas"),
    ("Create Kubernetes DaemonSet", "{purpose}"),
    ("Create Kubernetes CronJob", "schedule: {schedule}"),
    ("Create Kubernetes Job", "{completions} completions"),
    ("Create Kubernetes HorizontalPodAutoscaler", "target {metric} {target}%"),
    ("Create Kubernetes NetworkPolicy", "{policy_type}"),
    ("Create Kubernetes Role", "{permissions}"),
    ("Create Kubernetes RoleBinding", ""),
    ("Create Kubernetes ServiceAccount", ""),
    ("Create Kubernetes PodDisruptionBudget", "minAvailable: {min}"),
    ("Create Kubernetes ResourceQuota", ""),
    ("Create Kubernetes LimitRange", ""),
]

DOCKER_TEMPLATES = [
    ("Create Dockerfile for {lang} application", "{features}"),
    ("Create multi-stage Dockerfile for {lang}", "minimal final image"),
    ("Create Dockerfile with {base} base image", ""),
    ("Create secure Dockerfile", "non-root user, {base} base"),
    ("Create docker-compose.yml", "{services}"),
    ("Create docker-compose.yml with {db}", ""),
    ("Create docker-compose.yml for {stack}", ""),
]

CICD_TEMPLATES = [
    ("Create GitHub Actions workflow for {action}", "{trigger}"),
    ("Create GitHub Actions for Docker build", "push to {registry}"),
    ("Create GitHub Actions for Terraform", "{tf_action}"),
    ("Create GitHub Actions for Kubernetes", "deploy with {tool}"),
    ("Create GitHub Actions with {notification}", ""),
    ("Create GitHub Actions for {test_type} tests", ""),
    ("Create GitHub Actions with matrix strategy", "{matrix}"),
    ("Create GitLab CI pipeline", "{stages}"),
    ("Create GitLab CI for Docker", ""),
    ("Create GitLab CI with {feature}", ""),
    ("Create Jenkins pipeline", "{type}"),
    ("Create Jenkins with parallel stages", ""),
]

# AWS CloudFormation Templates
CLOUDFORMATION_TEMPLATES = [
    # Compute
    ("Create CloudFormation for EC2 instance", "{type} instance type"),
    ("Create CloudFormation for Auto Scaling Group", "min: {min}, max: {max}"),
    ("Create CloudFormation for ECS cluster", "with {lb} load balancer"),
    ("Create CloudFormation for Lambda function", "{runtime} runtime"),
    ("Create CloudFormation for EKS cluster", "{zones} availability zones"),
    # Storage
    ("Create CloudFormation for S3 bucket", "{features}"),
    ("Create CloudFormation for EBS volume", "{size}GB"),
    ("Create CloudFormation for EFS file system", ""),
    # Networking
    ("Create CloudFormation for VPC", "CIDR: {cidr}"),
    ("Create CloudFormation for VPC with subnets", "{public} public, {private} private subnets"),
    ("Create CloudFormation for Application Load Balancer", "{listener} listener"),
    ("Create CloudFormation for security group", "allow {ports}"),
    ("Create CloudFormation for API Gateway", "{api_type} API"),
    # Database
    ("Create CloudFormation for RDS {engine}", "{features}"),
    ("Create CloudFormation for DynamoDB table", "partition key: {pk}"),
    ("Create CloudFormation for Aurora cluster", "{replicas} read replicas"),
    ("Create CloudFormation for ElastiCache {engine}", "{nodes} nodes"),
    # IAM & Security
    ("Create CloudFormation for IAM role", "{access} access"),
    ("Create CloudFormation for IAM policy", "{permissions}"),
    ("Create CloudFormation for KMS key", "{rotation}"),
    ("Create CloudFormation for Secrets Manager secret", ""),
    # Serverless
    ("Create CloudFormation SAM template", "Lambda + API Gateway"),
    ("Create CloudFormation for Step Functions", ""),
    ("Create CloudFormation for SQS queue", "{features}"),
    ("Create CloudFormation for SNS topic", ""),
    ("Create CloudFormation for EventBridge rule", ""),
]

# Pulumi Templates (TypeScript/Python)
PULUMI_TEMPLATES = [
    # AWS
    ("Create Pulumi TypeScript for AWS EC2", "{type} instance"),
    ("Create Pulumi Python for AWS EC2", "{type} instance"),
    ("Create Pulumi for AWS S3 bucket", "{features}"),
    ("Create Pulumi for AWS VPC", "CIDR: {cidr}"),
    ("Create Pulumi for AWS Lambda", "{runtime} runtime"),
    ("Create Pulumi for AWS EKS cluster", "{zones} availability zones"),
    ("Create Pulumi for AWS RDS {engine}", "{features}"),
    ("Create Pulumi for AWS API Gateway", "{api_type} API"),
    ("Create Pulumi for AWS ECS Fargate", ""),
    # GCP
    ("Create Pulumi for GCP Compute instance", "{machine_type}"),
    ("Create Pulumi for GCP GKE cluster", "{mode} mode"),
    ("Create Pulumi for GCP Cloud Storage", "{features}"),
    ("Create Pulumi for GCP Cloud Function", "{runtime}"),
    ("Create Pulumi for GCP Cloud SQL", "{engine}"),
    # Azure
    ("Create Pulumi for Azure VM", "{os}"),
    ("Create Pulumi for Azure AKS", "{node_pools} node pools"),
    ("Create Pulumi for Azure Storage Account", ""),
    ("Create Pulumi for Azure Function", "{runtime}"),
    ("Create Pulumi for Azure SQL Database", ""),
    # Kubernetes
    ("Create Pulumi for Kubernetes Deployment", "{image}, {replicas} replicas"),
    ("Create Pulumi for Kubernetes Service", "{svc_type} type"),
    ("Create Pulumi for Kubernetes Ingress", "{features}"),
    ("Create Pulumi for Kubernetes ConfigMap", ""),
    ("Create Pulumi for Kubernetes Secret", ""),
]

# Ansible Playbooks
ANSIBLE_TEMPLATES = [
    # Server Setup
    ("Create Ansible playbook for {os} server setup", "install {packages}"),
    ("Create Ansible playbook for Docker installation", "{os}"),
    ("Create Ansible playbook for Kubernetes node setup", "{role} node"),
    ("Create Ansible playbook for Nginx installation", "with SSL"),
    ("Create Ansible playbook for PostgreSQL installation", "{version}"),
    # Configuration Management
    ("Create Ansible playbook for user management", "create {num_users} users"),
    ("Create Ansible playbook for SSH hardening", ""),
    ("Create Ansible playbook for firewall configuration", "allow {ports}"),
    ("Create Ansible playbook for package updates", "{os}"),
    ("Create Ansible playbook for timezone configuration", ""),
    # Application Deployment
    ("Create Ansible playbook for {lang} application deployment", ""),
    ("Create Ansible playbook for Docker container deployment", "{image}"),
    ("Create Ansible playbook for systemd service", "{service_name}"),
    ("Create Ansible role for {service_name}", ""),
    ("Create Ansible playbook for log rotation", "{retention} days"),
    # Monitoring & Security
    ("Create Ansible playbook for Prometheus node exporter", ""),
    ("Create Ansible playbook for Filebeat installation", ""),
    ("Create Ansible playbook for fail2ban setup", ""),
    ("Create Ansible playbook for SSL certificate renewal", ""),
    ("Create Ansible playbook for backup configuration", "{destination}"),
    # Cloud
    ("Create Ansible playbook for AWS CLI setup", ""),
    ("Create Ansible playbook for EC2 instance provisioning", "{type}"),
    ("Create Ansible playbook for S3 bucket management", ""),
]

# Helm Charts
HELM_TEMPLATES = [
    # Application Charts
    ("Create Helm chart for {lang} application", "{replicas} replicas"),
    ("Create Helm chart values.yaml", "production environment"),
    ("Create Helm chart with ingress", "{ingress_class}"),
    ("Create Helm chart with HPA", "target {metric} {threshold}%"),
    ("Create Helm chart with PVC", "{size} storage"),
    # Database Charts
    ("Create Helm chart for PostgreSQL", "{replicas} replicas"),
    ("Create Helm chart for Redis", "{mode} mode"),
    ("Create Helm chart for MongoDB", "{replicas} replicas"),
    ("Create Helm chart for Elasticsearch", "{nodes} nodes"),
    # Infrastructure Charts
    ("Create Helm chart for Nginx Ingress Controller", ""),
    ("Create Helm chart for Prometheus stack", ""),
    ("Create Helm chart for Grafana", "with {datasource} datasource"),
    ("Create Helm chart for Cert-Manager", ""),
    ("Create Helm chart for External DNS", "{provider} provider"),
    # Helpers
    ("Create Helm _helpers.tpl", "for {app_name}"),
    ("Create Helm chart with ConfigMap", ""),
    ("Create Helm chart with Secret", ""),
    ("Create Helm chart with ServiceAccount", ""),
    ("Create Helmfile for multi-chart deployment", "{num_charts} charts"),
]

# Azure ARM/Bicep Templates
AZURE_TEMPLATES = [
    # Compute
    ("Create Azure Bicep for VM", "{os}"),
    ("Create Azure ARM template for VM Scale Set", "min: {min}, max: {max}"),
    ("Create Azure Bicep for AKS cluster", "{node_pools} node pools"),
    ("Create Azure Bicep for Container Instance", "{image}"),
    ("Create Azure Bicep for Function App", "{runtime}"),
    # Storage
    ("Create Azure Bicep for Storage Account", "{sku}"),
    ("Create Azure Bicep for Blob Container", "{access_level}"),
    ("Create Azure Bicep for File Share", "{size}GB"),
    # Networking
    ("Create Azure Bicep for Virtual Network", "CIDR: {cidr}"),
    ("Create Azure Bicep for Application Gateway", ""),
    ("Create Azure Bicep for Load Balancer", "{sku}"),
    ("Create Azure Bicep for NSG", "allow {ports}"),
    # Database
    ("Create Azure Bicep for SQL Database", "{tier}"),
    ("Create Azure Bicep for Cosmos DB", "{api} API"),
    ("Create Azure Bicep for Redis Cache", "{sku}"),
    ("Create Azure Bicep for PostgreSQL Flexible Server", ""),
    # Security & Identity
    ("Create Azure Bicep for Key Vault", ""),
    ("Create Azure Bicep for Managed Identity", ""),
    ("Create Azure Bicep for App Configuration", ""),
]

# Crossplane Compositions
CROSSPLANE_TEMPLATES = [
    ("Create Crossplane Composition for AWS VPC", ""),
    ("Create Crossplane Composition for AWS RDS", "{engine}"),
    ("Create Crossplane Composition for AWS EKS", ""),
    ("Create Crossplane XRD for database", ""),
    ("Create Crossplane Claim for {resource_type}", ""),
    ("Create Crossplane ProviderConfig for AWS", ""),
    ("Create Crossplane ProviderConfig for GCP", ""),
    ("Create Crossplane Composition for GCP GKE", ""),
    ("Create Crossplane Composition for Azure AKS", ""),
]

# Monitoring & Observability Templates
MONITORING_TEMPLATES = [
    # Prometheus
    ("Create Prometheus scrape config", "for {service_name} on port {port}"),
    ("Create Prometheus alerting rules", "{alert_type} for {service_name}"),
    ("Create Prometheus recording rules", "for {metric} aggregation"),
    ("Create Prometheus ServiceMonitor", "for {service_name}"),
    ("Create Prometheus PodMonitor", "for {app_name}"),
    ("Create Prometheus federation config", ""),
    # Grafana
    ("Create Grafana dashboard JSON", "for {service_name} metrics"),
    ("Create Grafana dashboard for Kubernetes", "cluster overview"),
    ("Create Grafana dashboard for {db}", "performance metrics"),
    ("Create Grafana alerting rules", "{alert_type}"),
    ("Create Grafana datasource config", "{datasource}"),
    # Alertmanager
    ("Create Alertmanager config", "with {notification} notification"),
    ("Create Alertmanager silence", "for {alert_type}"),
    ("Create Alertmanager routing rules", "team: {team}"),
    # Logging
    ("Create Fluentd config", "parse {log_format} logs"),
    ("Create Fluent Bit config", "forward to {destination}"),
    ("Create Loki config", "with {storage} storage"),
    ("Create Promtail config", "scrape {log_source}"),
    ("Create Elasticsearch index template", "for {log_type} logs"),
    ("Create Logstash pipeline", "{source} to {destination}"),
    ("Create Filebeat config", "collect {log_type} logs"),
    # Tracing
    ("Create Jaeger deployment", "{storage} backend"),
    ("Create OpenTelemetry Collector config", ""),
    ("Create Zipkin deployment", ""),
    ("Create Tempo config", "with {storage} storage"),
    # Kubernetes Monitoring
    ("Create kube-state-metrics deployment", ""),
    ("Create node-exporter DaemonSet", ""),
    ("Create metrics-server deployment", ""),
    # Cloud Monitoring
    ("Create CloudWatch dashboard", "for {service_name}"),
    ("Create CloudWatch alarm", "{metric} > {threshold}"),
    ("Create Datadog monitor", "{monitor_type} for {service_name}"),
    ("Create New Relic alert policy", ""),
    ("Create PagerDuty service integration", ""),
    # SLO/SLI
    ("Create SLO definition", "{slo_type} for {service_name}"),
    ("Create error budget alert", ""),
]

# Error Handling & Debugging Templates
ERROR_TEMPLATES = [
    # Kubernetes Troubleshooting
    ("Debug Kubernetes pod CrashLoopBackOff", "{pod_name}"),
    ("Debug Kubernetes pod ImagePullBackOff", "{image}"),
    ("Debug Kubernetes pod OOMKilled", "{pod_name}"),
    ("Debug Kubernetes service not reachable", "{service_name}"),
    ("Debug Kubernetes ingress 502 error", ""),
    ("Debug Kubernetes PVC pending", ""),
    ("Debug Kubernetes node NotReady", ""),
    ("Debug Kubernetes DNS resolution", ""),
    ("Create Kubernetes debug pod", "with {tools}"),
    # Terraform Errors
    ("Fix Terraform state lock error", ""),
    ("Fix Terraform provider authentication", "{provider}"),
    ("Fix Terraform resource dependency cycle", ""),
    ("Fix Terraform plan drift", "{resource_type}"),
    ("Debug Terraform apply timeout", ""),
    # Docker Errors
    ("Debug Docker container exit code {exit_code}", ""),
    ("Fix Docker build failed", "layer {layer}"),
    ("Debug Docker network connectivity", ""),
    ("Fix Docker volume permission denied", ""),
    ("Debug Docker compose service unhealthy", "{service_name}"),
    # CI/CD Errors
    ("Debug GitHub Actions workflow failure", "{step}"),
    ("Fix GitLab CI pipeline timeout", ""),
    ("Debug Jenkins build failure", "{stage}"),
    ("Fix ArgoCD sync failed", "{app_name}"),
    ("Debug Helm upgrade failed", "{release}"),
    # Cloud Errors
    ("Debug AWS Lambda timeout", "{function_name}"),
    ("Fix AWS IAM permission denied", "{action}"),
    ("Debug AWS ECS task stopped", "{reason}"),
    ("Fix AWS RDS connection timeout", ""),
    ("Debug GCP Cloud Run cold start", ""),
    ("Fix Azure Function app crash", ""),
    # Application Errors
    ("Debug high latency in {service_name}", "p99 > {threshold}ms"),
    ("Debug memory leak in {lang} application", ""),
    ("Fix connection pool exhausted", "{db}"),
    ("Debug intermittent 5xx errors", "{service_name}"),
    ("Create runbook for {incident_type}", ""),
    # Network Errors
    ("Debug SSL/TLS handshake failure", ""),
    ("Fix CORS errors", "{origin}"),
    ("Debug load balancer health check failing", ""),
    ("Fix DNS propagation issues", "{domain}"),
]

MLOPS_TEMPLATES = [
    # Model Training Infrastructure
    ("Create Kubernetes manifest for ML training job", "{framework} with {gpu_type} GPU"),
    ("Create Terraform for SageMaker training job", "{instance_type} instance"),
    ("Create Terraform for SageMaker endpoint", "{model_type} model"),
    ("Create Kubernetes manifest for distributed training", "{num_workers} workers, {framework}"),
    ("Create docker-compose for ML development environment", "{framework} with Jupyter"),
    # MLflow
    ("Create docker-compose for MLflow tracking server", "{backend_store} backend"),
    ("Create Kubernetes manifest for MLflow", "with {storage} artifact storage"),
    ("Create Terraform for MLflow on AWS", "S3 + RDS backend"),
    # Kubeflow
    ("Create Kubeflow pipeline", "{pipeline_type}"),
    ("Create Kubeflow notebook server", "{framework} with {gpu_type}"),
    ("Create Kubeflow serving manifest", "{model_format} model"),
    # Feature Store
    ("Create Terraform for Feast feature store", "{offline_store} offline store"),
    ("Create Kubernetes manifest for Feast", "with Redis online store"),
    # Model Serving
    ("Create Kubernetes manifest for TensorFlow Serving", "{model_name} model"),
    ("Create Kubernetes manifest for TorchServe", "{model_type}"),
    ("Create Kubernetes manifest for Triton Inference Server", "{backend} backend"),
    ("Create Kubernetes manifest for Seldon Core", "{protocol} protocol"),
    ("Create Kubernetes manifest for KServe", "{framework} predictor"),
    ("Create Terraform for SageMaker serverless endpoint", ""),
    # Data Pipeline
    ("Create Airflow DAG for ML pipeline", "{schedule}"),
    ("Create Airflow DAG for data preprocessing", "{source} to {destination}"),
    ("Create Kubernetes manifest for Argo Workflows", "{workflow_type}"),
    ("Create Prefect flow for ML training", "{schedule}"),
    # GPU Infrastructure
    ("Create Terraform for GPU EC2 instances", "{gpu_type}, {num_gpus} GPUs"),
    ("Create Kubernetes manifest with GPU resources", "{gpu_type}, limits: {num_gpus}"),
    ("Create Terraform for GCP GPU VM", "{gpu_type}"),
    # Experiment Tracking
    ("Create docker-compose for Weights & Biases server", ""),
    ("Create Kubernetes manifest for Neptune", ""),
    # Model Registry
    ("Create Terraform for S3 model registry", "versioned bucket"),
    ("Create Kubernetes manifest for model registry", "{registry_type}"),
    # Monitoring
    ("Create Kubernetes manifest for model monitoring", "{metrics_type} metrics"),
    ("Create Prometheus rules for ML model", "{alert_type} alerts"),
    ("Create Grafana dashboard for ML metrics", "{model_type} model"),
    # Ray/Distributed
    ("Create Kubernetes manifest for Ray cluster", "{num_workers} workers"),
    ("Create Kubernetes manifest for Ray Serve", "{model_type}"),
    ("Create Terraform for Ray on AWS", "{instance_type}"),
    # Spark ML
    ("Create Kubernetes manifest for Spark ML job", "{num_executors} executors"),
    ("Create Terraform for EMR Spark cluster", "{instance_type}"),
    # Vector Database
    ("Create Kubernetes manifest for Milvus", "{storage_type} storage"),
    ("Create docker-compose for Qdrant", ""),
    ("Create Kubernetes manifest for Weaviate", "{replicas} replicas"),
    ("Create Terraform for Pinecone", ""),
    # LLM Infrastructure
    ("Create Kubernetes manifest for vLLM", "{model_name}"),
    ("Create Kubernetes manifest for Text Generation Inference", "{model_name}"),
    ("Create docker-compose for Ollama", ""),
    ("Create Terraform for Bedrock", "{model_id}"),
]

# Parameter variations for generating 500+ tasks
PARAMS = {
    "type": ["t2.micro", "t3.small", "t3.medium", "m5.large", "c5.xlarge", "r5.large"],
    "min": ["1", "2", "3"], "max": ["5", "10", "20"],
    "zones": ["2", "3"], "lb": ["ALB", "NLB"],
    "runtime": ["python3.9", "python3.11", "nodejs18.x", "nodejs20.x", "go1.x", "java17"],
    "features": ["versioning", "encryption", "lifecycle policy", "versioning and encryption", "public access blocked"],
    "size": ["10", "50", "100", "500"],
    "cidr": ["10.0.0.0/16", "172.16.0.0/16", "192.168.0.0/16"],
    "public": ["2", "3"], "private": ["2", "3"],
    "listener": ["HTTP", "HTTPS", "TCP"],
    "ports": ["SSH", "HTTP", "HTTPS", "SSH and HTTP", "80 and 443"],
    "engine": ["PostgreSQL", "MySQL", "Redis", "Memcached"],
    "nodes": ["2", "3", "5"], "replicas": ["2", "3", "5"],
    "pk": ["id", "user_id", "order_id"], "sk": ["timestamp", "created_at", "sort_key"],
    "access": ["S3 read-only", "EC2 full", "DynamoDB read-write", "Lambda invoke"],
    "permissions": ["s3:GetObject", "ec2:*", "logs:*"],
    "metric": ["CPU", "Memory", "NetworkIn"], "threshold": ["70", "80", "90"],
    "retention": ["7", "14", "30", "90"],
    "rotation": ["enabled", "disabled"],
    "rule": ["rate-limiting", "geo-blocking", "SQL injection protection"],
    "machine_type": ["n1-standard-1", "n1-standard-2", "e2-medium"],
    "mode": ["standard", "autopilot"],
    "os": ["Ubuntu 20.04", "Ubuntu 22.04", "Windows Server 2019"],
    "node_pools": ["1", "2", "3"],
    "image": ["nginx", "redis", "postgres", "mongodb", "node", "python"],
    "cpu": ["100m", "250m", "500m", "1000m"], "memory": ["128Mi", "256Mi", "512Mi", "1Gi"],
    "probe_type": ["readiness", "liveness", "startup"],
    "port": ["80", "443", "8080", "3000", "5432"],
    "sidecar_type": ["logging", "proxy", "monitoring"],
    "affinity_type": ["node", "pod", "anti"],
    "schedule": ["*/5 * * * *", "0 * * * *", "0 0 * * *", "0 0 * * 0"],
    "completions": ["1", "3", "5"],
    "policy_type": ["deny-all", "allow-namespace", "allow-specific-pods"],
    "lang": ["Python", "Node.js", "Go", "Java", "Rust", "Ruby", "PHP"],
    "base": ["alpine", "slim", "distroless", "ubuntu"],
    "services": ["web + db", "web + redis + db", "api + worker + db"],
    "db": ["PostgreSQL", "MySQL", "MongoDB", "Redis"],
    "stack": ["MEAN", "MERN", "Django + PostgreSQL", "Rails + PostgreSQL"],
    "action": ["tests", "lint", "build", "deploy", "release"],
    "trigger": ["on push", "on PR", "on tag", "on schedule"],
    "registry": ["ECR", "DockerHub", "GCR", "ACR"],
    "tf_action": ["plan on PR", "apply on merge", "destroy on branch delete"],
    "tool": ["kubectl", "helm", "kustomize", "argocd"],
    "notification": ["Slack", "Teams", "Discord", "email"],
    "test_type": ["unit", "integration", "e2e", "security"],
    "matrix": ["node versions", "python versions", "os matrix"],
    "stages": ["build, test, deploy", "lint, test, build, deploy"],
    "feature": ["environments", "caching", "artifacts"],
    # MLOps specific parameters
    "framework": ["PyTorch", "TensorFlow", "JAX", "scikit-learn", "XGBoost", "LightGBM"],
    "gpu_type": ["nvidia-tesla-t4", "nvidia-tesla-v100", "nvidia-a100", "nvidia-l4"],
    "num_gpus": ["1", "2", "4", "8"],
    "num_workers": ["2", "4", "8", "16"],
    "instance_type": ["ml.m5.large", "ml.p3.2xlarge", "ml.g4dn.xlarge", "ml.p4d.24xlarge"],
    "model_type": ["classification", "regression", "object-detection", "nlp", "llm"],
    "backend_store": ["SQLite", "PostgreSQL", "MySQL"],
    "storage": ["S3", "GCS", "MinIO", "Azure Blob"],
    "pipeline_type": ["training", "inference", "preprocessing", "feature-engineering"],
    "model_format": ["TensorFlow SavedModel", "PyTorch", "ONNX", "TensorRT"],
    "offline_store": ["BigQuery", "Redshift", "Snowflake", "PostgreSQL"],
    "model_name": ["resnet50", "bert-base", "gpt2", "llama-7b", "mistral-7b"],
    "backend": ["tensorflow", "pytorch", "onnx", "tensorrt", "python"],
    "protocol": ["REST", "gRPC", "V2"],
    "source": ["S3", "BigQuery", "Snowflake", "PostgreSQL"],
    "destination": ["S3", "feature-store", "data-warehouse"],
    "workflow_type": ["training", "batch-inference", "data-pipeline", "hyperparameter-tuning"],
    "registry_type": ["MLflow", "Vertex AI", "SageMaker", "custom"],
    "metrics_type": ["drift", "accuracy", "latency", "throughput"],
    "alert_type": ["accuracy-drop", "latency-spike", "error-rate", "drift-detection"],
    "num_executors": ["2", "4", "8", "16"],
    "storage_type": ["S3", "MinIO", "local"],
    "model_id": ["anthropic.claude-v2", "amazon.titan-text", "meta.llama2-70b"],
    # Additional parameters for new templates
    "api_type": ["REST", "HTTP", "WebSocket"],
    "svc_type": ["ClusterIP", "NodePort", "LoadBalancer"],
    "packages": ["nginx, docker", "postgresql, redis", "prometheus, grafana", "git, vim, curl"],
    "role": ["master", "worker", "control-plane"],
    "version": ["13", "14", "15", "16"],
    "num_users": ["3", "5", "10"],
    "service_name": ["nginx", "redis", "api", "worker", "scheduler"],
    "ingress_class": ["nginx", "traefik", "alb", "kong"],
    "datasource": ["Prometheus", "Loki", "InfluxDB", "Elasticsearch"],
    "provider": ["aws", "gcp", "azure", "cloudflare"],
    "app_name": ["api", "web", "worker", "backend", "frontend"],
    "num_charts": ["3", "5", "7"],
    "sku": ["Standard_LRS", "Premium_LRS", "Standard_GRS"],
    "access_level": ["private", "blob", "container"],
    "tier": ["Basic", "Standard", "Premium", "Serverless"],
    "api": ["SQL", "MongoDB", "Cassandra", "Gremlin", "Table"],
    "resource_type": ["database", "vpc", "cluster", "bucket"],
    # Monitoring & Error parameters
    "team": ["platform", "backend", "frontend", "data", "ml"],
    "log_format": ["json", "apache", "nginx", "syslog", "custom"],
    "log_source": ["/var/log", "kubernetes pods", "docker containers", "systemd"],
    "log_type": ["application", "access", "error", "audit", "system"],
    "monitor_type": ["metric", "log", "apm", "synthetics"],
    "slo_type": ["availability", "latency", "throughput", "error-rate"],
    "pod_name": ["api-server", "web-frontend", "worker", "scheduler"],
    "tools": ["curl, netcat", "dig, nslookup", "tcpdump, strace", "busybox"],
    "exit_code": ["1", "137", "139", "143"],
    "layer": ["npm install", "pip install", "COPY", "apt-get"],
    "step": ["build", "test", "deploy", "lint"],
    "stage": ["Build", "Test", "Deploy", "Integration"],
    "release": ["api-v1", "web-v2", "backend-v3"],
    "function_name": ["processOrder", "sendNotification", "generateReport"],
    "action": ["s3:GetObject", "ec2:StartInstances", "lambda:InvokeFunction"],
    "reason": ["OutOfMemory", "TaskFailedToStart", "CannotPullContainer"],
    "incident_type": ["high-cpu", "database-down", "certificate-expiry", "disk-full"],
    "origin": ["localhost:3000", "example.com", "*.app.com"],
    "domain": ["api.example.com", "www.example.com", "cdn.example.com"],
}


def _generate_tasks() -> List[Dict]:
    """Generate 500+ tasks from templates"""
    tasks = []
    task_id = 0

    # Terraform tasks (200+)
    for template, input_template in TERRAFORM_TEMPLATES:
        for _ in range(5):  # 5 variations per template
            task_id += 1
            input_str = input_template
            for key, values in PARAMS.items():
                if f"{{{key}}}" in input_str:
                    import random
                    input_str = input_str.replace(f"{{{key}}}", random.choice(values))
            tasks.append({
                "id": f"tf-{task_id:03d}", "instruction": template, "input": input_str,
                "category": "terraform", "difficulty": "medium"
            })

    # Kubernetes tasks (150+)
    for template, input_template in K8S_TEMPLATES:
        for _ in range(6):
            task_id += 1
            input_str = input_template
            for key, values in PARAMS.items():
                if f"{{{key}}}" in input_str:
                    import random
                    input_str = input_str.replace(f"{{{key}}}", random.choice(values))
            tasks.append({
                "id": f"k8s-{task_id:03d}", "instruction": template, "input": input_str,
                "category": "kubernetes", "difficulty": "medium"
            })

    # Docker tasks (70+)
    for template, input_template in DOCKER_TEMPLATES:
        for _ in range(10):
            task_id += 1
            input_str = input_template
            for key, values in PARAMS.items():
                if f"{{{key}}}" in input_str:
                    import random
                    input_str = input_str.replace(f"{{{key}}}", random.choice(values))
            instr = template
            for key, values in PARAMS.items():
                if f"{{{key}}}" in instr:
                    import random
                    instr = instr.replace(f"{{{key}}}", random.choice(values))
            tasks.append({
                "id": f"docker-{task_id:03d}", "instruction": instr, "input": input_str,
                "category": "docker", "difficulty": "medium"
            })

    # CI/CD tasks (100+)
    for template, input_template in CICD_TEMPLATES:
        for _ in range(8):
            task_id += 1
            input_str = input_template
            instr = template
            for key, values in PARAMS.items():
                if f"{{{key}}}" in input_str:
                    import random
                    input_str = input_str.replace(f"{{{key}}}", random.choice(values))
                if f"{{{key}}}" in instr:
                    import random
                    instr = instr.replace(f"{{{key}}}", random.choice(values))
            tasks.append({
                "id": f"cicd-{task_id:03d}", "instruction": instr, "input": input_str,
                "category": "cicd", "difficulty": "medium"
            })

    # MLOps tasks (150+)
    for template, input_template in MLOPS_TEMPLATES:
        for _ in range(4):
            task_id += 1
            input_str = input_template
            instr = template
            for key, values in PARAMS.items():
                if f"{{{key}}}" in input_str:
                    import random
                    input_str = input_str.replace(f"{{{key}}}", random.choice(values))
                if f"{{{key}}}" in instr:
                    import random
                    instr = instr.replace(f"{{{key}}}", random.choice(values))
            tasks.append({
                "id": f"mlops-{task_id:03d}", "instruction": instr, "input": input_str,
                "category": "mlops", "difficulty": "medium"
            })

    # CloudFormation tasks (100+)
    for template, input_template in CLOUDFORMATION_TEMPLATES:
        for _ in range(4):
            task_id += 1
            input_str = input_template
            instr = template
            for key, values in PARAMS.items():
                if f"{{{key}}}" in input_str:
                    import random
                    input_str = input_str.replace(f"{{{key}}}", random.choice(values))
                if f"{{{key}}}" in instr:
                    import random
                    instr = instr.replace(f"{{{key}}}", random.choice(values))
            tasks.append({
                "id": f"cfn-{task_id:03d}", "instruction": instr, "input": input_str,
                "category": "cloudformation", "difficulty": "medium"
            })

    # Pulumi tasks (100+)
    for template, input_template in PULUMI_TEMPLATES:
        for _ in range(4):
            task_id += 1
            input_str = input_template
            instr = template
            for key, values in PARAMS.items():
                if f"{{{key}}}" in input_str:
                    import random
                    input_str = input_str.replace(f"{{{key}}}", random.choice(values))
                if f"{{{key}}}" in instr:
                    import random
                    instr = instr.replace(f"{{{key}}}", random.choice(values))
            tasks.append({
                "id": f"pulumi-{task_id:03d}", "instruction": instr, "input": input_str,
                "category": "pulumi", "difficulty": "medium"
            })

    # Ansible tasks (100+)
    for template, input_template in ANSIBLE_TEMPLATES:
        for _ in range(4):
            task_id += 1
            input_str = input_template
            instr = template
            for key, values in PARAMS.items():
                if f"{{{key}}}" in input_str:
                    import random
                    input_str = input_str.replace(f"{{{key}}}", random.choice(values))
                if f"{{{key}}}" in instr:
                    import random
                    instr = instr.replace(f"{{{key}}}", random.choice(values))
            tasks.append({
                "id": f"ansible-{task_id:03d}", "instruction": instr, "input": input_str,
                "category": "ansible", "difficulty": "medium"
            })

    # Helm tasks (80+)
    for template, input_template in HELM_TEMPLATES:
        for _ in range(4):
            task_id += 1
            input_str = input_template
            instr = template
            for key, values in PARAMS.items():
                if f"{{{key}}}" in input_str:
                    import random
                    input_str = input_str.replace(f"{{{key}}}", random.choice(values))
                if f"{{{key}}}" in instr:
                    import random
                    instr = instr.replace(f"{{{key}}}", random.choice(values))
            tasks.append({
                "id": f"helm-{task_id:03d}", "instruction": instr, "input": input_str,
                "category": "helm", "difficulty": "medium"
            })

    # Azure Bicep/ARM tasks (80+)
    for template, input_template in AZURE_TEMPLATES:
        for _ in range(4):
            task_id += 1
            input_str = input_template
            instr = template
            for key, values in PARAMS.items():
                if f"{{{key}}}" in input_str:
                    import random
                    input_str = input_str.replace(f"{{{key}}}", random.choice(values))
                if f"{{{key}}}" in instr:
                    import random
                    instr = instr.replace(f"{{{key}}}", random.choice(values))
            tasks.append({
                "id": f"azure-{task_id:03d}", "instruction": instr, "input": input_str,
                "category": "azure", "difficulty": "medium"
            })

    # Crossplane tasks (36+)
    for template, input_template in CROSSPLANE_TEMPLATES:
        for _ in range(4):
            task_id += 1
            input_str = input_template
            instr = template
            for key, values in PARAMS.items():
                if f"{{{key}}}" in input_str:
                    import random
                    input_str = input_str.replace(f"{{{key}}}", random.choice(values))
                if f"{{{key}}}" in instr:
                    import random
                    instr = instr.replace(f"{{{key}}}", random.choice(values))
            tasks.append({
                "id": f"crossplane-{task_id:03d}", "instruction": instr, "input": input_str,
                "category": "crossplane", "difficulty": "medium"
            })

    # Monitoring tasks (140+)
    for template, input_template in MONITORING_TEMPLATES:
        for _ in range(4):
            task_id += 1
            input_str = input_template
            instr = template
            for key, values in PARAMS.items():
                if f"{{{key}}}" in input_str:
                    import random
                    input_str = input_str.replace(f"{{{key}}}", random.choice(values))
                if f"{{{key}}}" in instr:
                    import random
                    instr = instr.replace(f"{{{key}}}", random.choice(values))
            tasks.append({
                "id": f"mon-{task_id:03d}", "instruction": instr, "input": input_str,
                "category": "monitoring", "difficulty": "medium"
            })

    # Error/Debugging tasks (140+)
    for template, input_template in ERROR_TEMPLATES:
        for _ in range(4):
            task_id += 1
            input_str = input_template
            instr = template
            for key, values in PARAMS.items():
                if f"{{{key}}}" in input_str:
                    import random
                    input_str = input_str.replace(f"{{{key}}}", random.choice(values))
                if f"{{{key}}}" in instr:
                    import random
                    instr = instr.replace(f"{{{key}}}", random.choice(values))
            tasks.append({
                "id": f"debug-{task_id:03d}", "instruction": instr, "input": input_str,
                "category": "debugging", "difficulty": "medium"
            })

    return tasks


# Pre-generate tasks
TASKS = _generate_tasks()


class IaCBench:
    """InfraMind-Bench: 500+ Infrastructure-as-Code tasks"""

    def __init__(self, categories: Optional[List[str]] = None, size: Optional[int] = None):
        self.tasks = TASKS.copy()
        if categories:
            self.tasks = [t for t in self.tasks if t["category"] in categories]
        if size:
            self.tasks = self.tasks[:size]

    def __len__(self): return len(self.tasks)
    def __getitem__(self, idx): return self.tasks[idx]
    def __iter__(self): yield from self.tasks

    def to_alpaca_format(self) -> List[Dict]:
        return [{"instruction": t["instruction"], "input": t["input"], "output": ""} for t in self.tasks]

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"name": "InfraMind-Bench", "version": "1.0", "total": len(self.tasks), "tasks": self.tasks}, f, indent=2)


def create_dataset(categories: Optional[List[str]] = None, size: Optional[int] = None) -> IaCBench:
    return IaCBench(categories=categories, size=size)
