"""
InfraMind IaC Templates - All template definitions for dataset generation.

Categories:
- Terraform (AWS, GCP, Azure)
- Kubernetes
- Docker (Dockerfile, docker-compose)
- CI/CD (GitHub Actions, GitLab CI, Jenkins)
- CloudFormation
- Pulumi
- Ansible
- Helm
- Monitoring (Prometheus, Grafana)
- Debugging/Errors
- MLOps
"""

# Out-of-domain response for non-IaC queries
OUT_OF_DOMAIN_RESPONSE = "I'm a DevOps and Infrastructure assistant. I can help with Terraform, Kubernetes, Docker, CI/CD, cloud infrastructure, monitoring, and debugging. For other topics, please use a general-purpose assistant."

OUT_OF_DOMAIN_EXAMPLES = [
    ("What's the best movie of 2024?", ""),
    ("Recommend me a Netflix series", ""),
    ("Write a poem about nature", ""),
    ("Tell me a joke", ""),
    ("What's a good recipe for pasta?", ""),
    ("How do I lose weight?", ""),
    ("Explain Shakespeare's Hamlet", ""),
    ("What's the capital of France?", ""),
    ("Solve this math problem: 2x + 5 = 15", ""),
    ("Help me with my homework", ""),
    ("How do I ask someone on a date?", ""),
    ("What's the meaning of life?", ""),
    ("I have a headache, what medicine should I take?", ""),
    ("Should I invest in Bitcoin?", ""),
    ("What stocks should I buy?", ""),
    ("Who won the Super Bowl?", ""),
    ("Plan a trip to Paris for me", ""),
    ("Write a React component for a button", ""),
    ("How do I center a div in CSS?", ""),
    ("What's 2 + 2?", ""),
    ("Write a cover letter for me", ""),
]

# =============================================================================
# TERRAFORM TEMPLATES
# =============================================================================
TERRAFORM_TEMPLATES = [
    ("Create Terraform for AWS EC2 instance", "{type} instance type"),
    ("Create Terraform for Auto Scaling Group", "min: {min}, max: {max}"),
    ("Create Terraform for EKS cluster", "{zones} availability zones"),
    ("Create Terraform for ECS Fargate service", "with {lb} load balancer"),
    ("Create Terraform for Lambda function", "{runtime} runtime"),
    ("Create Terraform for S3 bucket", "{features}"),
    ("Create Terraform for EBS volume", "{size}GB, {ebs_type} type"),
    ("Create Terraform for VPC", "CIDR: {cidr}"),
    ("Create Terraform for VPC with subnets", "{public} public, {private} private subnets"),
    ("Create Terraform for Application Load Balancer", "{listener} listener"),
    ("Create Terraform for security group", "allow {ports}"),
    ("Create Terraform for NAT Gateway", ""),
    ("Create Terraform for Route53 hosted zone", ""),
    ("Create Terraform for RDS {engine}", "{features}"),
    ("Create Terraform for DynamoDB table", "partition key: {pk}"),
    ("Create Terraform for ElastiCache {engine}", "{nodes} nodes"),
    ("Create Terraform for Aurora cluster", "{replicas} read replicas"),
    ("Create Terraform for API Gateway", "{api_type} API"),
    ("Create Terraform for SQS queue", "{features}"),
    ("Create Terraform for SNS topic", ""),
    ("Create Terraform for IAM role", "{access} access"),
    ("Create Terraform for IAM policy", "{permissions}"),
    ("Create Terraform for CloudWatch alarm", "{metric} > {threshold}"),
    ("Create Terraform for CloudWatch log group", "{retention} day retention"),
    ("Create Terraform for KMS key", "{rotation}"),
    ("Create Terraform for Secrets Manager secret", ""),
    ("Create Terraform for GCE instance", "{machine_type}"),
    ("Create Terraform for GCS bucket", "{features}"),
    ("Create Terraform for GKE cluster", "{mode} mode"),
    ("Create Terraform for Azure VM", "{os}"),
    ("Create Terraform for AKS cluster", "{node_pools} node pools"),
]

# =============================================================================
# KUBERNETES TEMPLATES
# =============================================================================
K8S_TEMPLATES = [
    ("Create Kubernetes Deployment", "{image}, {replicas} replicas"),
    ("Create Kubernetes Deployment with resources", "CPU: {cpu}, Memory: {memory}"),
    ("Create Kubernetes Deployment with probes", "{probe_type} probe on port {port}"),
    ("Create Kubernetes Deployment with init container", ""),
    ("Create Kubernetes Deployment with sidecar", "{sidecar_type} sidecar"),
    ("Create Kubernetes Service", "{svc_type} type, port {port}"),
    ("Create Kubernetes Ingress", "{features}"),
    ("Create Kubernetes ConfigMap", ""),
    ("Create Kubernetes Secret", "{secret_type} type"),
    ("Create Kubernetes PersistentVolumeClaim", "{size} storage"),
    ("Create Kubernetes StatefulSet", "{app}, {replicas} replicas"),
    ("Create Kubernetes DaemonSet", "{purpose}"),
    ("Create Kubernetes CronJob", "schedule: {schedule}"),
    ("Create Kubernetes Job", "{completions} completions"),
    ("Create Kubernetes HorizontalPodAutoscaler", "target {metric} {target}%"),
    ("Create Kubernetes NetworkPolicy", "{policy_type}"),
    ("Create Kubernetes Role", "{permissions}"),
    ("Create Kubernetes ServiceAccount", ""),
    ("Create Kubernetes PodDisruptionBudget", "minAvailable: {min}"),
    ("Create Kubernetes ResourceQuota", ""),
]

# =============================================================================
# DOCKER TEMPLATES
# =============================================================================
DOCKER_TEMPLATES = [
    ("Create Dockerfile for {lang} application", "{features}"),
    ("Create multi-stage Dockerfile for {lang}", "minimal final image"),
    ("Create Dockerfile with {base} base image", ""),
    ("Create secure Dockerfile", "non-root user, {base} base"),
    ("Create docker-compose.yml", "{services}"),
    ("Create docker-compose.yml with {db}", ""),
    ("Create docker-compose.yml for {stack}", ""),
]

# =============================================================================
# CI/CD TEMPLATES
# =============================================================================
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
    ("Create Jenkins pipeline", "{pipeline_type}"),
]

# =============================================================================
# CLOUDFORMATION TEMPLATES
# =============================================================================
CLOUDFORMATION_TEMPLATES = [
    ("Create CloudFormation for EC2 instance", "{type} instance type"),
    ("Create CloudFormation for Auto Scaling Group", "min: {min}, max: {max}"),
    ("Create CloudFormation for Lambda function", "{runtime} runtime"),
    ("Create CloudFormation for S3 bucket", "{features}"),
    ("Create CloudFormation for VPC", "CIDR: {cidr}"),
    ("Create CloudFormation for RDS {engine}", "{features}"),
    ("Create CloudFormation for DynamoDB table", "partition key: {pk}"),
    ("Create CloudFormation for IAM role", "{access} access"),
    ("Create CloudFormation SAM template", "Lambda + API Gateway"),
    ("Create CloudFormation for SQS queue", "{features}"),
]

# =============================================================================
# PULUMI TEMPLATES
# =============================================================================
PULUMI_TEMPLATES = [
    ("Create Pulumi TypeScript for AWS EC2", "{type} instance"),
    ("Create Pulumi Python for AWS EC2", "{type} instance"),
    ("Create Pulumi for AWS S3 bucket", "{features}"),
    ("Create Pulumi for AWS VPC", "CIDR: {cidr}"),
    ("Create Pulumi for AWS Lambda", "{runtime} runtime"),
    ("Create Pulumi for AWS EKS cluster", "{zones} availability zones"),
    ("Create Pulumi for GCP Compute instance", "{machine_type}"),
    ("Create Pulumi for GCP GKE cluster", "{mode} mode"),
    ("Create Pulumi for Kubernetes Deployment", "{image}, {replicas} replicas"),
    ("Create Pulumi for Kubernetes Service", "{svc_type} type"),
]

# =============================================================================
# ANSIBLE TEMPLATES
# =============================================================================
ANSIBLE_TEMPLATES = [
    ("Create Ansible playbook for {os} server setup", "install {packages}"),
    ("Create Ansible playbook for Docker installation", "{os}"),
    ("Create Ansible playbook for Kubernetes node setup", "{role} node"),
    ("Create Ansible playbook for Nginx installation", "with SSL"),
    ("Create Ansible playbook for PostgreSQL installation", "{version}"),
    ("Create Ansible playbook for user management", "create {num_users} users"),
    ("Create Ansible playbook for SSH hardening", ""),
    ("Create Ansible playbook for firewall configuration", "allow {ports}"),
    ("Create Ansible playbook for {lang} application deployment", ""),
    ("Create Ansible playbook for Docker container deployment", "{image}"),
]

# =============================================================================
# HELM TEMPLATES
# =============================================================================
HELM_TEMPLATES = [
    ("Create Helm chart for {lang} application", "{replicas} replicas"),
    ("Create Helm chart values.yaml", "production environment"),
    ("Create Helm chart with ingress", "{ingress_class}"),
    ("Create Helm chart with HPA", "target {metric} {threshold}%"),
    ("Create Helm chart for PostgreSQL", "{replicas} replicas"),
    ("Create Helm chart for Redis", "{redis_mode} mode"),
    ("Create Helm chart for Nginx Ingress Controller", ""),
    ("Create Helm chart for Prometheus stack", ""),
    ("Create Helm chart for Grafana", "with {datasource} datasource"),
    ("Create Helmfile for multi-chart deployment", "{num_charts} charts"),
]

# =============================================================================
# MONITORING TEMPLATES
# =============================================================================
MONITORING_TEMPLATES = [
    ("Create Prometheus scrape config", "for {service_name} on port {port}"),
    ("Create Prometheus alerting rules", "{alert_type} for {service_name}"),
    ("Create Prometheus ServiceMonitor", "for {service_name}"),
    ("Create Grafana dashboard JSON", "for {service_name} metrics"),
    ("Create Grafana dashboard for Kubernetes", "cluster overview"),
    ("Create Alertmanager config", "with {notification} notification"),
    ("Create Fluentd config", "parse {log_format} logs"),
    ("Create Loki config", "with {storage} storage"),
    ("Create CloudWatch dashboard", "for {service_name}"),
    ("Create CloudWatch alarm", "{metric} > {threshold}"),
]

# =============================================================================
# ERROR/DEBUGGING TEMPLATES
# =============================================================================
ERROR_TEMPLATES = [
    ("Debug Kubernetes pod CrashLoopBackOff", "{pod_name}"),
    ("Debug Kubernetes pod ImagePullBackOff", "{image}"),
    ("Debug Kubernetes pod OOMKilled", "{pod_name}"),
    ("Debug Kubernetes service not reachable", "{service_name}"),
    ("Fix Terraform state lock error", ""),
    ("Fix Terraform provider authentication", "{provider}"),
    ("Debug Docker container exit code {exit_code}", ""),
    ("Fix Docker build failed", "layer {layer}"),
    ("Debug GitHub Actions workflow failure", "{step}"),
    ("Fix ArgoCD sync failed", "{app_name}"),
    ("Debug AWS Lambda timeout", "{function_name}"),
    ("Fix AWS IAM permission denied", "{iam_action}"),
    ("Debug high latency in {service_name}", "p99 > {latency_threshold}ms"),
    ("Create runbook for {incident_type}", ""),
]

# =============================================================================
# MLOPS TEMPLATES
# =============================================================================
MLOPS_TEMPLATES = [
    ("Create Kubernetes manifest for ML training job", "{framework} with {gpu_type} GPU"),
    ("Create Terraform for SageMaker training job", "{instance_type} instance"),
    ("Create Terraform for SageMaker endpoint", "{model_type} model"),
    ("Create Kubernetes manifest for distributed training", "{num_workers} workers, {framework}"),
    ("Create docker-compose for MLflow tracking server", "{backend_store} backend"),
    ("Create Kubernetes manifest for MLflow", "with {storage} artifact storage"),
    ("Create Kubeflow pipeline", "{pipeline_type}"),
    ("Create Kubernetes manifest for TensorFlow Serving", "{model_name} model"),
    ("Create Kubernetes manifest for TorchServe", "{model_type}"),
    ("Create Kubernetes manifest for Triton Inference Server", "{backend} backend"),
    ("Create Kubernetes manifest for KServe", "{framework} predictor"),
    ("Create Airflow DAG for ML pipeline", "{schedule}"),
    ("Create Terraform for GPU EC2 instances", "{gpu_type}, {num_gpus} GPUs"),
    ("Create Kubernetes manifest with GPU resources", "{gpu_type}, limits: {num_gpus}"),
    ("Create Kubernetes manifest for Ray cluster", "{num_workers} workers"),
    ("Create Kubernetes manifest for Milvus", "{storage_type} storage"),
    ("Create Kubernetes manifest for vLLM", "{model_name}"),
    ("Create Kubernetes manifest for Text Generation Inference", "{model_name}"),
    ("Create docker-compose for Ollama", ""),
]

# =============================================================================
# PARAMETER VALUES FOR TEMPLATE FILLING
# =============================================================================
PARAMS = {
    "type": ["t2.micro", "t3.small", "t3.medium", "m5.large", "c5.xlarge"],
    "min": ["1", "2", "3"],
    "max": ["5", "10", "20"],
    "zones": ["2", "3"],
    "lb": ["ALB", "NLB"],
    "runtime": ["python3.9", "python3.11", "nodejs18.x", "nodejs20.x", "go1.x"],
    "features": ["versioning", "encryption", "lifecycle policy", "versioning and encryption"],
    "size": ["10", "50", "100", "500"],
    "ebs_type": ["gp3", "io1", "st1"],
    "cidr": ["10.0.0.0/16", "172.16.0.0/16", "192.168.0.0/16"],
    "public": ["2", "3"],
    "private": ["2", "3"],
    "listener": ["HTTP", "HTTPS", "TCP"],
    "ports": ["SSH", "HTTP", "HTTPS", "80 and 443"],
    "engine": ["PostgreSQL", "MySQL", "Redis", "Memcached"],
    "nodes": ["2", "3", "5"],
    "replicas": ["2", "3", "5"],
    "pk": ["id", "user_id", "order_id"],
    "access": ["S3 read-only", "EC2 full", "DynamoDB read-write", "Lambda invoke"],
    "permissions": ["s3:GetObject", "ec2:*", "logs:*"],
    "metric": ["CPU", "Memory", "NetworkIn"],
    "threshold": ["70", "80", "90"],
    "target": ["70", "80"],
    "retention": ["7", "14", "30", "90"],
    "rotation": ["enabled", "disabled"],
    "machine_type": ["n1-standard-1", "n1-standard-2", "e2-medium"],
    "mode": ["standard", "autopilot"],
    "os": ["Ubuntu 20.04", "Ubuntu 22.04", "Windows Server 2019"],
    "node_pools": ["1", "2", "3"],
    "image": ["nginx", "redis", "postgres", "mongodb", "node", "python"],
    "cpu": ["100m", "250m", "500m", "1000m"],
    "memory": ["128Mi", "256Mi", "512Mi", "1Gi"],
    "probe_type": ["readiness", "liveness", "startup"],
    "port": ["80", "443", "8080", "3000", "5432"],
    "sidecar_type": ["logging", "proxy", "monitoring"],
    "secret_type": ["Opaque", "kubernetes.io/tls", "kubernetes.io/dockerconfigjson"],
    "purpose": ["logging", "monitoring", "networking"],
    "schedule": ["*/5 * * * *", "0 * * * *", "0 0 * * *", "0 0 * * 0"],
    "completions": ["1", "3", "5"],
    "policy_type": ["deny-all", "allow-namespace", "allow-specific-pods"],
    "app": ["redis", "mysql", "mongodb", "elasticsearch"],
    "lang": ["Python", "Node.js", "Go", "Java", "Rust"],
    "base": ["alpine", "slim", "distroless", "ubuntu"],
    "services": ["web + db", "web + redis + db", "api + worker + db"],
    "db": ["PostgreSQL", "MySQL", "MongoDB", "Redis"],
    "stack": ["MEAN", "MERN", "Django + PostgreSQL", "Rails + PostgreSQL"],
    "action": ["tests", "lint", "build", "deploy", "release"],
    "trigger": ["on push", "on PR", "on tag", "on schedule"],
    "registry": ["ECR", "DockerHub", "GCR", "ACR"],
    "tf_action": ["plan on PR", "apply on merge"],
    "tool": ["kubectl", "helm", "kustomize", "argocd"],
    "notification": ["Slack", "Teams", "Discord"],
    "test_type": ["unit", "integration", "e2e", "security"],
    "matrix": ["node versions", "python versions", "os matrix"],
    "stages": ["build, test, deploy", "lint, test, build, deploy"],
    "pipeline_type": ["Declarative", "Scripted"],
    "api_type": ["REST", "HTTP", "WebSocket"],
    "svc_type": ["ClusterIP", "NodePort", "LoadBalancer"],
    "packages": ["nginx, docker", "postgresql, redis", "prometheus, grafana"],
    "role": ["master", "worker", "control-plane"],
    "version": ["13", "14", "15", "16"],
    "num_users": ["3", "5", "10"],
    "service_name": ["nginx", "redis", "api", "worker", "scheduler"],
    "ingress_class": ["nginx", "traefik", "alb"],
    "redis_mode": ["standalone", "cluster", "sentinel"],
    "datasource": ["Prometheus", "Loki", "InfluxDB"],
    "num_charts": ["3", "5", "7"],
    "alert_type": ["high-cpu", "high-memory", "error-rate", "latency"],
    "log_format": ["json", "apache", "nginx", "syslog"],
    "storage": ["S3", "GCS", "MinIO"],
    "pod_name": ["api-server", "web-frontend", "worker"],
    "provider": ["aws", "gcp", "azure"],
    "exit_code": ["1", "137", "139", "143"],
    "layer": ["npm install", "pip install", "COPY"],
    "step": ["build", "test", "deploy", "lint"],
    "function_name": ["processOrder", "sendNotification", "generateReport"],
    "iam_action": ["s3:GetObject", "ec2:StartInstances", "lambda:InvokeFunction"],
    "latency_threshold": ["100", "500", "1000"],
    "incident_type": ["high-cpu", "database-down", "certificate-expiry"],
    "framework": ["PyTorch", "TensorFlow", "JAX", "scikit-learn"],
    "gpu_type": ["nvidia-tesla-t4", "nvidia-tesla-v100", "nvidia-a100"],
    "num_gpus": ["1", "2", "4", "8"],
    "num_workers": ["2", "4", "8", "16"],
    "instance_type": ["ml.m5.large", "ml.p3.2xlarge", "ml.g4dn.xlarge"],
    "model_type": ["classification", "regression", "object-detection", "nlp"],
    "backend_store": ["SQLite", "PostgreSQL", "MySQL"],
    "model_name": ["resnet50", "bert-base", "gpt2", "llama-7b", "mistral-7b"],
    "backend": ["tensorflow", "pytorch", "onnx", "tensorrt"],
    "storage_type": ["S3", "MinIO", "local"],
}

# All templates grouped for easy iteration
ALL_TEMPLATES = {
    "terraform": (TERRAFORM_TEMPLATES, 5),
    "kubernetes": (K8S_TEMPLATES, 6),
    "docker": (DOCKER_TEMPLATES, 10),
    "cicd": (CICD_TEMPLATES, 8),
    "cloudformation": (CLOUDFORMATION_TEMPLATES, 4),
    "pulumi": (PULUMI_TEMPLATES, 4),
    "ansible": (ANSIBLE_TEMPLATES, 4),
    "helm": (HELM_TEMPLATES, 4),
    "monitoring": (MONITORING_TEMPLATES, 4),
    "debugging": (ERROR_TEMPLATES, 4),
    "mlops": (MLOPS_TEMPLATES, 4),
}
