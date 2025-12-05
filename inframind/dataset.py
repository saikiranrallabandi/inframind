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
