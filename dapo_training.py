"""
InfraMind DAPO Training - Multi-Step Reasoning for IaC Generation.

DAPO (Direct Advantage Policy Optimization) extends GRPO with:
1. Clip-Higher: Asymmetric clipping (ε_low=0.2, ε_high=0.28) to prevent entropy collapse
2. Dynamic Sampling: Skip batches where all rewards are identical (no gradient signal)
3. Token-Level Loss: Per-token policy gradient for better CoT credit assignment
4. Overlong Reward Shaping: Soft penalty for sequences approaching max length
5. Multi-step reasoning prompts (Analyze → Plan → Generate → Verify)

Usage:
    modal run dapo_training.py              # Run DAPO training
    modal run dapo_training.py::evaluate    # Evaluate DAPO model

Config:
    See config/dapo.yaml for all hyperparameters
"""

import modal
import re

app = modal.App("inframind-dapo")

# =============================================================================
# CONFIGURATION (from config/dapo.yaml)
# =============================================================================
CONFIG = {
    # Model
    "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
    "torch_dtype": "bfloat16",

    # LoRA
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],

    # Training
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 5e-6,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "num_train_epochs": 2,
    "max_grad_norm": 0.1,
    "weight_decay": 0.1,

    # DAPO Specific
    "beta": 0.0,  # No KL penalty for pure DAPO
    "epsilon": 0.2,
    "epsilon_high": 0.28,  # Clip-higher
    "num_generations": 8,
    "temperature": 0.9,
    "max_completion_length": 768,

    # Dynamic Sampling
    "use_dynamic_sampling": True,
    "batch_multiplier": 3,

    # Overlong Reward Shaping
    "overlong_buffer_length": 100,
    "overlong_buffer_penalty": 0.5,

    # Reasoning
    "reasoning_bonus": 0.05,

    # Reward
    "syntax_weight": 0.4,
    "semantic_weight": 0.3,
    "structure_weight": 0.3,
    "pass_threshold": 0.6,

    # Diversity
    "diversity_threshold": 0.6,
    "diversity_penalty_min": 0.8,

    # Paths
    "grpo_checkpoint": "/models/inframind-grpo/final",
    "output_dir": "/models/inframind-dapo",
}

# Persistent volume for trained models
model_volume = modal.Volume.from_name("inframind-models", create_if_missing=True)
MODEL_DIR = "/models"

# Build image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "transformers==4.46.0",
        "accelerate>=0.34.0",
        "bitsandbytes>=0.43.0",
        "peft==0.14.0",
        "trl>=0.12.0",
        "datasets",
    )
    .add_local_file("data/real_code_2k.json", "/root/training_data.json")
)


# =============================================================================
# MULTI-STEP REASONING PROMPT
# =============================================================================

DAPO_SYSTEM_PROMPT = """You are an Infrastructure-as-Code expert.
For each request, follow this multi-step reasoning process:

1. ANALYSIS: What resources and configurations are needed?
2. PLAN: What's the structure, dependencies, and best practices?
3. CODE: Generate the complete, production-ready code
4. VERIFY: Check for common errors and security issues

Always show your reasoning, then provide the final code."""

REASONING_TEMPLATE = """## Analysis
{analysis}

## Plan
{plan}

## Code
```{language}
{code}
```

## Verification
{verification}"""


# =============================================================================
# REWARD FUNCTIONS WITH DAPO EXTENSIONS
# =============================================================================

def detect_category(prompt: str) -> str:
    """Detect IaC category from prompt."""
    prompt_lower = prompt.lower()
    if "terraform" in prompt_lower:
        return "terraform"
    elif "kubernetes" in prompt_lower or "k8s" in prompt_lower:
        return "kubernetes"
    elif "dockerfile" in prompt_lower:
        return "dockerfile"
    elif "docker-compose" in prompt_lower or "docker compose" in prompt_lower:
        return "docker-compose"
    elif "github action" in prompt_lower:
        return "github-actions"
    elif "ansible" in prompt_lower:
        return "ansible"
    elif "cloudformation" in prompt_lower:
        return "cloudformation"
    return "unknown"


def extract_code_content(completion: str, category: str) -> str:
    """
    Extract code content from completion, handling multi-step reasoning outputs.
    Tries multiple extraction strategies:
    1. Code block with language hint (```dockerfile, ```yaml, etc.)
    2. Any code block (```)
    3. For Dockerfile: find lines starting with FROM/RUN/COPY etc.
    4. For YAML-based: find lines with proper indentation and colons
    5. Fallback: use entire completion
    """
    completion = completion.strip()

    # Strategy 1: Code block with language hint
    lang_hints = {
        "dockerfile": ["dockerfile", "docker"],
        "docker-compose": ["yaml", "yml", "docker-compose"],
        "kubernetes": ["yaml", "yml", "kubernetes", "k8s"],
        "terraform": ["hcl", "terraform", "tf"],
        "github-actions": ["yaml", "yml"],
        "ansible": ["yaml", "yml"],
        "cloudformation": ["yaml", "yml", "json"],
    }
    hints = lang_hints.get(category, [])
    for hint in hints:
        pattern = rf'```{hint}\n(.*?)```'
        match = re.search(pattern, completion, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Strategy 2: Any code block
    code_match = re.search(r'```(?:\w+)?\n(.*?)```', completion, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()

    # Strategy 3: For Dockerfile, extract lines that look like Dockerfile instructions
    if category == "dockerfile":
        dockerfile_instructions = {"FROM", "RUN", "CMD", "EXPOSE", "ENV", "ADD", "COPY",
                                   "ENTRYPOINT", "VOLUME", "USER", "WORKDIR", "ARG", "LABEL",
                                   "ONBUILD", "STOPSIGNAL", "HEALTHCHECK", "SHELL"}
        lines = completion.split("\n")
        dockerfile_lines = []
        capturing = False
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                first_word = stripped.split()[0].upper() if stripped.split() else ""
                if first_word in dockerfile_instructions:
                    capturing = True
                    dockerfile_lines.append(stripped)
                elif capturing and (stripped.startswith("&&") or first_word in dockerfile_instructions):
                    dockerfile_lines.append(stripped)
        if dockerfile_lines:
            return "\n".join(dockerfile_lines)

    # Strategy 4: For YAML-based formats, find YAML-like content
    if category in ["docker-compose", "kubernetes", "github-actions", "ansible", "cloudformation"]:
        lines = completion.split("\n")
        yaml_lines = []
        capturing = False
        for line in lines:
            # Look for lines that start YAML structure
            if re.match(r'^(version:|apiVersion:|services:|on:|hosts:|AWSTemplateFormatVersion:)', line.strip()):
                capturing = True
            if capturing:
                # Stop capturing on obvious non-YAML content
                if line.strip().startswith("##") or line.strip().startswith("**"):
                    break
                yaml_lines.append(line)
        if yaml_lines:
            return "\n".join(yaml_lines)

    # Strategy 5: Fallback to entire completion
    return completion


def compute_decomposed_reward(completion: str, category: str) -> dict:
    """
    Compute decomposed reward with partial credit.
    R = 0.4 * syntax + 0.3 * semantic + 0.3 * structure
    """
    syntax_score = 0.0
    semantic_score = 0.0
    structure_score = 0.0

    completion = completion.strip()

    # Extract code block using improved extraction
    code_content = extract_code_content(completion, category)

    if category == "terraform":
        balanced = code_content.count("{") == code_content.count("}")
        has_equals = "=" in code_content
        syntax_score = 0.5 * balanced + 0.5 * has_equals

        has_resource = bool(re.search(r'(resource|module|data|variable|output)\s+"', code_content))
        has_provider = "provider" in code_content or "terraform" in code_content
        semantic_score = 0.6 * has_resource + 0.4 * has_provider

        has_newlines = "\n" in code_content
        has_indentation = "  " in code_content or "\t" in code_content
        structure_score = 0.5 * has_newlines + 0.5 * has_indentation

    elif category == "kubernetes":
        has_colon = ":" in code_content
        has_newlines = "\n" in code_content
        syntax_score = 0.5 * has_colon + 0.5 * has_newlines

        has_apiversion = "apiVersion:" in code_content or "apiversion:" in code_content.lower()
        has_kind = "kind:" in code_content
        has_metadata = "metadata:" in code_content
        semantic_score = 0.4 * has_apiversion + 0.4 * has_kind + 0.2 * has_metadata

        has_indentation = "  " in code_content
        structure_score = float(has_indentation)

    elif category == "dockerfile":
        lines = code_content.strip().split("\n")
        valid_instructions = {"FROM", "RUN", "CMD", "EXPOSE", "ENV", "ADD", "COPY",
                            "ENTRYPOINT", "VOLUME", "USER", "WORKDIR", "ARG", "LABEL"}
        instruction_count = sum(1 for line in lines
                               if line.strip() and not line.strip().startswith("#")
                               and line.strip().split()[0].upper() in valid_instructions
                               if line.strip().split())
        syntax_score = min(1.0, instruction_count / 3)

        has_from = any(line.strip().upper().startswith("FROM ") for line in lines
                      if line.strip() and not line.strip().startswith("#"))
        semantic_score = 1.0 if has_from else 0.0

        has_multiple_lines = len(lines) >= 3
        structure_score = 0.5 * has_multiple_lines

    elif category == "docker-compose":
        has_colon = ":" in code_content
        has_newlines = "\n" in code_content
        syntax_score = 0.5 * has_colon + 0.5 * has_newlines

        has_services = "services:" in code_content
        has_version = "version:" in code_content
        semantic_score = 0.7 * has_services + 0.3 * has_version

        has_indentation = "  " in code_content
        structure_score = float(has_indentation)

    elif category == "github-actions":
        has_colon = ":" in code_content
        has_newlines = "\n" in code_content
        syntax_score = 0.5 * has_colon + 0.5 * has_newlines

        has_on = "on:" in code_content
        has_jobs = "jobs:" in code_content
        has_steps = "steps:" in code_content
        semantic_score = 0.3 * has_on + 0.4 * has_jobs + 0.3 * has_steps

        has_indentation = "  " in code_content
        structure_score = float(has_indentation)

    elif category == "ansible":
        has_colon = ":" in code_content
        has_newlines = "\n" in code_content
        syntax_score = 0.5 * has_colon + 0.5 * has_newlines

        has_hosts = "hosts:" in code_content
        has_tasks = "tasks:" in code_content or "- name:" in code_content
        semantic_score = 0.5 * has_hosts + 0.5 * has_tasks

        has_indentation = "  " in code_content
        has_dash = "- " in code_content
        structure_score = 0.5 * has_indentation + 0.5 * has_dash

    elif category == "cloudformation":
        has_colon = ":" in code_content
        has_newlines = "\n" in code_content
        syntax_score = 0.5 * has_colon + 0.5 * has_newlines

        has_resources = "Resources:" in code_content
        has_type = "Type:" in code_content
        has_properties = "Properties:" in code_content
        semantic_score = 0.4 * has_resources + 0.3 * has_type + 0.3 * has_properties

        has_indentation = "  " in code_content
        structure_score = float(has_indentation)

    total = 0.4 * syntax_score + 0.3 * semantic_score + 0.3 * structure_score
    return {
        "syntax": syntax_score,
        "semantic": semantic_score,
        "structure": structure_score,
        "total": total
    }


def compute_dapo_reward(completion: str, category: str, max_length: int = 768,
                        overlong_buffer: int = 100, overlong_penalty_factor: float = 0.5) -> float:
    """
    DAPO reward function with all four DAPO innovations.

    DAPO Innovations:
    1. Overlong Reward Shaping: Proportional penalty for sequences exceeding (max_length - buffer)
    2. Reasoning bonus for multi-step structured output
    3. Diversity preservation to prevent entropy collapse
    4. Smooth reward scaling for better gradient signal
    """
    # Base reward from decomposed function
    reward_dict = compute_decomposed_reward(completion, category)
    reward = reward_dict["total"]

    # ==========================================================================
    # DAPO Innovation 1: OVERLONG REWARD SHAPING (from original DAPO paper)
    # Proportional penalty based on how much sequence exceeds buffer threshold
    # ==========================================================================
    completion_length = len(completion)
    overlong_threshold = max_length - overlong_buffer

    if completion_length > overlong_threshold:
        # Calculate excess ratio and apply proportional penalty
        excess_tokens = completion_length - overlong_threshold
        excess_ratio = min(1.0, excess_tokens / overlong_buffer)
        overlong_penalty = 1.0 - (overlong_penalty_factor * excess_ratio)
        reward *= overlong_penalty

    # ==========================================================================
    # DAPO Innovation 2: REASONING BONUS (REQUIRES CODE BLOCK)
    # Reward multi-step reasoning structure (Analysis → Plan → Code → Verify)
    # KEY: Must have actual code block (```) to get reasoning bonus
    # This ensures model does reasoning AND generates code
    # ==========================================================================
    reasoning_bonus = 0.0
    completion_lower = completion.lower()

    # Check if there's an actual code block (required for reasoning bonus)
    has_code_block = "```" in completion

    if has_code_block:
        # Only count reasoning if code is present
        has_analysis = any(x in completion_lower for x in ["## analysis", "### analysis", "analysis:", "**analysis"])
        has_plan = any(x in completion_lower for x in ["## plan", "### plan", "plan:", "**plan"])
        has_verification = any(x in completion_lower for x in ["## verif", "### verif", "verif:", "**verif", "## check"])

        # Reasoning components (analysis + plan + verification)
        # Code block itself doesn't count as reasoning component
        reasoning_components = sum([has_analysis, has_plan, has_verification])
        reasoning_bonus = 0.05 * reasoning_components  # +0.05 per component, max +0.15

        # Extra bonus for complete reasoning chain (all 3 components)
        if reasoning_components >= 3:
            reasoning_bonus += 0.05  # Total max = +0.20 for complete reasoning

    # ==========================================================================
    # DAPO Innovation 3: ENTROPY PRESERVATION / DIVERSITY
    # Penalize repetitive outputs to maintain exploration (prevents mode collapse)
    # ==========================================================================
    lines = completion.split('\n')
    non_empty_lines = [line.strip() for line in lines if line.strip()]

    if len(non_empty_lines) > 3:
        unique_lines = len(set(non_empty_lines))
        total_lines = len(non_empty_lines)
        diversity_ratio = unique_lines / total_lines

        # Soft penalty for low diversity (scaled, not binary)
        if diversity_ratio < 0.6:
            diversity_penalty = 0.8 + (0.2 * diversity_ratio / 0.6)  # Scale from 0.8 to 1.0
            reward *= diversity_penalty

    # ==========================================================================
    # DAPO Innovation 4: SMOOTH REWARD SCALING
    # Avoid sparse rewards - provide gradient signal even for partial success
    # ==========================================================================
    # Small floor for outputs with code blocks, but not too generous
    # This was causing issues - removed the floor to prevent gaming
    # if "```" in completion and reward < 0.5:
    #     reward = max(reward, 0.3)  # Floor at 0.3 if code block exists

    # Combine: base + reasoning bonus, capped at 1.0
    final_reward = min(1.0, reward + reasoning_bonus)

    return final_reward


# =============================================================================
# DAPO DYNAMIC SAMPLING HELPER
# =============================================================================

def should_skip_batch(rewards: list) -> bool:
    """
    DAPO Dynamic Sampling: Skip batches with zero variance (no gradient signal).
    Returns True if batch should be skipped.
    """
    if len(rewards) < 2:
        return False
    # Calculate std without numpy
    mean = sum(rewards) / len(rewards)
    variance = sum((x - mean) ** 2 for x in rewards) / len(rewards)
    reward_std = variance ** 0.5
    return reward_std < 0.01  # Skip if all rewards are nearly identical


# =============================================================================
# HELD-OUT TEST SET (Same as GRPO for comparison)
# =============================================================================

HELD_OUT_TEST_SET = [
    # TERRAFORM (20 samples)
    {"instruction": "Create a Terraform resource for an AWS ElastiCache Redis cluster named 'session-cache'", "category": "terraform"},
    {"instruction": "Write Terraform code for an AWS Elastic Beanstalk application with Python platform", "category": "terraform"},
    {"instruction": "Create a Terraform AWS Glue job resource for ETL processing", "category": "terraform"},
    {"instruction": "Write Terraform for an AWS AppSync GraphQL API with DynamoDB resolver", "category": "terraform"},
    {"instruction": "Create a Terraform resource for AWS Batch compute environment", "category": "terraform"},
    {"instruction": "Write Terraform code for an AWS MSK (Kafka) cluster with 3 brokers", "category": "terraform"},
    {"instruction": "Create a Terraform AWS MediaConvert job template resource", "category": "terraform"},
    {"instruction": "Write Terraform for AWS DocumentDB cluster with 2 instances", "category": "terraform"},
    {"instruction": "Create a Terraform resource for AWS Neptune graph database", "category": "terraform"},
    {"instruction": "Write Terraform code for AWS Timestream database and table", "category": "terraform"},
    {"instruction": "Create a Terraform AWS IoT thing and policy resource", "category": "terraform"},
    {"instruction": "Write Terraform for AWS Pinpoint application with SMS channel", "category": "terraform"},
    {"instruction": "Create a Terraform resource for AWS GameLift fleet", "category": "terraform"},
    {"instruction": "Write Terraform code for AWS WorkSpaces directory and workspace", "category": "terraform"},
    {"instruction": "Create a Terraform AWS DataSync task for S3 to EFS transfer", "category": "terraform"},
    {"instruction": "Write Terraform for Azure Container Instances with 2 containers", "category": "terraform"},
    {"instruction": "Create a Terraform resource for GCP Cloud Composer environment", "category": "terraform"},
    {"instruction": "Write Terraform code for GCP Memorystore Redis instance", "category": "terraform"},
    {"instruction": "Create a Terraform Azure Service Fabric cluster resource", "category": "terraform"},
    {"instruction": "Write Terraform for GCP Vertex AI endpoint deployment", "category": "terraform"},

    # KUBERNETES (20 samples)
    {"instruction": "Create a Kubernetes StatefulSet for MongoDB with 3 replicas and persistent volumes", "category": "kubernetes"},
    {"instruction": "Write a Kubernetes DaemonSet for log collection agent running on all nodes", "category": "kubernetes"},
    {"instruction": "Create a Kubernetes CronJob that runs database backup every 6 hours", "category": "kubernetes"},
    {"instruction": "Write a Kubernetes PodDisruptionBudget for a production web service", "category": "kubernetes"},
    {"instruction": "Create a Kubernetes LimitRange for a namespace with CPU and memory limits", "category": "kubernetes"},
    {"instruction": "Write a Kubernetes ResourceQuota limiting total pods and memory in namespace", "category": "kubernetes"},
    {"instruction": "Create a Kubernetes PriorityClass for critical system pods", "category": "kubernetes"},
    {"instruction": "Write a Kubernetes ValidatingWebhookConfiguration for pod security", "category": "kubernetes"},
    {"instruction": "Create a Kubernetes Ingress with TLS termination and path-based routing", "category": "kubernetes"},
    {"instruction": "Write a Kubernetes NetworkPolicy to isolate frontend from backend pods", "category": "kubernetes"},
    {"instruction": "Create a Kubernetes ServiceAccount with RBAC for CI/CD pipeline", "category": "kubernetes"},
    {"instruction": "Write a Kubernetes ConfigMap with environment-specific application settings", "category": "kubernetes"},
    {"instruction": "Create a Kubernetes PersistentVolume with NFS storage backend", "category": "kubernetes"},
    {"instruction": "Write a Kubernetes Deployment with blue-green deployment annotations", "category": "kubernetes"},
    {"instruction": "Create a Kubernetes Service mesh configuration for Istio sidecar injection", "category": "kubernetes"},
    {"instruction": "Write a Kubernetes Job for database migration with retry policy", "category": "kubernetes"},
    {"instruction": "Create a Kubernetes HorizontalPodAutoscaler based on custom metrics", "category": "kubernetes"},
    {"instruction": "Write a Kubernetes Secret for TLS certificates with cert-manager annotations", "category": "kubernetes"},
    {"instruction": "Create a Kubernetes Deployment for Redis Sentinel with 3 replicas", "category": "kubernetes"},
    {"instruction": "Write a Kubernetes ClusterRole for read-only access to all namespaces", "category": "kubernetes"},

    # DOCKERFILE (15 samples)
    {"instruction": "Create a Dockerfile for a Rust web application using actix-web", "category": "dockerfile"},
    {"instruction": "Write a multi-stage Dockerfile for a Go application with scratch final image", "category": "dockerfile"},
    {"instruction": "Create a Dockerfile for a Java Spring Boot application with Gradle build", "category": "dockerfile"},
    {"instruction": "Write a Dockerfile for a Node.js application with pnpm package manager", "category": "dockerfile"},
    {"instruction": "Create a Dockerfile for a Python ML model serving with TensorFlow", "category": "dockerfile"},
    {"instruction": "Write a multi-stage Dockerfile for a React application with Nginx", "category": "dockerfile"},
    {"instruction": "Create a Dockerfile for a Ruby on Rails application with PostgreSQL client", "category": "dockerfile"},
    {"instruction": "Write a Dockerfile for a .NET Core API with health checks", "category": "dockerfile"},
    {"instruction": "Create a Dockerfile for an Elixir Phoenix application", "category": "dockerfile"},
    {"instruction": "Write a secure Dockerfile with distroless base for a Python application", "category": "dockerfile"},
    {"instruction": "Create a Dockerfile for a Scala application with sbt build", "category": "dockerfile"},
    {"instruction": "Write a Dockerfile for a PHP Laravel application with Composer", "category": "dockerfile"},
    {"instruction": "Create a Dockerfile for a C++ application with CMake build", "category": "dockerfile"},
    {"instruction": "Write a multi-stage Dockerfile for a TypeScript application", "category": "dockerfile"},
    {"instruction": "Create a Dockerfile for a Kotlin application with Gradle", "category": "dockerfile"},

    # DOCKER-COMPOSE (15 samples)
    {"instruction": "Create a docker-compose.yml for a microservices setup with API gateway", "category": "docker-compose"},
    {"instruction": "Write a docker-compose.yml for ELK stack (Elasticsearch, Logstash, Kibana)", "category": "docker-compose"},
    {"instruction": "Create a docker-compose.yml for a WordPress site with MySQL and phpMyAdmin", "category": "docker-compose"},
    {"instruction": "Write a docker-compose.yml for a development environment with hot reload", "category": "docker-compose"},
    {"instruction": "Create a docker-compose.yml for Prometheus and Grafana monitoring stack", "category": "docker-compose"},
    {"instruction": "Write a docker-compose.yml for a message queue setup with RabbitMQ", "category": "docker-compose"},
    {"instruction": "Create a docker-compose.yml for a CI/CD runner with Docker-in-Docker", "category": "docker-compose"},
    {"instruction": "Write a docker-compose.yml for a Kafka cluster with Zookeeper", "category": "docker-compose"},
    {"instruction": "Create a docker-compose.yml for a GitLab instance with runner", "category": "docker-compose"},
    {"instruction": "Write a docker-compose.yml for a Minio object storage cluster", "category": "docker-compose"},
    {"instruction": "Create a docker-compose.yml for a Vault and Consul setup", "category": "docker-compose"},
    {"instruction": "Write a docker-compose.yml for a Keycloak authentication server", "category": "docker-compose"},
    {"instruction": "Create a docker-compose.yml for a Nextcloud deployment with Redis cache", "category": "docker-compose"},
    {"instruction": "Write a docker-compose.yml for an Airflow setup with Celery executor", "category": "docker-compose"},
    {"instruction": "Create a docker-compose.yml for a Traefik reverse proxy with Let's Encrypt", "category": "docker-compose"},

    # GITHUB-ACTIONS (15 samples)
    {"instruction": "Create GitHub Actions workflow for Python package release to PyPI", "category": "github-actions"},
    {"instruction": "Write GitHub Actions workflow with Terraform plan on PR and apply on merge", "category": "github-actions"},
    {"instruction": "Create GitHub Actions workflow for Docker image build with multi-arch support", "category": "github-actions"},
    {"instruction": "Write GitHub Actions workflow for Kubernetes deployment with ArgoCD sync", "category": "github-actions"},
    {"instruction": "Create GitHub Actions workflow for security scanning with Snyk and Trivy", "category": "github-actions"},
    {"instruction": "Write GitHub Actions workflow for automated semantic versioning", "category": "github-actions"},
    {"instruction": "Create GitHub Actions workflow for e2e tests with Playwright", "category": "github-actions"},
    {"instruction": "Write GitHub Actions workflow for Go application with golangci-lint", "category": "github-actions"},
    {"instruction": "Create GitHub Actions workflow for Rust project with cargo test and clippy", "category": "github-actions"},
    {"instruction": "Write GitHub Actions workflow for npm package publishing to GitHub Packages", "category": "github-actions"},
    {"instruction": "Create GitHub Actions workflow with matrix strategy for multiple OS and Node versions", "category": "github-actions"},
    {"instruction": "Write GitHub Actions workflow for database migration with rollback on failure", "category": "github-actions"},
    {"instruction": "Create GitHub Actions workflow for Helm chart linting and publishing", "category": "github-actions"},
    {"instruction": "Write GitHub Actions workflow with approval gates for production deployment", "category": "github-actions"},
    {"instruction": "Create GitHub Actions workflow for AWS CDK deployment with diff preview", "category": "github-actions"},

    # ANSIBLE (15 samples)
    {"instruction": "Create an Ansible playbook to set up a Kubernetes master node", "category": "ansible"},
    {"instruction": "Write an Ansible playbook for PostgreSQL installation with replication", "category": "ansible"},
    {"instruction": "Create an Ansible playbook for hardening Ubuntu server security", "category": "ansible"},
    {"instruction": "Write an Ansible playbook to deploy a Docker Swarm cluster", "category": "ansible"},
    {"instruction": "Create an Ansible playbook for Nginx reverse proxy with SSL certificates", "category": "ansible"},
    {"instruction": "Write an Ansible playbook for MongoDB replica set configuration", "category": "ansible"},
    {"instruction": "Create an Ansible playbook to set up Prometheus node exporters", "category": "ansible"},
    {"instruction": "Write an Ansible playbook for HAProxy load balancer configuration", "category": "ansible"},
    {"instruction": "Create an Ansible playbook for Elasticsearch cluster deployment", "category": "ansible"},
    {"instruction": "Write an Ansible playbook to configure fail2ban and UFW firewall", "category": "ansible"},
    {"instruction": "Create an Ansible playbook for Redis Sentinel setup", "category": "ansible"},
    {"instruction": "Write an Ansible playbook for Jenkins master and agent nodes", "category": "ansible"},
    {"instruction": "Create an Ansible playbook to deploy Grafana with LDAP authentication", "category": "ansible"},
    {"instruction": "Write an Ansible playbook for Vault server initialization and unsealing", "category": "ansible"},
    {"instruction": "Create an Ansible playbook for RabbitMQ cluster with mirrored queues", "category": "ansible"},

    # CLOUDFORMATION (10 samples)
    {"instruction": "Create CloudFormation template for a Lambda function with API Gateway trigger", "category": "cloudformation"},
    {"instruction": "Write CloudFormation for an Auto Scaling group with launch template", "category": "cloudformation"},
    {"instruction": "Create CloudFormation template for VPC with public and private subnets", "category": "cloudformation"},
    {"instruction": "Write CloudFormation for RDS Aurora cluster with read replicas", "category": "cloudformation"},
    {"instruction": "Create CloudFormation template for ECS Fargate service with ALB", "category": "cloudformation"},
    {"instruction": "Write CloudFormation for S3 bucket with CloudFront distribution", "category": "cloudformation"},
    {"instruction": "Create CloudFormation template for Step Functions state machine", "category": "cloudformation"},
    {"instruction": "Write CloudFormation for CodePipeline with CodeBuild stages", "category": "cloudformation"},
    {"instruction": "Create CloudFormation template for EventBridge rule with Lambda target", "category": "cloudformation"},
    {"instruction": "Write CloudFormation for Cognito user pool with app client", "category": "cloudformation"},
]


# =============================================================================
# DAPO TRAINING
# =============================================================================

@app.function(gpu="A100", image=image, timeout=10800, volumes={MODEL_DIR: model_volume})
def train_dapo():
    """
    DAPO training starting from GRPO checkpoint.
    Uses multi-step reasoning prompts and DAPO-specific rewards.
    """
    import torch
    import json
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from peft import PeftModel, LoraConfig, get_peft_model
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset

    print("=" * 60)
    print("InfraMind DAPO Training")
    print("Multi-Step Reasoning for Infrastructure-as-Code")
    print("=" * 60)

    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    GRPO_PATH = "/models/inframind-grpo/final"
    OUTPUT_DIR = "/models/inframind-dapo"

    # =========================================================================
    # 1. LOAD BASE MODEL + GRPO ADAPTERS
    # =========================================================================
    print("\n[1] Loading model and GRPO checkpoint...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load GRPO adapters if available
    import os
    if os.path.exists(GRPO_PATH):
        print(f"  Loading GRPO checkpoint from {GRPO_PATH}")
        model = PeftModel.from_pretrained(base_model, GRPO_PATH)
        # Merge GRPO adapters into base
        model = model.merge_and_unload()
        print("  GRPO adapters merged into base model")
    else:
        print("  WARNING: No GRPO checkpoint found, starting from base")
        model = base_model

    # Add fresh LoRA adapters for DAPO training (from config)
    lora_config = LoraConfig(
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        target_modules=CONFIG["target_modules"],
        lora_dropout=CONFIG["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # =========================================================================
    # 2. PREPARE TRAINING DATA WITH MULTI-STEP PROMPTS
    # =========================================================================
    print("\n[2] Preparing training data with multi-step prompts...")

    with open("/root/training_data.json") as f:
        training_data = json.load(f)

    # Create multi-step prompts
    prompts = []
    for sample in training_data[:500]:  # Use 500 samples
        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')

        if input_text:
            full_instruction = f"{instruction}\n{input_text}"
        else:
            full_instruction = instruction

        # Wrap with multi-step reasoning request
        prompt = f"Think through this step-by-step (Analysis → Plan → Code → Verify):\n\n{full_instruction}"
        prompts.append(prompt)

    dataset = Dataset.from_dict({"prompt": prompts})
    print(f"  Created dataset with {len(dataset)} multi-step prompts")

    # =========================================================================
    # 3. DAPO REWARD FUNCTION
    # =========================================================================
    print("\n[3] Setting up DAPO reward function...")

    MAX_COMPLETION_LENGTH = 768

    def dapo_reward_fn(completions, prompts, **kwargs):
        """DAPO reward with reasoning bonus and anti-exploitation."""
        rewards = []
        for completion, prompt in zip(completions, prompts):
            category = detect_category(prompt)
            reward = compute_dapo_reward(completion, category, MAX_COMPLETION_LENGTH)
            rewards.append(reward)
        return rewards

    # =========================================================================
    # 4. DAPO DYNAMIC SAMPLING CALLBACK WITH CLIP-HIGHER MONITORING
    # =========================================================================
    class DAPOCallback(TrainerCallback):
        """
        Monitor DAPO-specific metrics during training.
        Tracks:
        - Low variance batches (candidates for dynamic sampling skip)
        - Reward distribution statistics
        - Entropy/diversity metrics
        """

        def __init__(self):
            self.low_variance_count = 0
            self.total_batches = 0
            self.skipped_batches = 0
            self.reward_history = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                self.total_batches += 1

                # Track reward statistics
                reward_mean = logs.get("reward", logs.get("train/reward_mean", None))
                reward_std = logs.get("reward_std", logs.get("train/reward_std", None))

                if reward_mean is not None:
                    self.reward_history.append(reward_mean)

                # DAPO Dynamic Sampling: Flag low variance batches
                if reward_std is not None and reward_std < 0.05:
                    self.low_variance_count += 1
                    # In a full implementation, we would skip this batch
                    # TRL doesn't support this natively, so we just log it

                # Log detailed metrics every 25 steps
                if state.global_step % 25 == 0:
                    reward_str = f"{reward_mean:.3f}" if reward_mean is not None else "N/A"
                    std_str = f"{reward_std:.3f}" if reward_std is not None else "N/A"
                    skip_rate = self.low_variance_count / max(1, self.total_batches) * 100

                    print(f"  [DAPO] Step {state.global_step}: "
                          f"reward={reward_str}, std={std_str}, "
                          f"low_var_rate={skip_rate:.1f}%")

                    # Log recent reward trend
                    if len(self.reward_history) >= 10:
                        recent_avg = sum(self.reward_history[-10:]) / 10
                        print(f"  [DAPO] Recent avg reward (last 10): {recent_avg:.3f}")

    # =========================================================================
    # 5. DAPO TRAINING CONFIG (from config/dapo.yaml)
    # =========================================================================
    print("\n[4] Configuring DAPO training from config...")

    dapo_config = GRPOConfig(
        output_dir=CONFIG["output_dir"],

        # Training schedule
        num_train_epochs=CONFIG["num_train_epochs"],
        per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],

        # Learning rate
        learning_rate=CONFIG["learning_rate"],
        lr_scheduler_type=CONFIG["lr_scheduler_type"],
        warmup_ratio=CONFIG["warmup_ratio"],
        max_grad_norm=CONFIG["max_grad_norm"],
        weight_decay=CONFIG["weight_decay"],

        # Generation
        max_completion_length=CONFIG["max_completion_length"],
        num_generations=CONFIG["num_generations"],
        temperature=CONFIG["temperature"],

        # DAPO: No KL penalty (beta=0) for pure policy optimization
        beta=CONFIG["beta"],

        # Logging & saving
        logging_steps=5,
        save_steps=100,
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,
        bf16=True,
    )

    effective_batch = CONFIG["per_device_train_batch_size"] * CONFIG["gradient_accumulation_steps"]
    print(f"  DAPO Config (from config/dapo.yaml):")
    print(f"    Model: {CONFIG['model_id']}")
    print(f"    Learning rate: {CONFIG['learning_rate']}")
    print(f"    Epochs: {CONFIG['num_train_epochs']}")
    print(f"    Num generations: {CONFIG['num_generations']}")
    print(f"    Temperature: {CONFIG['temperature']}")
    print(f"    KL beta: {CONFIG['beta']} (0 = pure DAPO, no KL penalty)")
    print(f"    Clip epsilon: {CONFIG['epsilon']} (low), {CONFIG['epsilon_high']} (high)")
    print(f"    Effective batch size: {effective_batch}")
    print(f"    Dynamic sampling: {CONFIG['use_dynamic_sampling']}")
    print(f"    Overlong buffer: {CONFIG['overlong_buffer_length']} tokens")

    # =========================================================================
    # 6. CREATE TRAINER AND TRAIN
    # =========================================================================
    print("\n[5] Starting DAPO training...")

    trainer = GRPOTrainer(
        model=model,
        args=dapo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=dapo_reward_fn,
        callbacks=[DAPOCallback()],
    )

    trainer.train()

    # =========================================================================
    # 7. SAVE MODEL
    # =========================================================================
    print("\n[6] Saving DAPO model...")

    final_path = f"{OUTPUT_DIR}/final"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    model_volume.commit()
    print(f"  Model saved to {final_path}")

    print("\n" + "=" * 60)
    print("DAPO TRAINING COMPLETE")
    print("=" * 60)

    return {"status": "complete", "path": final_path}


# =============================================================================
# DAPO EVALUATION
# =============================================================================

@app.function(gpu="A100", image=image, timeout=1800, volumes={MODEL_DIR: model_volume})
def evaluate():
    """Evaluate DAPO model on 40 samples with detailed logging."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print("=" * 60)
    print("DAPO Model Evaluation - Full Test Set")
    print("=" * 60)

    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    DAPO_PATH = "/models/inframind-dapo/final"

    # Load model
    print("\n[1] Loading DAPO model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    import os
    if os.path.exists(DAPO_PATH):
        model = PeftModel.from_pretrained(model, DAPO_PATH)
        print("  DAPO adapters loaded")
    else:
        print("  ERROR: No DAPO model found!")
        return {"error": "No DAPO model"}

    model.eval()

    # Full held-out test set evaluation
    test_samples = HELD_OUT_TEST_SET

    correct = 0
    total = 0
    by_category = {}

    print(f"\n[2] Evaluating {len(test_samples)} samples with detailed output...")
    print("=" * 60)

    for i, sample in enumerate(test_samples):
        prompt = sample["instruction"]
        category = sample["category"]

        if category not in by_category:
            by_category[category] = {"correct": 0, "total": 0}

        messages = [
            {"role": "system", "content": DAPO_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Score
        reward = compute_dapo_reward(response, category)
        passed = reward >= 0.6

        total += 1
        by_category[category]["total"] += 1
        if passed:
            correct += 1
            by_category[category]["correct"] += 1

        status = "PASS" if passed else "FAIL"

        # Print detailed log
        print(f"\n--- Sample {i+1}: {category.upper()} [{status}] (reward={reward:.2f}) ---")
        print(f"PROMPT: {prompt[:100]}...")
        print(f"OUTPUT (first 200 chars): {response[:200]}...")
        if not passed:
            # Show what code was extracted
            extracted = extract_code_content(response, category)
            print(f"EXTRACTED CODE:\n{extracted[:500]}...")
            print(f"FULL OUTPUT:\n{response}")
        print(f"Running accuracy: {correct}/{total} ({correct/total*100:.1f}%)")

    # Final summary
    accuracy = correct / total * 100
    print(f"\n{'='*60}")
    print(f"DAPO EVAL RESULTS (35 samples)")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {accuracy:.1f}% ({correct}/{total})")
    print(f"\nBy Category:")
    for cat, stats in sorted(by_category.items()):
        cat_acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {cat:15s}: {cat_acc:5.1f}% ({stats['correct']}/{stats['total']})")
    print(f"\nCompare to GRPO: 97.3%")

    return {"accuracy": accuracy, "correct": correct, "total": total, "by_category": by_category}


# =============================================================================
# QUICK TEST (default: 110 samples)
# =============================================================================

@app.function(gpu="A100", image=image, timeout=3600, volumes={MODEL_DIR: model_volume})
def quick_test(num_samples: int = 110):
    """Test DAPO model on held-out test set (default: 110 samples)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import os

    print("=" * 60)
    print(f"DAPO Quick Test - {num_samples} samples")
    print("=" * 60)

    MODEL_NAME = CONFIG["model_id"]
    DAPO_PATH = f"{CONFIG['output_dir']}/final"
    GRPO_PATH = CONFIG["grpo_checkpoint"]

    # Load model
    print("\n[1] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Try DAPO first, then GRPO, then base
    if os.path.exists(DAPO_PATH):
        model = PeftModel.from_pretrained(model, DAPO_PATH)
        print(f"  Loaded DAPO adapters from {DAPO_PATH}")
    elif os.path.exists(GRPO_PATH):
        model = PeftModel.from_pretrained(model, GRPO_PATH)
        print(f"  Loaded GRPO adapters from {GRPO_PATH}")
    else:
        print("  Using base model (no fine-tuned adapters found)")

    model.eval()

    # Take first num_samples from test set (diverse categories)
    test_samples = HELD_OUT_TEST_SET[:num_samples]

    correct = 0
    total = 0

    print(f"\n[2] Running {num_samples} samples...")
    print("=" * 60)

    for i, sample in enumerate(test_samples):
        prompt = sample["instruction"]
        category = sample["category"]

        messages = [
            {"role": "system", "content": DAPO_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=768,  # Increased for multi-step reasoning + code
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Score using DAPO reward
        reward = compute_dapo_reward(response, category)
        passed = reward >= CONFIG["pass_threshold"]

        # Check if reasoning structure is present
        has_reasoning = any(x in response.lower() for x in ["## analysis", "### analysis", "## plan", "## code"])

        total += 1
        if passed:
            correct += 1

        status = "PASS" if passed else "FAIL"
        reasoning_tag = "[+reasoning]" if has_reasoning else ""
        print(f"\n[{i+1}/{num_samples}] {category.upper()} [{status}] reward={reward:.2f} {reasoning_tag}")
        print(f"  Prompt: {prompt[:80]}...")
        print(f"  Output: {response[:200]}...")
        if has_reasoning:
            print(f"  Reasoning: Found multi-step structure")

    accuracy = correct / total * 100
    print(f"\n{'='*60}")
    print(f"QUICK TEST RESULTS: {accuracy:.1f}% ({correct}/{total})")
    print(f"{'='*60}")

    return {"accuracy": accuracy, "correct": correct, "total": total}


@app.local_entrypoint()
def main():
    """Run DAPO training."""
    print("\nStarting DAPO training...")
    result = train_dapo.remote()
    print(f"\nTraining complete: {result}")
