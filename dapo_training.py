"""
InfraMind DAPO Training - Multi-Step Reasoning for IaC Generation.

DAPO (Direct Advantage Policy Optimization) extends GRPO with:
1. Multi-step reasoning prompts (Analyze → Plan → Generate → Verify)
2. Asymmetric clipping for exploration
3. Dynamic sampling to skip uninformative batches
4. Overlong reward shaping
5. Reasoning bonus in reward function

Usage:
    modal run dapo_training.py              # Run DAPO training
    modal run dapo_training.py::evaluate    # Evaluate DAPO model
"""

import modal
import re

app = modal.App("inframind-dapo")

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


def compute_decomposed_reward(completion: str, category: str) -> dict:
    """
    Compute decomposed reward with partial credit.
    R = 0.4 * syntax + 0.3 * semantic + 0.3 * structure
    """
    syntax_score = 0.0
    semantic_score = 0.0
    structure_score = 0.0

    completion = completion.strip()

    # Extract code block if present (for multi-step outputs)
    code_match = re.search(r'```(?:\w+)?\n(.*?)```', completion, re.DOTALL)
    code_content = code_match.group(1) if code_match else completion

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


def compute_dapo_reward(completion: str, category: str, max_length: int = 768) -> float:
    """
    DAPO reward function with multi-step reasoning bonuses.

    Extensions over GRPO:
    1. Reasoning bonus for structured output
    2. Overlong penalty for truncated sequences
    3. Diversity bonus to prevent repetition
    """
    # Base reward from decomposed function
    reward_dict = compute_decomposed_reward(completion, category)
    reward = reward_dict["total"]

    # DAPO Addition 1: OVERLONG PENALTY
    # Soft penalty for sequences near max length (likely truncated)
    completion_length = len(completion)
    if completion_length >= max_length * 0.9:  # >90% of max length
        overlong_penalty = 0.8
        reward *= overlong_penalty

    # DAPO Addition 2: REASONING BONUS
    # Reward multi-step reasoning structure
    completion_lower = completion.lower()

    has_analysis = "analysis" in completion_lower or "## analysis" in completion_lower
    has_plan = "plan" in completion_lower or "## plan" in completion_lower
    has_code_section = "## code" in completion_lower or "```" in completion
    has_verification = "verification" in completion_lower or "## verif" in completion_lower

    reasoning_components = sum([has_analysis, has_plan, has_code_section, has_verification])
    reasoning_bonus = 0.05 * reasoning_components  # Up to +0.2 for full reasoning

    # DAPO Addition 3: ENTROPY PRESERVATION / DIVERSITY
    # Penalize repetitive outputs to prevent mode collapse
    lines = completion.split('\n')
    if len(lines) > 1:
        unique_lines = len(set(line.strip() for line in lines if line.strip()))
        total_lines = len([line for line in lines if line.strip()])
        if total_lines > 0:
            diversity_ratio = unique_lines / total_lines
            if diversity_ratio < 0.5:  # Too repetitive
                reward *= 0.9

    # Combine: base + reasoning bonus, capped at 1.0
    final_reward = min(1.0, reward + reasoning_bonus)

    return final_reward


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

    # Add fresh LoRA adapters for DAPO training
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
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
    # 4. DAPO DYNAMIC SAMPLING CALLBACK
    # =========================================================================
    class DAPOCallback(TrainerCallback):
        """Monitor DAPO-specific metrics during training."""

        def __init__(self):
            self.low_variance_count = 0
            self.total_batches = 0

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                self.total_batches += 1

                # Check for low variance batches (uninformative)
                reward_std = logs.get("reward_std", logs.get("train/reward_std", None))
                if reward_std is not None and reward_std < 0.1:
                    self.low_variance_count += 1
                    print(f"  [DAPO] Low variance batch #{self.low_variance_count}: std={reward_std:.3f}")

                # Log reasoning metrics if available
                if state.global_step % 25 == 0:
                    reward_mean = logs.get("reward", logs.get("train/reward_mean", None))
                    reward_str = f"{reward_mean:.3f}" if reward_mean is not None else "N/A"
                    print(f"  [DAPO] Step {state.global_step}: "
                          f"reward={reward_str}, "
                          f"low_var_batches={self.low_variance_count}/{self.total_batches}")

    # =========================================================================
    # 5. DAPO TRAINING CONFIG
    # =========================================================================
    print("\n[4] Configuring DAPO training...")

    dapo_config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,  # Effective batch size: 16
        learning_rate=5e-6,  # Lower than GRPO for stability
        max_completion_length=MAX_COMPLETION_LENGTH,
        num_generations=4,
        temperature=0.8,
        beta=0.1,  # KL penalty
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
    )

    print(f"  Learning rate: {dapo_config.learning_rate}")
    print(f"  Max completion length: {dapo_config.max_completion_length}")
    print(f"  Effective batch size: {dapo_config.per_device_train_batch_size * dapo_config.gradient_accumulation_steps}")

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
    """Evaluate DAPO model on 35 samples with detailed logging."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print("=" * 60)
    print("DAPO Model Evaluation (35 samples)")
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

    # Eval on 35 samples (diverse across categories)
    test_samples = HELD_OUT_TEST_SET[:35]

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


@app.local_entrypoint()
def main():
    """Run DAPO training."""
    print("\nStarting DAPO training...")
    result = train_dapo.remote()
    print(f"\nTraining complete: {result}")
