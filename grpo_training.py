"""
InfraMind GRPO Training - Reinforcement Learning with Decomposed Rewards.

Standalone Modal app for GRPO training.

Usage:
    modal run grpo_training.py              # Run GRPO training
    modal run grpo_training.py::evaluate    # Evaluate GRPO model
"""

import modal
import re

app = modal.App("inframind-grpo")

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
        "huggingface_hub",
    )
    .add_local_file("data/real_code_2k.json", "/root/training_data.json")
)

# HuggingFace upload image (with hub library)
hf_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub", "transformers", "peft", "torch")
)


# =============================================================================
# REWARD FUNCTIONS
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
    # Initialize scores
    syntax_score = 0.0
    semantic_score = 0.0
    structure_score = 0.0

    # Remove leading prose
    completion = completion.strip()
    if completion.startswith(("Here", "This", "I ", "The ", "To ", "Below")):
        structure_score -= 0.5  # Penalty for prose

    if category == "terraform":
        # Syntax: balanced braces, has = signs
        balanced = completion.count("{") == completion.count("}")
        has_equals = "=" in completion
        syntax_score = 0.5 * balanced + 0.5 * has_equals

        # Semantic: has resource/module/data blocks
        has_resource = bool(re.search(r'(resource|module|data|variable|output)\s+"', completion))
        has_provider = "provider" in completion or "terraform" in completion
        semantic_score = 0.6 * has_resource + 0.4 * has_provider

        # Structure: proper formatting
        has_newlines = "\n" in completion
        has_indentation = "  " in completion or "\t" in completion
        structure_score = max(0, structure_score + 0.5 * has_newlines + 0.5 * has_indentation)

    elif category == "kubernetes":
        # Syntax: valid YAML structure
        has_colon = ":" in completion
        has_newlines = "\n" in completion
        syntax_score = 0.5 * has_colon + 0.5 * has_newlines

        # Semantic: K8s required fields
        has_apiversion = "apiVersion:" in completion or "apiversion:" in completion.lower()
        has_kind = "kind:" in completion
        has_metadata = "metadata:" in completion
        semantic_score = 0.4 * has_apiversion + 0.4 * has_kind + 0.2 * has_metadata

        # Structure: proper YAML indentation
        has_indentation = "  " in completion
        structure_score = max(0, structure_score + has_indentation)

    elif category == "dockerfile":
        # Syntax: valid Dockerfile instructions
        lines = completion.strip().split("\n")
        valid_instructions = {"FROM", "RUN", "CMD", "EXPOSE", "ENV", "ADD", "COPY",
                            "ENTRYPOINT", "VOLUME", "USER", "WORKDIR", "ARG", "LABEL"}
        instruction_count = sum(1 for line in lines
                               if line.strip() and not line.strip().startswith("#")
                               and line.strip().split()[0].upper() in valid_instructions
                               if line.strip().split())
        syntax_score = min(1.0, instruction_count / 3)

        # Semantic: has FROM instruction
        has_from = any(line.strip().upper().startswith("FROM ") for line in lines
                      if line.strip() and not line.strip().startswith("#"))
        semantic_score = 1.0 if has_from else 0.0

        # Structure: multi-line, proper ordering
        has_multiple_lines = len(lines) >= 3
        structure_score = max(0, structure_score + 0.5 * has_multiple_lines)

    elif category == "docker-compose":
        # Syntax: valid YAML
        has_colon = ":" in completion
        has_newlines = "\n" in completion
        syntax_score = 0.5 * has_colon + 0.5 * has_newlines

        # Semantic: has services
        has_services = "services:" in completion
        has_version = "version:" in completion
        semantic_score = 0.7 * has_services + 0.3 * has_version

        # Structure: proper indentation
        has_indentation = "  " in completion
        structure_score = max(0, structure_score + has_indentation)

    elif category == "github-actions":
        # Syntax: valid YAML
        has_colon = ":" in completion
        has_newlines = "\n" in completion
        syntax_score = 0.5 * has_colon + 0.5 * has_newlines

        # Semantic: GHA required fields
        has_on = "on:" in completion
        has_jobs = "jobs:" in completion
        has_steps = "steps:" in completion
        semantic_score = 0.3 * has_on + 0.4 * has_jobs + 0.3 * has_steps

        # Structure
        has_indentation = "  " in completion
        structure_score = max(0, structure_score + has_indentation)

    elif category == "ansible":
        # Syntax: valid YAML
        has_colon = ":" in completion
        has_dash = "- " in completion
        syntax_score = 0.5 * has_colon + 0.5 * has_dash

        # Semantic: Ansible structure
        has_hosts = "hosts:" in completion
        has_tasks = "tasks:" in completion or "- name:" in completion
        semantic_score = 0.5 * has_hosts + 0.5 * has_tasks

        # Structure
        has_indentation = "  " in completion
        structure_score = max(0, structure_score + has_indentation)

    elif category == "cloudformation":
        # Syntax: valid YAML/JSON
        has_colon = ":" in completion
        syntax_score = 1.0 if has_colon else 0.0

        # Semantic: CFN required sections
        has_resources = "Resources:" in completion or "resources:" in completion
        has_type = "Type:" in completion or "type:" in completion
        semantic_score = 0.6 * has_resources + 0.4 * has_type

        # Structure
        has_indentation = "  " in completion
        structure_score = max(0, structure_score + has_indentation)

    else:
        # Unknown category - basic checks
        has_content = len(completion) > 50
        has_structure = "\n" in completion
        syntax_score = 0.5 * has_content
        semantic_score = 0.5 * has_structure
        structure_score = 0.5

    # Clamp scores
    syntax_score = max(0, min(1, syntax_score))
    semantic_score = max(0, min(1, semantic_score))
    structure_score = max(0, min(1, structure_score))

    # Compute final reward
    total_reward = 0.4 * syntax_score + 0.3 * semantic_score + 0.3 * structure_score

    return {
        "total": total_reward,
        "syntax": syntax_score,
        "semantic": semantic_score,
        "structure": structure_score,
    }


def load_training_data():
    """Load training data from JSON."""
    import json
    import os

    paths = ["/root/training_data.json", "data/real_code_2k.json"]
    for path in paths:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return []


# =============================================================================
# HELD-OUT TEST SET (110 samples)
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
# GRPO TRAINING
# =============================================================================

@app.function(gpu="A100", image=image, timeout=7200, volumes={MODEL_DIR: model_volume})
def train_grpo():
    """
    GRPO training starting from SFT checkpoint.
    Uses decomposed rewards for IaC validation.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, LoraConfig, get_peft_model
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset

    print("=" * 60)
    print("InfraMind GRPO Training")
    print("=" * 60)

    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    SFT_PATH = "/models/inframind-sft/final"
    OUTPUT_DIR = "/models/inframind-grpo"

    # Step 1: Load SFT model
    print("\n[1/4] Loading SFT checkpoint...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    import os
    if os.path.exists(SFT_PATH):
        print(f"  Loading SFT adapters from: {SFT_PATH}")
        model = PeftModel.from_pretrained(model, SFT_PATH)
        model = model.merge_and_unload()
        print("  SFT adapters merged")
    else:
        print("  WARNING: No SFT checkpoint found, using base model")

    # Step 2: Add fresh LoRA for GRPO
    print("\n[2/4] Adding LoRA for GRPO...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Step 3: Prepare prompts
    print("\n[3/4] Preparing training prompts...")
    training_data = load_training_data()

    prompts = []
    for sample in training_data[:500]:
        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')
        prompt = f"{instruction}\n\n{input_text}".strip() if input_text else instruction
        prompts.append(prompt)

    prompt_dataset = Dataset.from_dict({"prompt": prompts})
    print(f"  Training prompts: {len(prompts)}")

    # Step 4: GRPO Training
    print("\n[4/4] Starting GRPO training...")

    def reward_fn(completions, prompts, **kwargs):
        """GRPO reward function."""
        rewards = []
        for completion, prompt in zip(completions, prompts):
            if isinstance(completion, list):
                completion = completion[0] if completion else ""
            comp_text = str(completion)
            prompt_text = str(prompt)
            category = detect_category(prompt_text)
            reward = compute_decomposed_reward(comp_text, category)
            rewards.append(reward["total"])
        return rewards

    grpo_config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_completion_length=512,
        num_generations=4,
        temperature=0.8,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        beta=0.1,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        processing_class=tokenizer,
        train_dataset=prompt_dataset,
        reward_funcs=reward_fn,
    )

    print("\nStarting training...")
    train_result = trainer.train()

    # Save
    print("\nSaving GRPO model...")
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    model_volume.commit()

    print(f"\nGRPO model saved to: {OUTPUT_DIR}/final")
    return {"status": "success", "output_dir": f"{OUTPUT_DIR}/final"}


# =============================================================================
# EVALUATION
# =============================================================================

@app.function(gpu="A100", image=image, timeout=3600, volumes={MODEL_DIR: model_volume})
def evaluate():
    """Evaluate GRPO model on held-out test set (110 samples)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import os

    print("=" * 60)
    print("Evaluating GRPO Model on Held-Out Test Set")
    print("=" * 60)

    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    GRPO_PATH = "/models/inframind-grpo/final"

    # Load model
    print("\nLoading GRPO model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    if os.path.exists(GRPO_PATH):
        model = PeftModel.from_pretrained(model, GRPO_PATH)
        print("  GRPO adapters loaded")
    else:
        print("  WARNING: No GRPO model found, using base!")

    model.eval()

    # Run evaluation
    results = {"correct": 0, "total": 0, "by_category": {}}

    print(f"\nEvaluating {len(HELD_OUT_TEST_SET)} samples...")
    for i, sample in enumerate(HELD_OUT_TEST_SET):
        prompt = sample["instruction"]
        category = sample["category"]

        # Generate
        messages = [
            {"role": "system", "content": "You are an Infrastructure-as-Code expert. Generate correct, production-ready code."},
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

        # Evaluate
        reward = compute_decomposed_reward(response, category)
        passed = reward["total"] >= 0.6

        results["total"] += 1
        if passed:
            results["correct"] += 1

        if category not in results["by_category"]:
            results["by_category"][category] = {"correct": 0, "total": 0}
        results["by_category"][category]["total"] += 1
        if passed:
            results["by_category"][category]["correct"] += 1

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(HELD_OUT_TEST_SET)}")

    # Calculate accuracy
    accuracy = results["correct"] / results["total"] * 100

    print(f"\n{'='*60}")
    print(f"GRPO Evaluation Results")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {accuracy:.1f}% ({results['correct']}/{results['total']})")
    print(f"\nBy Category:")
    for cat, stats in sorted(results["by_category"].items()):
        cat_acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {cat}: {cat_acc:.1f}% ({stats['correct']}/{stats['total']})")

    return {
        "accuracy": accuracy,
        "correct": results["correct"],
        "total": results["total"],
        "by_category": results["by_category"],
    }


# =============================================================================
# HUGGINGFACE UPLOAD
# =============================================================================

@app.function(
    image=hf_image,
    volumes={MODEL_DIR: model_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1800,
)
def upload_to_huggingface(repo_name: str = "inframind-0.5b-grpo"):
    """
    Upload GRPO model to HuggingFace Hub.

    Usage:
        modal run grpo_training.py::upload_to_huggingface
        modal run grpo_training.py::upload_to_huggingface --repo-name my-custom-name

    Prerequisites:
        1. Create HuggingFace token: https://huggingface.co/settings/tokens
        2. Create Modal secret:
           modal secret create huggingface-secret HUGGINGFACE_TOKEN=hf_xxxxx
    """
    import os
    from huggingface_hub import HfApi, create_repo

    print("=" * 60)
    print("Uploading GRPO Model to HuggingFace")
    print("=" * 60)

    GRPO_PATH = "/models/inframind-grpo/final"

    # Get HF token from environment (set by Modal secret)
    hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HUGGINGFACE_TOKEN not found. Create secret with: modal secret create huggingface-secret HUGGINGFACE_TOKEN=hf_xxxxx")

    # Check model exists
    if not os.path.exists(GRPO_PATH):
        raise FileNotFoundError(f"Model not found at {GRPO_PATH}. Run training first.")

    # List files to upload
    print(f"\nModel path: {GRPO_PATH}")
    files = os.listdir(GRPO_PATH)
    print(f"Files to upload: {files}")

    # Initialize API
    api = HfApi()

    # Get username
    user_info = api.whoami(token=hf_token)
    username = user_info["name"]
    full_repo_name = f"{username}/{repo_name}"

    print(f"\nUploading to: {full_repo_name}")

    # Create repo if it doesn't exist
    try:
        create_repo(
            repo_id=full_repo_name,
            token=hf_token,
            repo_type="model",
            exist_ok=True,
        )
        print(f"  Repository ready: https://huggingface.co/{full_repo_name}")
    except Exception as e:
        print(f"  Note: {e}")

    # Upload model files
    print("\nUploading model files...")
    api.upload_folder(
        folder_path=GRPO_PATH,
        repo_id=full_repo_name,
        repo_type="model",
        token=hf_token,
    )

    print(f"\n{'='*60}")
    print(f"SUCCESS! Model uploaded to:")
    print(f"https://huggingface.co/{full_repo_name}")
    print(f"{'='*60}")

    return {
        "status": "success",
        "repo": full_repo_name,
        "url": f"https://huggingface.co/{full_repo_name}",
    }


# =============================================================================
# MERGE AND UPLOAD (Full Model)
# =============================================================================

@app.function(
    gpu="A10G",
    image=image,
    volumes={MODEL_DIR: model_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
)
def merge_and_upload(repo_name: str = "inframind-0.5b-grpo"):
    """
    Merge LoRA adapter with base model and upload as complete model to HuggingFace.

    This makes the model easier to use (no need to load adapter separately) and
    ensures download tracking works properly.

    Usage:
        modal run grpo_training.py::merge_and_upload
    """
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from huggingface_hub import HfApi, create_repo

    print("=" * 60)
    print("Merging LoRA Adapter with Base Model")
    print("=" * 60)

    BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    ADAPTER_PATH = "/models/inframind-grpo/final"
    MERGED_PATH = "/models/inframind-grpo/merged"

    # Check adapter exists
    if not os.path.exists(ADAPTER_PATH):
        raise FileNotFoundError(f"Adapter not found at {ADAPTER_PATH}. Run training first.")

    # Step 1: Load base model
    print(f"\n[1/4] Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Step 2: Load LoRA adapter
    print(f"\n[2/4] Loading LoRA adapter: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    # Step 3: Merge and unload
    print("\n[3/4] Merging adapter into base model...")
    model = model.merge_and_unload()

    # Save merged model
    print(f"\n[4/4] Saving merged model to: {MERGED_PATH}")
    os.makedirs(MERGED_PATH, exist_ok=True)
    model.save_pretrained(MERGED_PATH, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_PATH)

    # Create config.json with model_type
    import json
    config_path = os.path.join(MERGED_PATH, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    # Ensure model_type is set
    if "model_type" not in config:
        config["model_type"] = "qwen2"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nMerged model files:")
    for f in os.listdir(MERGED_PATH):
        size = os.path.getsize(os.path.join(MERGED_PATH, f)) / (1024*1024)
        print(f"  {f}: {size:.1f} MB")

    # Upload to HuggingFace
    print("\n" + "=" * 60)
    print("Uploading Merged Model to HuggingFace")
    print("=" * 60)

    hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HUGGINGFACE_TOKEN not found")

    api = HfApi()
    user_info = api.whoami(token=hf_token)
    username = user_info["name"]
    full_repo_name = f"{username}/{repo_name}"

    print(f"\nUploading to: {full_repo_name}")

    # Create/update repo
    try:
        create_repo(
            repo_id=full_repo_name,
            token=hf_token,
            repo_type="model",
            exist_ok=True,
        )
    except Exception as e:
        print(f"  Note: {e}")

    # Upload merged model
    print("\nUploading merged model files...")
    api.upload_folder(
        folder_path=MERGED_PATH,
        repo_id=full_repo_name,
        repo_type="model",
        token=hf_token,
    )

    model_volume.commit()

    print(f"\n{'='*60}")
    print(f"SUCCESS! Merged model uploaded to:")
    print(f"https://huggingface.co/{full_repo_name}")
    print(f"\nUsers can now load with:")
    print(f'  model = AutoModelForCausalLM.from_pretrained("{full_repo_name}")')
    print(f"{'='*60}")

    return {
        "status": "success",
        "repo": full_repo_name,
        "url": f"https://huggingface.co/{full_repo_name}",
    }


@app.local_entrypoint()
def main():
    """Run GRPO training."""
    print("\nStarting InfraMind GRPO training on Modal (A100)...\n")
    result = train_grpo.remote()
    print(f"\nTraining complete: {result}")

    print("\nRunning evaluation...")
    eval_result = evaluate.remote()
    print(f"\nEvaluation complete!")
    print(f"Accuracy: {eval_result['accuracy']:.1f}%")
