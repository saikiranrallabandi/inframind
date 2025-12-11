"""
Evaluate GPT-4 on the InfraMind held-out test set.

Usage:
    python scripts/evaluate_gpt4.py

Requires:
    export OPENROUTER_API_KEY=sk-or-...

Cost estimate: ~$2-5 for 110 samples
"""

import os
import json
import re
from openai import OpenAI

# Same reward functions as GRPO training
def compute_decomposed_reward(completion: str, category: str) -> dict:
    """
    Compute decomposed reward with partial credit.
    R = 0.4 * syntax + 0.3 * semantic + 0.3 * structure
    """
    syntax_score = 0.0
    semantic_score = 0.0
    structure_score = 0.0

    completion = completion.strip()
    if completion.startswith(("Here", "This", "I ", "The ", "To ", "Below")):
        structure_score -= 0.5

    if category == "terraform":
        balanced = completion.count("{") == completion.count("}")
        has_equals = "=" in completion
        syntax_score = 0.5 * balanced + 0.5 * has_equals

        has_resource = bool(re.search(r'(resource|module|data|variable|output)\s+"', completion))
        has_provider = "provider" in completion or "terraform" in completion
        semantic_score = 0.6 * has_resource + 0.4 * has_provider

        has_newlines = "\n" in completion
        has_indentation = "  " in completion or "\t" in completion
        structure_score = max(0, structure_score + 0.5 * has_newlines + 0.5 * has_indentation)

    elif category == "kubernetes":
        has_colon = ":" in completion
        has_newlines = "\n" in completion
        syntax_score = 0.5 * has_colon + 0.5 * has_newlines

        has_apiversion = "apiVersion:" in completion or "apiversion:" in completion.lower()
        has_kind = "kind:" in completion
        has_metadata = "metadata:" in completion
        semantic_score = 0.4 * has_apiversion + 0.4 * has_kind + 0.2 * has_metadata

        has_indentation = "  " in completion
        structure_score = max(0, structure_score + has_indentation)

    elif category == "dockerfile":
        lines = completion.strip().split("\n")
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
        structure_score = max(0, structure_score + 0.5 * has_multiple_lines)

    elif category == "docker-compose":
        has_colon = ":" in completion
        has_newlines = "\n" in completion
        syntax_score = 0.5 * has_colon + 0.5 * has_newlines

        has_services = "services:" in completion
        has_version = "version:" in completion
        semantic_score = 0.7 * has_services + 0.3 * has_version

        has_indentation = "  " in completion
        structure_score = max(0, structure_score + has_indentation)

    elif category == "github-actions":
        has_colon = ":" in completion
        has_newlines = "\n" in completion
        syntax_score = 0.5 * has_colon + 0.5 * has_newlines

        has_on = "on:" in completion
        has_jobs = "jobs:" in completion
        has_steps = "steps:" in completion
        semantic_score = 0.3 * has_on + 0.4 * has_jobs + 0.3 * has_steps

        has_indentation = "  " in completion
        structure_score = max(0, structure_score + has_indentation)

    elif category == "ansible":
        has_colon = ":" in completion
        has_newlines = "\n" in completion
        syntax_score = 0.5 * has_colon + 0.5 * has_newlines

        has_hosts = "hosts:" in completion
        has_tasks = "tasks:" in completion or "- name:" in completion
        semantic_score = 0.5 * has_hosts + 0.5 * has_tasks

        has_indentation = "  " in completion
        has_dash = "- " in completion
        structure_score = max(0, structure_score + 0.5 * has_indentation + 0.5 * has_dash)

    elif category == "cloudformation":
        has_colon = ":" in completion
        has_newlines = "\n" in completion
        syntax_score = 0.5 * has_colon + 0.5 * has_newlines

        has_resources = "Resources:" in completion
        has_type = "Type:" in completion
        has_properties = "Properties:" in completion
        semantic_score = 0.4 * has_resources + 0.3 * has_type + 0.3 * has_properties

        has_indentation = "  " in completion
        structure_score = max(0, structure_score + has_indentation)

    total = 0.4 * syntax_score + 0.3 * semantic_score + 0.3 * structure_score
    return {
        "syntax": syntax_score,
        "semantic": semantic_score,
        "structure": structure_score,
        "total": total
    }


# 110 held-out test samples (same as GRPO evaluation)
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


def evaluate_gpt4(model: str = "openai/gpt-4o"):
    """Evaluate GPT-4 on the held-out test set via OpenRouter."""

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY")
    )

    # Setup logging
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model.replace("/", "_").replace("-", "_")
    log_file = f"eval_log_{model_name}_{timestamp}.txt"
    results_file = f"eval_results_{model_name}_{timestamp}.json"

    def log(msg):
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

    results = {
        "total": 0,
        "correct": 0,
        "by_category": {}
    }

    all_outputs = []

    log(f"\n{'='*60}")
    log(f"Evaluating {model} on {len(HELD_OUT_TEST_SET)} samples")
    log(f"Log file: {log_file}")
    log(f"Results file: {results_file}")
    log(f"{'='*60}\n")

    for i, sample in enumerate(HELD_OUT_TEST_SET):
        prompt = sample["instruction"]
        category = sample["category"]

        # Initialize category tracking
        if category not in results["by_category"]:
            results["by_category"][category] = {"total": 0, "correct": 0}

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an Infrastructure-as-Code expert. Generate correct, production-ready code. Output ONLY the code, no explanations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=512,
                temperature=0.7
            )

            completion = response.choices[0].message.content

            # Score the completion
            reward = compute_decomposed_reward(completion, category)
            passed = reward["total"] >= 0.6

            results["total"] += 1
            results["by_category"][category]["total"] += 1

            if passed:
                results["correct"] += 1
                results["by_category"][category]["correct"] += 1

            all_outputs.append({
                "prompt": prompt,
                "category": category,
                "output": completion,
                "reward": reward,
                "passed": passed
            })

            # Progress
            status = "PASS" if passed else "FAIL"
            log(f"[{i+1:3d}/110] {category:15s} | {status} | reward={reward['total']:.2f}")

            # Save intermediate results every 10 samples
            if (i + 1) % 10 == 0:
                with open(results_file, "w") as f:
                    json.dump({
                        "model": model,
                        "progress": f"{i+1}/110",
                        "accuracy": results["correct"] / results["total"] * 100 if results["total"] > 0 else 0,
                        "results": results,
                        "outputs": all_outputs
                    }, f, indent=2)

        except Exception as e:
            log(f"[{i+1:3d}/110] {category:15s} | ERROR: {e}")
            results["total"] += 1
            results["by_category"][category]["total"] += 1

    # Print summary
    log(f"\n{'='*60}")
    log(f"RESULTS: {model}")
    log(f"{'='*60}")

    accuracy = results["correct"] / results["total"] * 100 if results["total"] > 0 else 0
    log(f"\nOverall Accuracy: {accuracy:.1f}% ({results['correct']}/{results['total']})")

    log(f"\nBy Category:")
    for cat, stats in sorted(results["by_category"].items()):
        cat_acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        log(f"  {cat:15s}: {cat_acc:5.1f}% ({stats['correct']}/{stats['total']})")

    # Save final results
    with open(results_file, "w") as f:
        json.dump({
            "model": model,
            "accuracy": accuracy,
            "results": results,
            "outputs": all_outputs
        }, f, indent=2)

    log(f"\nLog saved to: {log_file}")
    log(f"Results saved to: {results_file}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-4o", help="Model to evaluate (openai/gpt-4o, openai/gpt-4-turbo)")
    args = parser.parse_args()

    evaluate_gpt4(args.model)
