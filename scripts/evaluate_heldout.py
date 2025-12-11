#!/usr/bin/env python3
"""
Held-out Test Set Evaluation for InfraMind SFT Model

Uses FRESH test prompts that were NOT in the training data.
No data leakage - these are completely new prompts.
"""

import json
import re
import requests
import argparse
from pathlib import Path
from typing import Optional


# =============================================================================
# FRESH TEST PROMPTS - NOT IN TRAINING DATA
# =============================================================================

HELD_OUT_TEST_SET = [
    # ========== TERRAFORM (20 samples) ==========
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

    # ========== KUBERNETES (20 samples) ==========
    {"instruction": "Create a Kubernetes StatefulSet for MongoDB with 3 replicas and persistent volumes", "category": "kubernetes"},
    {"instruction": "Write a Kubernetes DaemonSet for log collection agent running on all nodes", "category": "kubernetes"},
    {"instruction": "Create a Kubernetes CronJob that runs database backup every 6 hours", "category": "kubernetes"},
    {"instruction": "Write a Kubernetes PodDisruptionBudget for a production web service", "category": "kubernetes"},
    {"instruction": "Create a Kubernetes LimitRange for a namespace with CPU and memory limits", "category": "kubernetes"},
    {"instruction": "Write a Kubernetes ResourceQuota limiting total pods and memory in namespace", "category": "kubernetes"},
    {"instruction": "Create a Kubernetes PriorityClass for critical system workloads", "category": "kubernetes"},
    {"instruction": "Write a Kubernetes PodSecurityPolicy restricting privileged containers", "category": "kubernetes"},
    {"instruction": "Create a Kubernetes Vertical Pod Autoscaler for automatic resource adjustment", "category": "kubernetes"},
    {"instruction": "Write a Kubernetes Affinity rule to spread pods across availability zones", "category": "kubernetes"},
    {"instruction": "Create a Kubernetes EndpointSlice for external service integration", "category": "kubernetes"},
    {"instruction": "Write a Kubernetes CSIDriver configuration for EBS volumes", "category": "kubernetes"},
    {"instruction": "Create a Kubernetes RuntimeClass for gVisor sandbox containers", "category": "kubernetes"},
    {"instruction": "Write a Kubernetes MutatingWebhookConfiguration for pod injection", "category": "kubernetes"},
    {"instruction": "Create a Kubernetes ServiceMonitor for Prometheus scraping", "category": "kubernetes"},
    {"instruction": "Write a Kubernetes PodTemplate for use in batch job creation", "category": "kubernetes"},
    {"instruction": "Create a Kubernetes ClusterIssuer for Let's Encrypt certificates", "category": "kubernetes"},
    {"instruction": "Write a Kubernetes Gateway API HTTPRoute for traffic routing", "category": "kubernetes"},
    {"instruction": "Create a Kubernetes VolumeSnapshot for PVC backup", "category": "kubernetes"},
    {"instruction": "Write a Kubernetes StorageClass for high-IOPS SSD volumes", "category": "kubernetes"},

    # ========== DOCKERFILE (15 samples) ==========
    {"instruction": "Create a Dockerfile for a Rust application with multi-stage build", "category": "dockerfile"},
    {"instruction": "Write a Dockerfile for an Elixir Phoenix application with releases", "category": "dockerfile"},
    {"instruction": "Create a Dockerfile for a .NET 8 API with trimmed self-contained publish", "category": "dockerfile"},
    {"instruction": "Write a Dockerfile for Ruby on Rails with asset precompilation", "category": "dockerfile"},
    {"instruction": "Create a Dockerfile for a Scala SBT project with native-image", "category": "dockerfile"},
    {"instruction": "Write a Dockerfile for PHP Laravel with Octane and Swoole", "category": "dockerfile"},
    {"instruction": "Create a Dockerfile for a Haskell Stack project", "category": "dockerfile"},
    {"instruction": "Write a Dockerfile for Kotlin Ktor application with Gradle", "category": "dockerfile"},
    {"instruction": "Create a Dockerfile for Julia scientific computing with GPU support", "category": "dockerfile"},
    {"instruction": "Write a Dockerfile for Perl Dancer2 web application", "category": "dockerfile"},
    {"instruction": "Create a Dockerfile for OCaml with opam package manager", "category": "dockerfile"},
    {"instruction": "Write a Dockerfile for Crystal web application with Kemal", "category": "dockerfile"},
    {"instruction": "Create a Dockerfile for Nim application with nimble", "category": "dockerfile"},
    {"instruction": "Write a Dockerfile for Zig application compilation", "category": "dockerfile"},
    {"instruction": "Create a Dockerfile for Clojure Leiningen project", "category": "dockerfile"},

    # ========== DOCKER-COMPOSE (15 samples) ==========
    {"instruction": "Create a docker-compose.yml for Mattermost with PostgreSQL and object storage", "category": "docker-compose"},
    {"instruction": "Write a docker-compose.yml for Apache Kafka with Zookeeper and UI", "category": "docker-compose"},
    {"instruction": "Create a docker-compose.yml for GitLab CE with runner and registry", "category": "docker-compose"},
    {"instruction": "Write a docker-compose.yml for Airflow with Redis and PostgreSQL", "category": "docker-compose"},
    {"instruction": "Create a docker-compose.yml for Temporal workflow engine with UI", "category": "docker-compose"},
    {"instruction": "Write a docker-compose.yml for Minio S3-compatible object storage cluster", "category": "docker-compose"},
    {"instruction": "Create a docker-compose.yml for Kong API Gateway with PostgreSQL", "category": "docker-compose"},
    {"instruction": "Write a docker-compose.yml for HashiCorp Vault with Consul backend", "category": "docker-compose"},
    {"instruction": "Create a docker-compose.yml for Apache Druid analytics database", "category": "docker-compose"},
    {"instruction": "Write a docker-compose.yml for Jaeger distributed tracing", "category": "docker-compose"},
    {"instruction": "Create a docker-compose.yml for Keycloak identity provider with PostgreSQL", "category": "docker-compose"},
    {"instruction": "Write a docker-compose.yml for Superset with Redis and PostgreSQL", "category": "docker-compose"},
    {"instruction": "Create a docker-compose.yml for InfluxDB with Telegraf and Chronograf", "category": "docker-compose"},
    {"instruction": "Write a docker-compose.yml for RabbitMQ cluster with management UI", "category": "docker-compose"},
    {"instruction": "Create a docker-compose.yml for Sentry error tracking platform", "category": "docker-compose"},

    # ========== ANSIBLE (15 samples) ==========
    {"instruction": "Create an Ansible playbook to configure HAProxy load balancer with SSL termination", "category": "ansible"},
    {"instruction": "Write an Ansible role for installing and configuring TimescaleDB", "category": "ansible"},
    {"instruction": "Create an Ansible playbook for deploying Hashicorp Nomad cluster", "category": "ansible"},
    {"instruction": "Write an Ansible role to setup CockroachDB distributed database", "category": "ansible"},
    {"instruction": "Create an Ansible playbook for configuring WireGuard VPN server", "category": "ansible"},
    {"instruction": "Write an Ansible role for deploying Apache Cassandra cluster", "category": "ansible"},
    {"instruction": "Create an Ansible playbook to setup Ceph storage cluster", "category": "ansible"},
    {"instruction": "Write an Ansible role for installing Traefik reverse proxy", "category": "ansible"},
    {"instruction": "Create an Ansible playbook for deploying MinIO distributed mode", "category": "ansible"},
    {"instruction": "Write an Ansible role to configure Patroni for PostgreSQL HA", "category": "ansible"},
    {"instruction": "Create an Ansible playbook for setting up Netdata monitoring agents", "category": "ansible"},
    {"instruction": "Write an Ansible role for deploying ClickHouse analytics database", "category": "ansible"},
    {"instruction": "Create an Ansible playbook to install Consul service mesh", "category": "ansible"},
    {"instruction": "Write an Ansible role for configuring Envoy proxy", "category": "ansible"},
    {"instruction": "Create an Ansible playbook for deploying ScyllaDB cluster", "category": "ansible"},

    # ========== GITHUB-ACTIONS (15 samples) ==========
    {"instruction": "Create a GitHub Actions workflow for Rust project with cargo test and clippy", "category": "github-actions"},
    {"instruction": "Write a GitHub Actions workflow for iOS app with TestFlight deployment", "category": "github-actions"},
    {"instruction": "Create a GitHub Actions workflow for Android app with Play Store release", "category": "github-actions"},
    {"instruction": "Write a GitHub Actions workflow for Flutter app with multiple platform builds", "category": "github-actions"},
    {"instruction": "Create a GitHub Actions workflow for Terraform plan and apply with approval", "category": "github-actions"},
    {"instruction": "Write a GitHub Actions workflow for serverless framework deployment", "category": "github-actions"},
    {"instruction": "Create a GitHub Actions workflow for Kubernetes manifest validation and deploy", "category": "github-actions"},
    {"instruction": "Write a GitHub Actions workflow for monorepo with affected project detection", "category": "github-actions"},
    {"instruction": "Create a GitHub Actions workflow for Helm chart linting and packaging", "category": "github-actions"},
    {"instruction": "Write a GitHub Actions workflow for security scanning with CodeQL and Snyk", "category": "github-actions"},
    {"instruction": "Create a GitHub Actions workflow for database migration with Flyway", "category": "github-actions"},
    {"instruction": "Write a GitHub Actions workflow for Pulumi infrastructure deployment", "category": "github-actions"},
    {"instruction": "Create a GitHub Actions workflow for Deno project with deploy to Deno Deploy", "category": "github-actions"},
    {"instruction": "Write a GitHub Actions workflow for Electron app with auto-update releases", "category": "github-actions"},
    {"instruction": "Create a GitHub Actions workflow for performance benchmarking with comparison", "category": "github-actions"},

    # ========== CLOUDFORMATION (10 samples) ==========
    {"instruction": "Create a CloudFormation template for AWS Step Functions state machine", "category": "cloudformation"},
    {"instruction": "Write a CloudFormation template for AWS Cognito user pool with MFA", "category": "cloudformation"},
    {"instruction": "Create a CloudFormation template for AWS AppRunner service", "category": "cloudformation"},
    {"instruction": "Write a CloudFormation template for AWS Transfer Family SFTP server", "category": "cloudformation"},
    {"instruction": "Create a CloudFormation template for AWS Amplify app with custom domain", "category": "cloudformation"},
    {"instruction": "Write a CloudFormation template for AWS EventBridge with scheduled rules", "category": "cloudformation"},
    {"instruction": "Create a CloudFormation template for AWS CodePipeline with manual approval", "category": "cloudformation"},
    {"instruction": "Write a CloudFormation template for AWS FSx for Lustre filesystem", "category": "cloudformation"},
    {"instruction": "Create a CloudFormation template for AWS Kinesis Data Firehose to S3", "category": "cloudformation"},
    {"instruction": "Write a CloudFormation template for AWS Elemental MediaLive channel", "category": "cloudformation"},
]


# =============================================================================
# VALIDATORS - Same as before
# =============================================================================

def validate_terraform(output: str) -> dict:
    """Validate Terraform output"""
    checks = {
        "has_resource_or_module": bool(re.search(r'(resource|module|data)\s+"', output)),
        "balanced_braces": output.count("{") == output.count("}"),
        "has_equals": "=" in output,
        "no_prose_start": not output.strip().startswith(("Here", "This", "I ", "The ", "To ")),
    }
    passed = all(checks.values())
    return {"passed": passed, "checks": checks}


def validate_kubernetes(output: str) -> dict:
    """Validate Kubernetes YAML output"""
    checks = {
        "has_apiversion": "apiVersion:" in output or "apiversion:" in output.lower(),
        "has_kind": "kind:" in output,
        "has_metadata_or_spec": "metadata:" in output or "spec:" in output,
        "no_prose_start": not output.strip().startswith(("Here", "This", "I ", "The ", "To ")),
    }
    passed = all(checks.values())
    return {"passed": passed, "checks": checks}


def validate_dockerfile(output: str) -> dict:
    """Validate Dockerfile output"""
    lines = output.strip().split("\n")
    has_from = any(line.strip().upper().startswith("FROM ") for line in lines
                   if line.strip() and not line.strip().startswith("#"))

    valid_instructions = {"FROM", "RUN", "CMD", "EXPOSE", "ENV", "ADD", "COPY",
                          "ENTRYPOINT", "VOLUME", "USER", "WORKDIR", "ARG", "LABEL"}
    instruction_count = 0
    for line in lines:
        if line.strip() and not line.strip().startswith("#"):
            parts = line.strip().split()
            if parts and parts[0].upper() in valid_instructions:
                instruction_count += 1

    checks = {
        "has_from": has_from,
        "has_instructions": instruction_count >= 2,
        "no_prose_start": not output.strip().startswith(("Here", "This", "I ", "The ", "To ")),
    }
    passed = all(checks.values())
    return {"passed": passed, "checks": checks}


def validate_docker_compose(output: str) -> dict:
    """Validate Docker Compose output"""
    checks = {
        "has_services": "services:" in output,
        "has_image_or_build": "image:" in output or "build:" in output,
        "valid_yaml_structure": ":" in output and not output.strip().startswith(("Here", "This")),
    }
    passed = all(checks.values())
    return {"passed": passed, "checks": checks}


def validate_ansible(output: str) -> dict:
    """Validate Ansible output"""
    checks = {
        "has_tasks_or_hosts": "tasks:" in output or "hosts:" in output or "- name:" in output,
        "has_yaml_structure": ":" in output,
        "no_prose_start": not output.strip().startswith(("Here", "This", "I ", "The ", "To ")),
    }
    passed = all(checks.values())
    return {"passed": passed, "checks": checks}


def validate_github_actions(output: str) -> dict:
    """Validate GitHub Actions output"""
    checks = {
        "has_jobs_or_steps": "jobs:" in output or "steps:" in output,
        "has_runs_on_or_uses": "runs-on:" in output or "uses:" in output or "run:" in output,
        "no_prose_start": not output.strip().startswith(("Here", "This", "I ", "The ", "To ")),
    }
    passed = all(checks.values())
    return {"passed": passed, "checks": checks}


def validate_cloudformation(output: str) -> dict:
    """Validate CloudFormation output"""
    checks = {
        "has_resources_or_template": "Resources:" in output or "AWSTemplateFormatVersion" in output,
        "has_type": "Type:" in output,
        "no_prose_start": not output.strip().startswith(("Here", "This", "I ", "The ", "To ")),
    }
    passed = all(checks.values())
    return {"passed": passed, "checks": checks}


VALIDATORS = {
    "terraform": validate_terraform,
    "kubernetes": validate_kubernetes,
    "dockerfile": validate_dockerfile,
    "docker-compose": validate_docker_compose,
    "ansible": validate_ansible,
    "github-actions": validate_github_actions,
    "cloudformation": validate_cloudformation,
}


# =============================================================================
# MODEL CALLING
# =============================================================================

def call_model(endpoint: str, prompt: str, max_tokens: int = 500) -> str:
    """Call the deployed model endpoint"""
    try:
        resp = requests.post(
            endpoint,
            json={"prompt": prompt, "max_tokens": max_tokens},
            timeout=60
        )
        if resp.status_code == 200:
            return resp.json().get("response", "")
        else:
            return f"ERROR: {resp.status_code}"
    except Exception as e:
        return f"ERROR: {e}"


# =============================================================================
# EVALUATION
# =============================================================================

def run_evaluation(endpoint: str, test_set: list, max_tokens: int = 500) -> dict:
    """Run evaluation on held-out test set"""
    results = {
        "total": len(test_set),
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "by_category": {},
        "samples": []
    }

    for i, sample in enumerate(test_set):
        instruction = sample["instruction"]
        category = sample.get("category", "unknown")

        print(f"[{i+1}/{len(test_set)}] {category}: {instruction[:50]}...")

        # Call model
        output = call_model(endpoint, instruction, max_tokens)

        # Check for errors
        if output.startswith("ERROR:"):
            results["errors"] += 1
            print(f"  ⚠ {output}")
            continue

        # Validate
        validator = VALIDATORS.get(category, validate_terraform)
        result = validator(output)

        # Track results
        if result["passed"]:
            results["passed"] += 1
        else:
            results["failed"] += 1

        # Track by category
        if category not in results["by_category"]:
            results["by_category"][category] = {"total": 0, "passed": 0}
        results["by_category"][category]["total"] += 1
        if result["passed"]:
            results["by_category"][category]["passed"] += 1

        # Store sample result
        results["samples"].append({
            "instruction": instruction,
            "category": category,
            "output": output[:500],  # Truncate for storage
            "passed": result["passed"],
            "checks": result["checks"]
        })

        # Print result
        status = "✓" if result["passed"] else "✗"
        print(f"  {status} {result['checks']}")

    # Calculate accuracy (excluding errors)
    valid_total = results["total"] - results["errors"]
    results["accuracy"] = results["passed"] / valid_total if valid_total > 0 else 0

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="https://tideon--inframind-sft-inframindmodel-generate.modal.run")
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--output", default="results/heldout_eval.json")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples (0=all)")
    args = parser.parse_args()

    print("=" * 60)
    print("InfraMind Held-Out Test Set Evaluation")
    print("=" * 60)
    print("\n⚠ Using FRESH test prompts NOT in training data")

    # Use held-out test set
    test_set = HELD_OUT_TEST_SET
    if args.limit > 0:
        test_set = test_set[:args.limit]

    print(f"\n[1] Test set: {len(test_set)} samples")

    # Show category distribution
    cats = {}
    for s in test_set:
        c = s.get("category", "unknown")
        cats[c] = cats.get(c, 0) + 1
    print(f"  Categories: {cats}")

    # Run evaluation
    print(f"\n[2] Running evaluation against {args.endpoint}...")
    results = run_evaluation(args.endpoint, test_set, args.max_tokens)

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS (HELD-OUT TEST SET)")
    print("=" * 60)
    print(f"\nOverall: {results['passed']}/{results['total']-results['errors']} = {results['accuracy']*100:.1f}%")
    if results['errors'] > 0:
        print(f"Errors: {results['errors']} (excluded from accuracy)")

    print("\nBy Category:")
    for cat, data in sorted(results["by_category"].items()):
        acc = data["passed"] / data["total"] * 100 if data["total"] > 0 else 0
        print(f"  {cat}: {data['passed']}/{data['total']} = {acc:.1f}%")

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved to {args.output}")


if __name__ == "__main__":
    main()
