#!/usr/bin/env python3
"""
Real IaC Scraper - 2,000 samples target (no Claude API, $0 budget)

Usage:
    export GITHUB_TOKEN="your-github-token"  # Optional but recommended
    python scripts/scrape_real_code.py --max-samples 2000 --output data/real_code_data.json
"""

import os
import re
import json
import asyncio
import random
import hashlib
import argparse
from pathlib import Path
from typing import Optional
import aiohttp

# =============================================================================
# GITHUB REPOS TO SCRAPE
# =============================================================================

REPOS = [
    # Terraform - AWS Official
    "terraform-aws-modules/terraform-aws-vpc",
    "terraform-aws-modules/terraform-aws-eks",
    "terraform-aws-modules/terraform-aws-rds",
    "terraform-aws-modules/terraform-aws-s3-bucket",
    "terraform-aws-modules/terraform-aws-lambda",
    "terraform-aws-modules/terraform-aws-iam",
    "terraform-aws-modules/terraform-aws-alb",
    "terraform-aws-modules/terraform-aws-security-group",
    "terraform-aws-modules/terraform-aws-ec2-instance",
    "terraform-aws-modules/terraform-aws-dynamodb-table",
    "terraform-aws-modules/terraform-aws-sqs",
    "terraform-aws-modules/terraform-aws-sns",
    "terraform-aws-modules/terraform-aws-kms",
    "terraform-aws-modules/terraform-aws-ecs",
    "terraform-aws-modules/terraform-aws-ecr",
    "terraform-aws-modules/terraform-aws-autoscaling",
    "terraform-aws-modules/terraform-aws-elb",
    "terraform-aws-modules/terraform-aws-cloudwatch",
    "terraform-aws-modules/terraform-aws-route53",
    "terraform-aws-modules/terraform-aws-acm",
    "terraform-aws-modules/terraform-aws-apigateway-v2",
    "terraform-aws-modules/terraform-aws-eventbridge",
    "terraform-aws-modules/terraform-aws-step-functions",
    "terraform-aws-modules/terraform-aws-redshift",
    "terraform-aws-modules/terraform-aws-elasticache",
    # GCP
    "terraform-google-modules/terraform-google-network",
    "terraform-google-modules/terraform-google-kubernetes-engine",
    "terraform-google-modules/terraform-google-sql-db",
    "terraform-google-modules/terraform-google-vm",
    "terraform-google-modules/terraform-google-cloud-storage",
    "terraform-google-modules/terraform-google-iam",
    "terraform-google-modules/terraform-google-project-factory",
    "terraform-google-modules/terraform-google-lb-http",
    # Azure
    "Azure/terraform-azurerm-aks",
    "Azure/terraform-azurerm-vnet",
    "Azure/terraform-azurerm-compute",
    "Azure/terraform-azurerm-storage-account",
    "Azure/terraform-azurerm-postgresql",
    # Kubernetes
    "kubernetes/examples",
    "kubernetes/website",
    "GoogleCloudPlatform/kubernetes-engine-samples",
    "GoogleCloudPlatform/microservices-demo",
    "microservices-demo/microservices-demo",
    "argoproj/argo-cd",
    "argoproj/argocd-example-apps",
    "argoproj/argo-workflows",
    "prometheus-operator/kube-prometheus",
    "cert-manager/cert-manager",
    "bitnami/charts",
    "istio/istio",
    "kubernetes-sigs/kustomize",
    "kubernetes-sigs/external-dns",
    # Docker
    "docker/awesome-compose",
    "docker/getting-started",
    "jessfraz/dockerfiles",
    "vimagick/dockerfiles",
    "docker-library/docs",
    # CI/CD
    "actions/starter-workflows",
    "actions/cache",
    "actions/checkout",
    "docker/build-push-action",
    "aws-actions/configure-aws-credentials",
    "actions/setup-node",
    "actions/setup-python",
    "actions/setup-go",
    "codecov/codecov-action",
    "github/super-linter",
    # Ansible
    "ansible/ansible-examples",
    "geerlingguy/ansible-for-devops",
    "geerlingguy/ansible-role-docker",
    "geerlingguy/ansible-role-nginx",
    "geerlingguy/ansible-role-mysql",
    "geerlingguy/ansible-role-java",
    "geerlingguy/ansible-role-postgresql",
    # Helm
    "prometheus-community/helm-charts",
    "grafana/helm-charts",
    "elastic/helm-charts",
    # CloudFormation
    "aws-cloudformation/aws-cloudformation-samples",
    "aws-samples/aws-cloudformation-templates",
    "aws-samples/serverless-patterns",
]

FILE_EXTENSIONS = [".tf", ".yaml", ".yml", "Dockerfile", ".conf"]

# =============================================================================
# TYPE DETECTION FROM CONTENT (not from repo)
# =============================================================================

def detect_type(content: str) -> Optional[str]:
    """Detect IaC type from code content"""
    # Terraform
    if re.search(r'\bresource\s+"[^"]+"\s+"[^"]+"', content) or \
       re.search(r'\bmodule\s+"[^"]+"', content) or \
       re.search(r'\bdata\s+"[^"]+"\s+"[^"]+"', content):
        return "terraform"

    # Kubernetes
    if 'apiVersion:' in content and 'kind:' in content:
        return "kubernetes"

    # Dockerfile
    if any(line.strip().upper().startswith('FROM ') for line in content.split('\n')
           if line.strip() and not line.strip().startswith('#')):
        return "dockerfile"

    # Docker Compose
    if 'services:' in content and ('image:' in content or 'build:' in content):
        return "docker-compose"

    # GitHub Actions
    if 'jobs:' in content and ('runs-on:' in content or 'steps:' in content):
        return "github-actions"

    # Ansible
    if ('hosts:' in content or 'tasks:' in content) and \
       ('- name:' in content or 'become:' in content or 'vars:' in content):
        return "ansible"

    # CloudFormation
    if 'AWSTemplateFormatVersion' in content or \
       (re.search(r'Resources:', content) and re.search(r'Type:\s*AWS::', content)):
        return "cloudformation"

    return None

# =============================================================================
# FALLBACK INSTRUCTION GENERATOR (regex-based, no API)
# =============================================================================

def generate_instruction(code: str, code_type: str) -> str:
    """Generate instruction from code using regex patterns"""

    if code_type == "terraform":
        # Extract resource type and name
        match = re.search(r'resource\s+"([^"]+)"\s+"([^"]+)"', code)
        if match:
            resource_type = match.group(1)
            resource_name = match.group(2)
            # Parse resource type
            parts = resource_type.split("_")
            provider = parts[0] if parts else "aws"
            resource = "_".join(parts[1:]) if len(parts) > 1 else resource_type
            return f"Create a {provider.upper()} {resource.replace('_', ' ')} resource named {resource_name}"

        match = re.search(r'module\s+"([^"]+)"', code)
        if match:
            return f"Create a Terraform module for {match.group(1).replace('_', ' ')}"

        match = re.search(r'data\s+"([^"]+)"\s+"([^"]+)"', code)
        if match:
            return f"Define a Terraform data source for {match.group(1)} named {match.group(2)}"

    elif code_type == "kubernetes":
        kind_match = re.search(r'kind:\s*(\w+)', code)
        name_match = re.search(r'name:\s*([^\s\n]+)', code)
        ns_match = re.search(r'namespace:\s*([^\s\n]+)', code)

        kind = kind_match.group(1) if kind_match else "resource"
        name = name_match.group(1) if name_match else ""
        ns = ns_match.group(1) if ns_match else ""

        base = f"Create a Kubernetes {kind}"
        if name:
            base += f" named {name}"
        if ns and ns != "default":
            base += f" in namespace {ns}"
        return base

    elif code_type == "dockerfile":
        from_match = re.search(r'FROM\s+([^\s\n]+)', code, re.IGNORECASE)
        base_image = from_match.group(1) if from_match else "base"

        # Detect what the dockerfile does
        if re.search(r'npm|yarn|node', code, re.IGNORECASE):
            return f"Create a Dockerfile for a Node.js application based on {base_image}"
        elif re.search(r'pip|python|flask|django', code, re.IGNORECASE):
            return f"Create a Dockerfile for a Python application based on {base_image}"
        elif re.search(r'go\s+build|golang', code, re.IGNORECASE):
            return f"Create a Dockerfile for a Go application based on {base_image}"
        elif re.search(r'mvn|gradle|java', code, re.IGNORECASE):
            return f"Create a Dockerfile for a Java application based on {base_image}"
        else:
            return f"Create a Dockerfile based on {base_image}"

    elif code_type == "docker-compose":
        services = re.findall(r'^\s{2}(\w[\w-]*):', code, re.MULTILINE)
        if services:
            return f"Create a Docker Compose file with services: {', '.join(services[:5])}"
        return "Create a Docker Compose configuration"

    elif code_type == "github-actions":
        name_match = re.search(r'name:\s*([^\n]+)', code)
        on_match = re.search(r'on:\s*\[?([^\n\]]+)', code)

        name = name_match.group(1).strip() if name_match else "workflow"
        trigger = on_match.group(1).strip() if on_match else "push"

        return f"Create a GitHub Actions workflow named '{name}' triggered on {trigger}"

    elif code_type == "ansible":
        name_match = re.search(r'- name:\s*([^\n]+)', code)
        hosts_match = re.search(r'hosts:\s*([^\n]+)', code)

        task_name = name_match.group(1).strip() if name_match else "playbook"
        hosts = hosts_match.group(1).strip() if hosts_match else "all"

        return f"Create an Ansible playbook to {task_name.lower()} on {hosts}"

    elif code_type == "cloudformation":
        desc_match = re.search(r'Description:\s*([^\n]+)', code)
        if desc_match:
            return f"Create a CloudFormation template: {desc_match.group(1).strip()}"

        resources = re.findall(r'Type:\s*(AWS::[^\n]+)', code)
        if resources:
            return f"Create a CloudFormation template with {resources[0]}"

        return "Create an AWS CloudFormation template"

    return f"Create a {code_type} configuration"

# =============================================================================
# ASYNC GITHUB SCRAPER
# =============================================================================

class AsyncGitHubScraper:
    def __init__(self, concurrency: int = 20):
        self.file_semaphore = asyncio.Semaphore(concurrency)
        self.api_semaphore = asyncio.Semaphore(5)

        # GitHub token
        self.token = os.environ.get("GITHUB_TOKEN")
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "InfraMind-Scraper"
        }
        if self.token:
            self.headers["Authorization"] = f"token {self.token}"
            print("✓ GitHub token found")
        else:
            print("⚠ No GITHUB_TOKEN set - will hit rate limits quickly")
            print("  Get one at: https://github.com/settings/tokens")

        # Deduplication
        self.seen_hashes = set()

    def hash_code(self, code: str) -> str:
        """Hash first 500 chars for deduplication"""
        return hashlib.md5(code[:500].encode()).hexdigest()

    async def get_file(self, session: aiohttp.ClientSession, repo: str, path: str) -> Optional[str]:
        async with self.file_semaphore:
            for branch in ["main", "master"]:
                url = f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                        if resp.status == 200:
                            return await resp.text()
                except:
                    continue
            return None

    async def list_files(self, session: aiohttp.ClientSession, repo: str, path: str = "", depth: int = 0) -> list:
        if depth > 2:
            return []

        files = []
        url = f"https://api.github.com/repos/{repo}/contents/{path}"

        async with self.api_semaphore:
            try:
                async with session.get(url, headers=self.headers, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                    if resp.status == 403:
                        remaining = resp.headers.get("X-RateLimit-Remaining", "?")
                        print(f"    ⚠ Rate limited on {repo} (remaining: {remaining})")
                        return []
                    if resp.status != 200:
                        return []
                    contents = await resp.json()
            except Exception as e:
                return []

        if not isinstance(contents, list):
            return []

        for item in contents:
            name = item.get("name", "")
            if item["type"] == "file":
                for ext in FILE_EXTENSIONS:
                    if name.endswith(ext) or name == ext:
                        files.append(item["path"])
                        break
            elif item["type"] == "dir" and not name.startswith("."):
                subfiles = await self.list_files(session, repo, item["path"], depth + 1)
                files.extend(subfiles)

        return files

    def split_tf(self, content: str) -> list:
        """Split Terraform file into individual resources"""
        resources = []
        lines = content.split('\n')
        current = []
        braces = 0
        active = False

        for line in lines:
            if re.match(r'^(resource|module|data)\s+"', line.strip()):
                if current and active:
                    resources.append('\n'.join(current))
                current = [line]
                active = True
                braces = line.count('{') - line.count('}')
            elif active:
                current.append(line)
                braces += line.count('{') - line.count('}')
                if braces <= 0:
                    resources.append('\n'.join(current))
                    current = []
                    active = False

        if current:
            resources.append('\n'.join(current))

        return [r for r in resources if 50 < len(r) < 6000]

    def split_yaml_docs(self, content: str) -> list:
        """Split YAML into documents"""
        docs = content.split("---")
        return [d.strip() for d in docs if d.strip() and 50 < len(d.strip()) < 4000]

    async def scrape_repo(self, session: aiohttp.ClientSession, repo: str) -> list:
        samples = []
        try:
            files = await self.list_files(session, repo)
            if not files:
                return []

            print(f"  {repo}: {len(files)} files")
            random.shuffle(files)
            files = files[:30]  # Limit per repo

            tasks = [self.get_file(session, repo, f) for f in files]
            contents = await asyncio.gather(*tasks)

            for filepath, content in zip(files, contents):
                if not content or len(content) < 50:
                    continue

                # Split content into chunks
                chunks = []
                if filepath.endswith('.tf'):
                    chunks = self.split_tf(content)
                elif filepath.endswith(('.yaml', '.yml')):
                    chunks = self.split_yaml_docs(content)
                else:
                    chunks = [content] if len(content) < 6000 else []

                for chunk in chunks:
                    # Detect type from content
                    code_type = detect_type(chunk)
                    if not code_type:
                        continue

                    # Deduplicate
                    code_hash = self.hash_code(chunk)
                    if code_hash in self.seen_hashes:
                        continue
                    self.seen_hashes.add(code_hash)

                    samples.append({
                        "code": chunk,
                        "type": code_type,
                        "source": repo
                    })

        except Exception as e:
            print(f"    Error: {e}")

        return samples

# =============================================================================
# MAIN PIPELINE
# =============================================================================

async def run_pipeline(output_path: str, max_samples: int = 2000):
    print("=" * 60)
    print(f"Real IaC Scraper - Target: {max_samples} samples")
    print("=" * 60)

    checkpoint_path = f"{output_path}.checkpoint"

    # Load checkpoint if exists
    all_samples = []
    scraped_repos = set()
    if Path(checkpoint_path).exists():
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
            all_samples = checkpoint.get("samples", [])
            scraped_repos = set(checkpoint.get("scraped_repos", []))
            print(f"\n✓ Resuming from checkpoint: {len(all_samples)} samples")

    scraper = AsyncGitHubScraper(concurrency=20)

    # Restore seen hashes from checkpoint
    for s in all_samples:
        scraper.seen_hashes.add(scraper.hash_code(s["output"]))

    async with aiohttp.ClientSession() as session:
        print("\n[Phase 1] Scraping repos...")

        # Filter repos not yet scraped
        repos_to_scrape = [r for r in REPOS if r not in scraped_repos]
        random.shuffle(repos_to_scrape)

        for i, repo in enumerate(repos_to_scrape):
            if len(all_samples) >= max_samples:
                break

            samples = await scraper.scrape_repo(session, repo)
            scraped_repos.add(repo)

            # Generate instructions for samples
            for s in samples:
                instruction = generate_instruction(s["code"], s["type"])
                all_samples.append({
                    "instruction": instruction,
                    "input": "",
                    "output": s["code"],
                    "category": s["type"]
                })

            # Checkpoint every 500 samples
            if len(all_samples) % 500 < len(samples) and len(all_samples) >= 500:
                print(f"\n  💾 Checkpoint: {len(all_samples)} samples saved")
                with open(checkpoint_path, "w") as f:
                    json.dump({
                        "samples": all_samples,
                        "scraped_repos": list(scraped_repos)
                    }, f)

            # Progress
            print(f"    Total: {len(all_samples)} samples")

            # Small delay to be nice
            if i % 5 == 0:
                await asyncio.sleep(0.5)

    # Limit to max samples
    all_samples = all_samples[:max_samples]

    # Stats
    cats = {}
    for s in all_samples:
        cats[s["category"]] = cats.get(s["category"], 0) + 1

    print(f"\n[Results] {len(all_samples)} unique samples")
    for c, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {c}: {n}")

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_samples, f, indent=2)

    print(f"\n✓ Saved to {output_path}")

    # Clean up checkpoint
    if Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()
        print("✓ Checkpoint cleaned up")

    return all_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/real_code_data.json")
    parser.add_argument("--max-samples", type=int, default=2000)
    args = parser.parse_args()

    asyncio.run(run_pipeline(args.output, args.max_samples))


if __name__ == "__main__":
    main()
