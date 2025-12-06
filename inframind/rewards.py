"""InfraMind Reward Functions - Infrastructure-aware scoring"""
import re
from typing import Dict, Tuple

# Infrastructure keywords for validation
TF_RESOURCES = {"aws_instance", "aws_s3_bucket", "aws_vpc", "aws_subnet", "aws_security_group",
                "aws_lambda_function", "aws_iam_role", "aws_rds_cluster", "aws_eks_cluster",
                "aws_sagemaker", "aws_emr", "google_compute", "azurerm_machine_learning"}
K8S_KINDS = {"Deployment", "Service", "ConfigMap", "Secret", "Ingress", "StatefulSet", "CronJob",
             "Job", "Pod", "PersistentVolumeClaim", "ServiceAccount"}
DOCKER_KEYWORDS = {"FROM", "RUN", "COPY", "CMD", "ENTRYPOINT", "EXPOSE", "WORKDIR"}
MLOPS_KEYWORDS = {"mlflow", "kubeflow", "sagemaker", "vertex", "feast", "ray", "airflow", "prefect",
                  "seldon", "kserve", "triton", "torchserve", "tensorflow_serving", "vllm", "ollama",
                  "nvidia.com/gpu", "resources:", "limits:", "requests:", "gpu", "training", "inference",
                  "model", "pipeline", "dag", "experiment", "tracking", "serving", "endpoint"}


class IaCReward:
    """Reward calculator for IaC generation"""

    def __init__(self, alpha: float = 0.4, beta: float = 0.3, gamma: float = 0.3):
        self.alpha = alpha  # syntax weight
        self.beta = beta    # correctness weight
        self.gamma = gamma  # format weight

    def score(self, response: str, category: str) -> Tuple[float, Dict]:
        """Calculate reward: R = α*syntax + β*correctness + γ*format"""
        if category == "terraform":
            syntax, correctness, fmt = self._score_terraform(response)
        elif category == "kubernetes":
            syntax, correctness, fmt = self._score_kubernetes(response)
        elif category == "docker":
            syntax, correctness, fmt = self._score_docker(response)
        elif category == "cicd":
            syntax, correctness, fmt = self._score_cicd(response)
        elif category == "mlops":
            syntax, correctness, fmt = self._score_mlops(response)
        else:
            syntax, correctness, fmt = 0.0, 0.0, 0.0

        total = self.alpha * syntax + self.beta * correctness + self.gamma * fmt
        return total, {"syntax": syntax, "correctness": correctness, "format": fmt}

    def _score_terraform(self, response: str) -> Tuple[float, float, float]:
        """Score Terraform configuration"""
        syntax = 1.0 if 'resource "' in response or 'module "' in response else 0.0
        correctness = 1.0 if any(r in response for r in TF_RESOURCES) else 0.5 if "resource" in response else 0.0
        fmt = 1.0 if response.count("{") == response.count("}") else 0.5
        return syntax, correctness, fmt

    def _score_kubernetes(self, response: str) -> Tuple[float, float, float]:
        """Score Kubernetes manifest"""
        syntax = 1.0 if "apiVersion:" in response and "kind:" in response else 0.0
        correctness = 1.0 if any(k in response for k in K8S_KINDS) else 0.0
        fmt = 1.0 if "metadata:" in response and "spec:" in response else 0.5
        return syntax, correctness, fmt

    def _score_docker(self, response: str) -> Tuple[float, float, float]:
        """Score Dockerfile or docker-compose"""
        is_dockerfile = "FROM" in response.upper()
        is_compose = "services:" in response or "version:" in response
        syntax = 1.0 if is_dockerfile or is_compose else 0.0
        correctness = 1.0 if sum(1 for k in DOCKER_KEYWORDS if k in response.upper()) >= 3 else 0.5
        fmt = 1.0 if len(response.strip()) > 50 else 0.5
        return syntax, correctness, fmt

    def _score_cicd(self, response: str) -> Tuple[float, float, float]:
        """Score CI/CD pipeline"""
        is_gha = "jobs:" in response or "runs-on:" in response
        is_gitlab = "stages:" in response or "script:" in response
        is_jenkins = "pipeline" in response.lower() or "stage(" in response
        syntax = 1.0 if is_gha or is_gitlab or is_jenkins else 0.0
        correctness = 1.0 if "steps:" in response or "script:" in response else 0.5
        fmt = 1.0 if response.count(":") >= 3 else 0.5
        return syntax, correctness, fmt

    def _score_mlops(self, response: str) -> Tuple[float, float, float]:
        """Score MLOps configuration"""
        resp_lower = response.lower()

        # Syntax: valid MLOps patterns
        has_k8s = "apiVersion:" in response and ("gpu" in resp_lower or "serving" in resp_lower)
        has_tf = 'resource "' in response and any(p in resp_lower for p in ["sagemaker", "vertex", "mlflow"])
        has_docker = "FROM" in response and any(f in resp_lower for f in ["pytorch", "tensorflow", "cuda"])
        has_pipeline = any(p in resp_lower for p in ["dag", "pipeline", "@task", "kubeflow"])
        syntax = 1.0 if any([has_k8s, has_tf, has_docker, has_pipeline]) else 0.0

        # Correctness: MLOps keywords
        mlops_hits = sum(1 for k in MLOPS_KEYWORDS if k in resp_lower)
        correctness = 1.0 if mlops_hits >= 4 else 0.5 if mlops_hits >= 2 else 0.0

        # Format: structure
        fmt = 1.0 if len(response) > 100 and response.count(":") >= 3 else 0.5
        return syntax, correctness, fmt


def get_score_fn(category: str):
    """Get scoring function for a category"""
    reward = IaCReward()
    return lambda response, _: reward.score(response, category)[0]
