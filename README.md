# IAPO: Infrastructure-Aware Policy Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**IAPO** is the first reinforcement learning framework for training small language models on Infrastructure-as-Code (IaC) tasks using GRPO (Group Relative Policy Optimization).

## Features

- 🚀 **500+ IaC Tasks**: Terraform, Kubernetes, Docker, CI/CD
- 🧠 **GRPO Training**: Group Relative Policy Optimization for reasoning
- 📊 **IaC-Bench**: Comprehensive benchmark for evaluation
- 🔧 **Qwen-based**: Built on Qwen for strong base performance
- 📦 **Alpaca Format**: Compatible with standard training pipelines

## Installation

```bash
pip install iapo
# or
git clone https://github.com/yourusername/iapo.git
cd iapo
pip install -e .
```

## Quick Start

```python
from iapo import IaCBench, IAPOTrainer, create_dataset

# Load dataset
dataset = create_dataset(size=100)
print(f"Tasks: {len(dataset)}")

# Train model
trainer = IAPOTrainer(model_name="Qwen/Qwen2.5-0.5B-Instruct")
trainer.train(dataset, epochs=1)
trainer.save("./iapo-model")
```

## Dataset Format (Alpaca-compatible)

```json
{
  "instruction": "Create Terraform for AWS EC2 instance",
  "input": "t2.micro instance type",
  "output": ""
}
```

## IaC-Bench Categories

| Category | Tasks | Examples |
|----------|-------|----------|
| Terraform | 200+ | EC2, S3, VPC, RDS, EKS, Lambda |
| Kubernetes | 150+ | Deployments, Services, Ingress, RBAC |
| Docker | 70+ | Dockerfiles, docker-compose |
| CI/CD | 100+ | GitHub Actions, GitLab CI, Jenkins |

## Training

```bash
# Train on full dataset
python scripts/train.py --epochs 3 --batch-size 4

# Train on specific category
python scripts/train.py --category terraform --epochs 5
```

## Evaluation

```bash
python scripts/evaluate.py --model ./iapo-model --benchmark iac-bench
```

## Citation

```bibtex
@article{iapo2024,
  title={IAPO: Infrastructure-Aware Policy Optimization for Small Language Models},
  author={Rallabandi, Sai Kiran},
  year={2024}
}
```

## License

MIT License
