# IAPO: Infrastructure-Aware Policy Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2024.XXXXX-b31b1b.svg)](https://arxiv.org/)

**A reinforcement learning framework for fine-tuning small language models to reason about Infrastructure-as-Code.**

> **Note**: IAPO is a *training method*, not a model. It fine-tunes existing SLMs (like Qwen) using GRPO with domain-specific rewards. The resulting models (e.g., `Qwen-0.5B-IAPO`) can generate valid IaC.

## What is IAPO?

IAPO (Infrastructure-Aware Policy Optimization) is a **training framework** that:

1. Takes an existing small language model (Qwen, Llama, etc.)
2. Fine-tunes it using reinforcement learning (GRPO)
3. Uses infrastructure-specific reward functions to guide learning
4. Produces a model capable of generating valid Infrastructure-as-Code

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────────┐
│  Base Model     │  →   │  IAPO Training  │  →   │  Fine-tuned Model   │
│  (Qwen 0.5B)    │      │  (GRPO + IaC    │      │  (Qwen-0.5B-IAPO)   │
│                 │      │   Rewards)      │      │                     │
└─────────────────┘      └─────────────────┘      └─────────────────────┘
```

### What IAPO Provides

| Component | Description |
|-----------|-------------|
| **IaC-Bench** | Benchmark dataset with 500+ IaC tasks |
| **IAPO Rewards** | Domain-specific reward functions for Terraform, K8s, Docker, CI/CD |
| **Training Pipeline** | GRPO implementation adapted for infrastructure reasoning |

## The Problem

Large Language Models (GPT-4, Claude) can generate Infrastructure-as-Code, but:
- **Cost**: API calls add up ($100s-$1000s/month for teams)
- **Privacy**: Your infrastructure code is sent to external servers
- **Offline**: Doesn't work in air-gapped/secure environments
- **Customization**: Can't fine-tune on your specific patterns

Small open-source models (< 1B parameters) fail at IaC because:
- They **hallucinate** resource names (`aws_ec2` instead of `aws_instance`)
- They generate **invalid syntax** that won't pass `terraform validate`
- They **ignore security** best practices
- Traditional fine-tuning (SFT/LoRA) only **memorizes patterns**, doesn't teach reasoning

## Our Solution

**IAPO** fine-tunes small models using reinforcement learning to **reason** about infrastructure, not just memorize examples.

| Approach | Method | Result |
|----------|--------|--------|
| SFT/LoRA | "Memorize this Terraform example" | Copies patterns, fails on novel tasks |
| **IAPO** | "Generate Terraform, I'll score if it's valid" | Learns reasoning, handles new tasks |

### Key Innovation

IAPO extends GRPO (Group Relative Policy Optimization) with **infrastructure-specific rewards**:

```
IAPO Reward = α × Syntax + β × Correctness + γ × Format

Where:
- Syntax: Does it pass `terraform validate`?
- Correctness: Are the right resources used?
- Format: Is the structure proper?
```

## Features

- **IaC-Bench**: 500+ tasks across Terraform, Kubernetes, Docker, CI/CD
- **GRPO Training**: Reinforcement learning that teaches reasoning
- **Model Agnostic**: Works with Qwen, Llama, Mistral, or any HuggingFace model
- **Alpaca Format**: Compatible with standard training pipelines
- **Local-first**: Runs entirely on your machine

## Installation

```bash
pip install iapo
```

Or from source:

```bash
git clone https://github.com/saikiranrallabandi/iapo.git
cd iapo
pip install -e .
```

## Quick Start

```python
from iapo import create_dataset, IAPOTrainer

# Load 500+ IaC tasks
dataset = create_dataset(size=100)

# Fine-tune with IAPO (GRPO + IaC rewards)
trainer = IAPOTrainer(model_name="Qwen/Qwen2.5-0.5B-Instruct")
trainer.train(dataset, epochs=1)

# Save your fine-tuned model
trainer.save("./qwen-0.5b-iapo")
```

## How It Works

### 1. IaC-Bench Dataset

529 infrastructure tasks in [Alpaca format](https://github.com/tatsu-lab/stanford_alpaca):

```json
{
  "instruction": "Create Terraform for AWS EC2 instance",
  "input": "t2.micro instance type",
  "output": ""
}
```

| Category | Tasks | Examples |
|----------|-------|----------|
| Terraform | 225 | EC2, S3, VPC, RDS, EKS, Lambda, IAM |
| Kubernetes | 138 | Deployments, Services, Ingress, RBAC |
| Docker | 70 | Dockerfiles, docker-compose |
| CI/CD | 96 | GitHub Actions, GitLab CI, Jenkins |

### 2. IAPO Training Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                    IAPO TRAINING                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  For each IaC task:                                            │
│                                                                 │
│  1. GENERATE: Model produces multiple IaC outputs              │
│     "Create EC2" → [output1, output2]                          │
│                                                                 │
│  2. SCORE: IAPO reward function evaluates each                 │
│     output1: syntax=1.0, correct=0.8, format=0.9 → 0.89        │
│     output2: syntax=0.0, correct=0.5, format=0.7 → 0.38        │
│                                                                 │
│  3. ADVANTAGE: Compare within group (GRPO)                     │
│     output1: above average → positive advantage                │
│     output2: below average → negative advantage                │
│                                                                 │
│  4. UPDATE: Increase probability of better outputs             │
│     Model learns: "valid syntax = higher reward"               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3. IAPO Reward Function

```python
from iapo import IaCReward

reward = IaCReward(alpha=0.4, beta=0.3, gamma=0.3)

# Score a Terraform output
score, details = reward.score(terraform_code, category="terraform")
# score: 0.85
# details: {"syntax": 1.0, "correctness": 0.8, "format": 0.75}
```

**Reward components:**

| Component | Weight | What it measures |
|-----------|--------|------------------|
| Syntax | 0.4 | Valid resource declarations |
| Correctness | 0.3 | Right resource types used |
| Format | 0.3 | Proper structure (balanced braces, etc.) |

## Training

### Basic Training

```bash
python scripts/train.py --epochs 3 --output ./iapo-model
```

### Category-Specific Training

```bash
# Train only on Terraform
python scripts/train.py --category terraform --epochs 5

# Train only on Kubernetes
python scripts/train.py --category kubernetes --epochs 5
```

### Custom Training

```python
from iapo import create_dataset, IAPOTrainer

# Load specific categories
dataset = create_dataset(categories=["terraform", "kubernetes"], size=200)

# Configure trainer
trainer = IAPOTrainer(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    lr=1e-5,
    group_size=4  # More samples per task for better GRPO
)

# Train
history = trainer.train(dataset, epochs=3)

# Check progress
for epoch in history:
    print(f"Epoch {epoch['epoch']}: Reward = {epoch['mean_reward']:.3f}")
```

## Comparison with Existing Work

| Project | Type | Method | IaC-specific |
|---------|------|--------|--------------|
| [devops-slm-v1](https://huggingface.co/lakhera2023/devops-slm-v1) | Fine-tuned Model | LoRA/SFT | Yes |
| [AIAC](https://github.com/gofireflyio/aiac) | CLI Tool | Prompting (API) | Yes |
| GPT-4 / Claude | API Service | - | No |
| **IAPO** | **Training Framework** | **GRPO/RL** | **Yes** |

**Key differentiator**: IAPO is a *training framework*, not a model or API wrapper. It uses reinforcement learning with infrastructure-specific rewards to fine-tune any SLM for IaC generation.

## Project Structure

```
iapo/
├── iapo/
│   ├── __init__.py      # Package exports
│   ├── dataset.py       # IaC-Bench (500+ tasks)
│   ├── rewards.py       # IAPO reward functions
│   └── train.py         # GRPO trainer
├── scripts/
│   └── train.py         # Training CLI
├── examples/
│   └── quickstart.py    # Quick start example
├── README.md
├── LICENSE
└── pyproject.toml
```

## Roadmap

- [x] IaC-Bench dataset (500+ tasks)
- [x] IAPO training framework with GRPO
- [x] Domain-specific reward functions
- [ ] Release fine-tuned models on HuggingFace (`qwen-0.5b-iapo`, `qwen-1.5b-iapo`)
- [ ] Real validation integration (`terraform validate`)
- [ ] Security scoring (`tfsec`, `checkov`)
- [ ] CLI tool (`iapo generate "create S3 bucket"`)
- [ ] VS Code extension

## Citation

```bibtex
@article{rallabandi2024iapo,
  title={IAPO: Infrastructure-Aware Policy Optimization for Fine-tuning Small Language Models},
  author={Rallabandi, Sai Kiran},
  journal={arXiv preprint arXiv:2024.XXXXX},
  year={2024}
}
```

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [DeepSeek](https://github.com/deepseek-ai) for GRPO
- [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) for data format
- [Reasoning Gym](https://github.com/open-thought/reasoning-gym) for architecture inspiration
- [AIAC](https://github.com/gofireflyio/aiac) for IaC task patterns
