# InfraMind Project Context

## What is InfraMind?

**InfraMind** is a fine-tuning toolkit (NOT a model) for training small language models on Infrastructure-as-Code tasks.

```
Base Model (Qwen 0.5B) → InfraMind Fine-tuning → qwen-0.5b-inframind
```

## Project Status

### Completed
- [x] Renamed from IAPO to InfraMind (IAPO overclaimed - we didn't create a new optimization algorithm)
- [x] Created InfraMind-Bench dataset (500+ tasks)
- [x] Categories: Terraform, Kubernetes, Docker, CI/CD
- [x] Added MLOps templates (in progress when session paused)
- [x] Reward functions for scoring IaC outputs
- [x] GRPO trainer implementation
- [x] Pushed to GitHub: github.com/saikiranrallabandi/iapo

### In Progress
- [ ] Adding MLOps category to dataset (templates added, reward scoring pending)
- [ ] SFT/LoRA training first (before GRPO)

### Pending
- [ ] Run SFT/LoRA training on Qwen
- [ ] Add LLM-as-Judge for evaluation
- [ ] Get experiment numbers
- [ ] Release fine-tuned models to HuggingFace

## Key Decisions Made

1. **Name**: InfraMind (not IAPO) - we're a fine-tuning toolkit, not creating new optimization
2. **Training approach**: SFT/LoRA first, then GRPO with LLM-as-Judge
3. **Base model**: Qwen/Qwen2.5-0.5B-Instruct
4. **Evaluation**: LLM-as-Judge is best (not keyword matching)

## Architecture

```
inframind/
├── inframind/
│   ├── __init__.py      # Exports: InfraMindTrainer, IaCBench, create_dataset, IaCReward
│   ├── dataset.py       # InfraMind-Bench (500+ tasks in Alpaca format)
│   ├── rewards.py       # Scoring functions for Terraform, K8s, Docker, CI/CD, MLOps
│   └── train.py         # GRPO trainer
├── scripts/
│   └── train.py         # CLI for training
├── examples/
│   └── quickstart.py
└── pyproject.toml       # Package config (name: inframind)
```

## Training Pipeline

```
1. DATA PREPARATION
   Raw Data → LLM-as-Judge → Filtered/Scored Data

2. SFT/LoRA TRAINING (do this first)
   Filtered Data → SFT/LoRA → Base Fine-tuned Model

3. GRPO TRAINING (optional, after SFT)
   SFT Model → GRPO + LLM-as-Judge rewards → Final Model

4. EVALUATION
   Final Model → Generate → LLM-as-Judge → Quality Score
```

## Dataset Categories

| Category | Tasks | Examples |
|----------|-------|----------|
| Terraform | 225 | EC2, S3, VPC, RDS, EKS, Lambda, IAM |
| Kubernetes | 138 | Deployments, Services, Ingress, RBAC |
| Docker | 70 | Dockerfiles, docker-compose |
| CI/CD | 96 | GitHub Actions, GitLab CI, Jenkins |
| MLOps | 150+ | SageMaker, MLflow, Kubeflow, Ray, vLLM (pending) |

## Reward Function

```
Reward = α × Syntax + β × Correctness + γ × Format
Where: α=0.4, β=0.3, γ=0.3
```

## Commands

```bash
# Install
pip install -e .

# Train
python scripts/train.py --epochs 3 --output ./inframind-model

# Category-specific
python scripts/train.py --category terraform --epochs 5
```

## Quick Start Code

```python
from inframind import create_dataset, InfraMindTrainer

dataset = create_dataset(size=100)
trainer = InfraMindTrainer(model_name="Qwen/Qwen2.5-0.5B-Instruct")
trainer.train(dataset, epochs=1)
trainer.save("./qwen-0.5b-inframind")
```

## Background Context

- Working on EB-1A visa case (Original Contribution criterion)
- H1B restrictions: Can't use tideon.ai as evidence
- Goal: Build artifacts (InfraMind) for adoption/citations
- Attorney suggested focusing on academic/research output
- CEO feedback: Don't just apply existing technique to domain - solve a real problem

## Reference Links

- Colab for SFT/LoRA: https://colab.research.google.com/drive/1UgTUI6AeVnSlknHoF3cEDhWLHYirghju
- NeurIPS 2025 best papers analyzed for patterns
- Stanford Alpaca format: https://github.com/tatsu-lab/stanford_alpaca

## Git Info

- Repo: github.com/saikiranrallabandi/iapo (package name is inframind)
- Commits as: saikiranrallabandi
- Branch: main

## Next Steps When Resuming

1. Complete MLOps reward scoring in `rewards.py` (add `_score_mlops` method)
2. Run SFT/LoRA training using the colab notebook
3. Evaluate with LLM-as-Judge
4. Then consider GRPO if SFT results are good
