---
license: mit
language:
- en
library_name: transformers
pipeline_tag: text-generation
tags:
- infrastructure-as-code
- terraform
- kubernetes
- docker
- devops
- iac
- grpo
- dapo
- reinforcement-learning
- fine-tuned
base_model: Qwen/Qwen2.5-0.5B-Instruct
datasets:
- custom
model-index:
- name: inframind-grpo
  results:
  - task:
      type: text-generation
      name: IaC Generation
    dataset:
      name: InfraMind-Bench
      type: custom
    metrics:
    - type: accuracy
      value: 97.3
      name: GRPO Accuracy
- name: inframind-dapo
  results:
  - task:
      type: text-generation
      name: IaC Generation
    dataset:
      name: InfraMind-Bench
      type: custom
    metrics:
    - type: accuracy
      value: 96.4
      name: DAPO Accuracy
---

# InfraMind: Infrastructure-as-Code Small Language Model

**InfraMind** is a 0.5B parameter language model fine-tuned for Infrastructure-as-Code (IaC) generation using reinforcement learning (GRPO/DAPO).

## Model Description

| Attribute | Value |
|-----------|-------|
| **Base Model** | Qwen/Qwen2.5-0.5B-Instruct |
| **Parameters** | 500M |
| **Training Method** | GRPO + DAPO (Reinforcement Learning) |
| **Domain** | Infrastructure-as-Code |
| **License** | MIT |

### Why InfraMind?

Unlike traditional fine-tuning (SFT/LoRA) that memorizes patterns, InfraMind uses **reinforcement learning with domain-specific rewards** to teach the model to *reason* about infrastructure.

| Approach | Method | Result |
|----------|--------|--------|
| SFT/LoRA | "Memorize this Terraform example" | Copies patterns, fails on novel tasks |
| **InfraMind** | "Generate Terraform, I'll score if it's valid" | Learns reasoning, handles new tasks |

## Evaluation Results

| Model | Training Method | Accuracy | Pass Threshold |
|-------|-----------------|----------|----------------|
| **inframind-grpo** | GRPO | **97.3%** | 0.6 |
| **inframind-dapo** | DAPO | **96.4%** | 0.6 |
| Base (Qwen2.5-0.5B) | None | ~30% | 0.6 |

Evaluated on **InfraMind-Bench** (110 held-out test samples) across:
- Terraform (AWS, GCP, Azure)
- Kubernetes (Deployments, Services, Ingress)
- Docker (Dockerfile, docker-compose)
- CI/CD (GitHub Actions, GitLab CI)

## Comparison with Other Models

| Model | Params | Training | Benchmarks | Edge Deploy |
|-------|--------|----------|------------|-------------|
| qwen3-devops | 1.7B | SFT | None | No |
| devops-slm-v1 | 7B | LoRA | None | No |
| **InfraMind** | **0.5B** | **GRPO/DAPO** | **97.3%** | **Yes** |

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("srallabandi0225/inframind-0.5b-grpo")
tokenizer = AutoTokenizer.from_pretrained("srallabandi0225/inframind-0.5b-grpo")

# Generate Terraform
prompt = """### Instruction:
Create Terraform for AWS EC2 instance
### Input:
t3.micro instance type
### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.pad_token_id
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Example Output

```hcl
resource "aws_instance" "web" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t3.micro"

  tags = {
    Name = "web-server"
  }
}
```

## Supported IaC Categories

| Category | Examples | Coverage |
|----------|----------|----------|
| **Terraform** | EC2, S3, VPC, RDS, EKS, Lambda, IAM | AWS, GCP, Azure |
| **Kubernetes** | Deployment, Service, Ingress, ConfigMap, RBAC | All K8s resources |
| **Docker** | Dockerfile, docker-compose | Multi-stage builds |
| **CI/CD** | GitHub Actions, GitLab CI, Jenkins | Workflows, pipelines |
| **Ansible** | Playbooks, roles | Server configuration |
| **Helm** | Charts, values.yaml | K8s package management |

## Training Details

### GRPO (Group Relative Policy Optimization)

First stage training using GRPO:

```yaml
Training:
  epochs: 3
  batch_size: 16 (effective)
  learning_rate: 5e-6
  beta (KL): 0.04
  generations_per_prompt: 4

LoRA:
  r: 16
  alpha: 32
  target_modules: [q_proj, k_proj, v_proj, o_proj]
```

### DAPO (Direct Advantage Policy Optimization)

Second stage training with DAPO innovations:

```yaml
Training:
  epochs: 2
  batch_size: 16 (effective)
  learning_rate: 5e-6
  beta (KL): 0.0  # Pure DAPO
  generations_per_prompt: 8

DAPO Innovations:
  1. Clip-Higher: Asymmetric clipping (ε_low=0.2, ε_high=0.28)
  2. Dynamic Sampling: Skip uniform reward batches
  3. Token-Level Loss: Per-token policy gradient
  4. Overlong Punishment: Soft length penalty
```

### Reward Function

Domain-specific reward for IaC quality:

```
Reward = α × Syntax + β × Correctness + γ × Format

Where:
- Syntax (α=0.4): Valid resource declarations
- Correctness (β=0.3): Correct resource types
- Format (γ=0.3): Proper structure
```

## Hardware Requirements

| Deployment | Memory | GPU |
|------------|--------|-----|
| Training | 16GB+ | A100/A10G |
| Inference | 2GB | Optional |
| Edge (Raspberry Pi 5) | 4GB | None |

The 0.5B model is small enough to run on edge devices, making it suitable for:
- Air-gapped environments
- Local development
- CI/CD pipelines
- IoT/Edge infrastructure

## Limitations

- **IaC-specific**: Optimized for infrastructure tasks, not general conversation
- **English only**: Training data is in English
- **No execution**: Generates code, does not execute or validate against real infrastructure
- **Version-sensitive**: Generated code may use older API versions
- **Security**: Always review generated code for security best practices

### Out-of-Scope Uses

- Legal or medical advice
- General-purpose chatbot
- Executing infrastructure changes without human review
- Production deployment without validation

## Intended Use

### Primary Use Cases
- Generating Terraform configurations
- Creating Kubernetes manifests
- Writing Dockerfiles and docker-compose
- Building CI/CD pipelines
- Infrastructure automation scripting

### Users
- DevOps engineers
- Platform engineers
- SREs
- Cloud architects
- Infrastructure developers

## Training Data

**InfraMind-Bench**: 2000+ IaC tasks in Alpaca format

| Category | Tasks |
|----------|-------|
| Terraform | 500+ |
| Kubernetes | 400+ |
| Docker | 300+ |
| CI/CD | 300+ |
| Ansible | 200+ |
| Helm | 150+ |
| Monitoring | 150+ |

Data format:
```json
{
  "instruction": "Create Terraform for AWS EC2 instance",
  "input": "t3.micro instance type",
  "output": ""
}
```

## Ethical Considerations

- Model may generate insecure configurations if not prompted for security
- Generated infrastructure code should always be reviewed before deployment
- Model does not have access to real infrastructure or credentials
- Users are responsible for validating generated code against their security policies

## Citation

```bibtex
@misc{rallabandi2024inframind,
  title={InfraMind: Fine-tuning Small Language Models for Infrastructure-as-Code Generation with Reinforcement Learning},
  author={Rallabandi, Sai Kiran},
  year={2024},
  publisher={HuggingFace},
  url={https://huggingface.co/srallabandi0225/inframind-0.5b-grpo}
}
```

## Links

- **GitHub**: [github.com/saikiranrallabandi/inframind](https://github.com/saikiranrallabandi/inframind)
- **GRPO Model**: [srallabandi0225/inframind-0.5b-grpo](https://huggingface.co/srallabandi0225/inframind-0.5b-grpo)
- **DAPO Model**: [srallabandi0225/inframind-0.5b-dapo](https://huggingface.co/srallabandi0225/inframind-0.5b-dapo)

## Acknowledgments

- [Qwen Team](https://github.com/QwenLM/Qwen) for the base model
- [DeepSeek](https://github.com/deepseek-ai) for GRPO
- [NVIDIA NeMo](https://docs.nvidia.com/nemo) for DAPO reference
- [TRL](https://github.com/huggingface/trl) for training infrastructure

## Model Card Contact

**Author**: Sai Kiran Rallabandi
**GitHub**: [@saikiranrallabandi](https://github.com/saikiranrallabandi)
