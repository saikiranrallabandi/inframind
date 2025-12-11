# Step 1: Load Base Model

## Objective
Load the base model (Qwen2.5-0.5B-Instruct) and verify it works before any fine-tuning.

## Why This Step?
- Verify Modal.com GPU setup works
- See baseline model behavior on IaC tasks
- Understand what the model outputs WITHOUT training

## Key Concepts

### Base Model
- **Qwen2.5-0.5B-Instruct**: Small but capable instruction-tuned model
- Already knows general language patterns
- NOT trained on Infrastructure-as-Code specifically

### Modal.com Setup
```python
import modal

app = modal.App("inframind-step1")

# GPU Image with dependencies
image = modal.Image.debian_slim(python_version="3.11").apt_install("git").pip_install(
    "torch",
    "transformers==4.45.0",
    "accelerate",
)

@app.function(gpu="A100", image=image, timeout=600)
def load_model():
    # Model loading code here
    pass
```

### What Happens
1. Modal spins up A100 GPU container
2. Downloads Qwen2.5-0.5B-Instruct from HuggingFace
3. Runs a test prompt (Terraform task)
4. Shows baseline output (usually incorrect for IaC)

## Expected Output
The base model will likely produce:
- Generic responses
- Incorrect Terraform syntax
- Missing IaC best practices

This is EXPECTED - that's why we need to fine-tune!

## Run Command
```bash
modal run modal_step1.py
```

## Interview Questions

**Q: Why start with a small model like 0.5B?**
A: Faster iteration, lower cost, proves the approach works before scaling up.

**Q: Why use Modal.com?**
A: On-demand A100 GPUs, pay-per-use, no infrastructure management.

**Q: What's the difference between base and instruct models?**
A: Instruct models are already fine-tuned to follow instructions, making them better starting points for task-specific fine-tuning.
