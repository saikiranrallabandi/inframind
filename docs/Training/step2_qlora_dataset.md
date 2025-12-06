# Step 2: QLoRA Quantization + InfraMind Dataset

## Objective
- Load model with **4-bit quantization** (QLoRA) to reduce memory
- Load **InfraMind dataset** for training
- Format dataset for SFT training

## Why QLoRA?
QLoRA (Quantized Low-Rank Adaptation) allows training large models on limited GPU memory:

| Precision | Memory per Parameter | 0.5B Model |
|-----------|---------------------|------------|
| FP32 | 4 bytes | ~2 GB |
| FP16/BF16 | 2 bytes | ~1 GB |
| **4-bit** | 0.5 bytes | **~0.25 GB** |

With QLoRA, we get **75% memory savings** while maintaining quality!

## Code: `modal_step2.py`

### Part 1: Load InfraMind Dataset
```python
from inframind import create_dataset

dataset = create_dataset(size=100)  # Start with 100 tasks
print(f"Tasks: {len(dataset)}")
print(f"Categories: terraform, kubernetes, docker, cicd, mlops")
```

### Part 2: Format for SFT
```python
def format_prompt(task):
    instruction = task.get("instruction", "")
    input_text = task.get("input", "")
    output = task.get("output", "")

    if input_text:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
```

### Part 3: QLoRA Configuration
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # 4-bit quantization
    bnb_4bit_quant_type="nf4",      # NormalFloat4 (best for LLMs)
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True, # Nested quantization
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
```

## Run Command
```bash
modal run modal_step2.py
```

## Key Concepts

### BitsAndBytesConfig Parameters
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `load_in_4bit` | True | Enable 4-bit quantization |
| `bnb_4bit_quant_type` | "nf4" | NormalFloat4 - optimal for pretrained weights |
| `bnb_4bit_compute_dtype` | bfloat16 | Compute precision during forward pass |
| `bnb_4bit_use_double_quant` | True | Quantize the quantization constants too |

### InfraMind Dataset Format (Alpaca-style)
```json
{
  "instruction": "Create Terraform for AWS S3 bucket",
  "input": "with versioning enabled",
  "output": "resource \"aws_s3_bucket\" {...}",
  "category": "terraform"
}
```

## Expected Output
```
[1/3] Loading InfraMind dataset...
  - Total tasks: 100
  - Categories: {'terraform': 40, 'kubernetes': 30, ...}

[2/3] Formatting dataset for SFT training...
  - HuggingFace Dataset: Dataset({features: ['text', 'category'], num_rows: 100})

[3/3] Loading model with QLoRA (4-bit quantization)...
  - Parameters: 494,032,768
  - FP16 would use: ~0.92 GB
  - 4-bit uses: ~0.23 GB
  - Savings: ~75%
```

## Next Step
[Step 3: Add LoRA Adapters](./step3_lora_adapters.md)
