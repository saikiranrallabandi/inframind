# Step 2: QLoRA Quantization + Dataset Loading

## Objective
Apply 4-bit quantization to reduce memory usage and load the InfraMind dataset.

## Why This Step?
- A100 has 40GB VRAM - quantization lets us fit larger effective batch sizes
- Load and verify our IaC training data
- Prepare for efficient fine-tuning

## Key Concepts

### QLoRA (Quantized Low-Rank Adaptation)
Combines two techniques:
1. **4-bit Quantization**: Compress model weights from 16-bit to 4-bit
2. **LoRA**: Train small adapter layers instead of full model

### BitsAndBytes Configuration
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # Enable 4-bit loading
    bnb_4bit_quant_type="nf4",           # NormalFloat4 - best for neural nets
    bnb_4bit_compute_dtype=torch.float16, # Compute in fp16 for speed
    bnb_4bit_use_double_quant=True,      # Quantize the quantization constants
)
```

### Why NF4 (NormalFloat4)?
- Neural network weights follow normal distribution
- NF4 optimizes quantization levels for this distribution
- Better accuracy than standard int4

### Memory Savings
| Precision | Memory per 1B params |
|-----------|---------------------|
| FP32      | 4 GB                |
| FP16      | 2 GB                |
| INT8      | 1 GB                |
| **NF4**   | **0.5 GB**          |

For 0.5B model: ~250MB instead of ~2GB

### InfraMind Dataset
```python
from inframind import create_dataset

dataset = create_dataset()  # 2000+ IaC tasks
# Categories: Terraform, Kubernetes, Docker, CI/CD, MLOps
```

Each sample has:
- `instruction`: What to do
- `input`: Context/requirements
- `output`: Correct IaC code

## Run Command
```bash
modal run modal_step2.py
```

## Expected Output
- Model loaded in 4-bit (~250MB VRAM)
- Dataset loaded with 2000+ samples
- Memory usage ~75% less than FP16

## Interview Questions

**Q: Why 4-bit instead of 8-bit?**
A: 4-bit allows larger batch sizes and faster training with minimal accuracy loss thanks to NF4.

**Q: What is double quantization?**
A: Quantizing the quantization constants themselves, saving additional ~0.4 bits per parameter.

**Q: Why NF4 over INT4?**
A: Neural network weights are normally distributed; NF4 places quantization levels optimally for this distribution.

**Q: How much memory does QLoRA save?**
A: ~75% reduction. A 7B model goes from 14GB to ~3.5GB.
