# Step 3: Add LoRA Adapters

## Objective
Add Low-Rank Adaptation (LoRA) layers to the quantized model. These are the **only parameters we'll train**.

## Why LoRA?
Instead of training all 494M parameters, LoRA adds small trainable matrices:

```
Original: W (frozen, 494M params)
LoRA:     W + A×B (trainable, ~2M params)
```

| Approach | Trainable Params | Memory | Quality |
|----------|------------------|--------|---------|
| Full Fine-tuning | 494M | ~4GB | Best |
| **LoRA** | **~2M** | **~0.5GB** | **95%+ of full** |
| Prompt Tuning | ~100K | ~0.1GB | 80-90% |

## Key Concepts

### LoRA Architecture
```
Input → [Frozen Weights W] → Output
         ↓
       [A matrix (r × d)] → [B matrix (d × r)]
         ↓
       Low-rank update ΔW = A × B
```

### LoRA Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `r` (rank) | 32 | Size of low-rank matrices. Higher = more capacity |
| `lora_alpha` | 64 | Scaling factor. Usually 2× rank |
| `lora_dropout` | 0.05 | Regularization during training |
| `target_modules` | q,k,v,o,gate,up,down | Which layers to adapt |

### Target Modules (Transformer Layers)
```
Attention Block:
  - q_proj: Query projection (how to look)
  - k_proj: Key projection (what to match)
  - v_proj: Value projection (what to retrieve)
  - o_proj: Output projection (combine heads)

MLP Block:
  - gate_proj: Gating mechanism
  - up_proj: Expand dimensions
  - down_proj: Contract dimensions
```

## Code: `modal_step3.py`

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Prepare model for training (enables gradient checkpointing)
model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    r=32,                    # Rank
    lora_alpha=64,           # Scaling (usually 2x rank)
    lora_dropout=0.05,       # Regularization
    bias="none",             # Don't train biases
    task_type="CAUSAL_LM",   # Language modeling
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

## Run Command
```bash
modal run modal_step3.py
```

## Expected Output
```
Trainable Parameters Summary
============================================================
trainable params: 2,359,296 || all params: 496,392,064 || trainable%: 0.4753

Breakdown:
  - Total parameters: 496,392,064
  - Trainable (LoRA): 2,359,296
  - Frozen (base): 494,032,768
  - Training: 0.48% of model
```

## Memory Savings
| Component | Memory |
|-----------|--------|
| Base model (4-bit) | ~0.23 GB |
| LoRA adapters | ~0.01 GB |
| Optimizer states | ~0.05 GB |
| Gradients | ~0.05 GB |
| **Total Training** | **~0.35 GB** |

Compare to full fine-tuning: **~8GB+**

## Important Notes

1. **Untrained LoRA** - At this step, LoRA weights are randomly initialized. Output will be similar to base model.

2. **prepare_model_for_kbit_training()** - This function:
   - Enables gradient checkpointing
   - Casts layer norms to float32
   - Makes model ready for backpropagation

3. **Why these target_modules?** - These are the attention and MLP layers where most model knowledge is stored.

## Next Step
[Step 4: Train for 1 Epoch](./step4_train.md)
