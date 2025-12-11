# Step 3: Add LoRA Adapters

## Objective
Configure and attach LoRA (Low-Rank Adaptation) adapters to the quantized model.

## Why This Step?
- Train only ~1% of parameters instead of 100%
- Much faster training
- Can merge adapters back into base model later

## Key Concepts

### What is LoRA?
Instead of updating all model weights W, LoRA adds small "adapter" matrices:

```
Original: Y = W × X
LoRA:     Y = W × X + (A × B) × X

Where:
- W is frozen (not trained)
- A is (d × r) matrix
- B is (r × d) matrix
- r << d (low rank)
```

### LoRA Configuration
```python
from peft import LoraConfig

peft_config = LoraConfig(
    r=32,                      # Rank - expressiveness of adapters
    lora_alpha=64,             # Scaling factor (usually 2×r)
    lora_dropout=0.05,         # Regularization
    bias="none",               # Don't train biases
    task_type="CAUSAL_LM",     # Language modeling task
    target_modules=[           # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"       # FFN
    ],
)
```

### Key Parameters Explained

| Parameter | Value | Why |
|-----------|-------|-----|
| `r` (rank) | 32 | Higher = more capacity, more params |
| `lora_alpha` | 64 | Scaling; alpha/r = effective learning rate multiplier |
| `target_modules` | attention + FFN | These layers capture most task-specific knowledge |
| `dropout` | 0.05 | Prevents overfitting on small datasets |

### Target Modules for Qwen2.5
```
Attention layers:
- q_proj: Query projection
- k_proj: Key projection
- v_proj: Value projection
- o_proj: Output projection

Feed-forward layers:
- gate_proj: Gating mechanism
- up_proj: Upward projection
- down_proj: Downward projection
```

### Parameter Count
```
Full model:     500M parameters (100%)
LoRA adapters:  ~5M parameters  (~1%)
```

Only 1% of parameters are trainable!

## Run Command
```bash
modal run modal_step3.py
```

## Expected Output
- LoRA adapters attached to model
- Trainable params: ~5M (1% of total)
- Model ready for training

## Interview Questions

**Q: Why rank 32?**
A: Balance between capacity and efficiency. Lower ranks (8-16) for simple tasks, higher (64+) for complex tasks. 32 is good for IaC.

**Q: What does lora_alpha do?**
A: Scales the adapter contribution. Higher alpha = stronger adapter influence. Rule of thumb: alpha = 2 × rank.

**Q: Why target attention AND FFN layers?**
A: Attention captures relationships (what to look at), FFN captures knowledge (what to output). Both matter for code generation.

**Q: Can you merge LoRA back into the base model?**
A: Yes! After training: `model = model.merge_and_unload()`. This gives a single model file with no adapter overhead.

**Q: What's the difference between LoRA and full fine-tuning?**
A: LoRA trains ~1% of params, 10-100x faster, similar quality for most tasks. Full fine-tuning is only needed for massive domain shifts.
