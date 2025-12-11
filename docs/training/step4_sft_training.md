# Step 4: SFT Training with TRL

## Objective
Run Supervised Fine-Tuning (SFT) using HuggingFace TRL library with best practices.

## Why This Step?
- Actually train the model on InfraMind dataset
- Use TRL's optimized SFTTrainer
- Apply all the preparation from Steps 1-3

## Key Concepts

### SFT (Supervised Fine-Tuning)
Train model to produce correct outputs given instructions:
```
Input:  "Write Terraform for S3 bucket with versioning"
Output: "resource \"aws_s3_bucket\" \"main\" { ... }"
```

### TRL Best Practices
```python
from trl import SFTTrainer, SFTConfig

sft_config = SFTConfig(
    output_dir="/models/checkpoints",

    # Training params
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,    # Effective batch = 4 × 4 = 16

    # Learning rate
    learning_rate=2e-4,               # Good for LoRA
    lr_scheduler_type="cosine",       # Smooth decay
    warmup_ratio=0.03,                # 3% warmup

    # TRL-specific
    packing=True,                     # Pack sequences for efficiency
    dataset_text_field="text",        # Field containing formatted text
    max_seq_length=1024,              # Max tokens per sequence

    # Optimization
    optim="adamw_torch",
    weight_decay=0.01,
    max_grad_norm=1.0,                # Gradient clipping

    # Precision
    bf16=True,                        # BFloat16 for A100

    # Logging
    logging_steps=10,
    save_strategy="epoch",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
    processing_class=tokenizer,
    peft_config=peft_config,          # TRL applies LoRA automatically!
)
```

### Key Parameters Explained

| Parameter | Value | Why |
|-----------|-------|-----|
| `packing=True` | - | Combines short sequences to maximize GPU utilization |
| `learning_rate=2e-4` | - | Standard for LoRA fine-tuning |
| `gradient_accumulation=4` | - | Simulates larger batch without more VRAM |
| `bf16=True` | - | A100 native format, better than fp16 |
| `max_seq_length=1024` | - | IaC code can be long |

### Packing Explained
Without packing:
```
Batch 1: [short_sample_____PAD_PAD_PAD]  # Wasted compute
Batch 2: [another_short___PAD_PAD_PAD]
```

With packing:
```
Batch 1: [short_sample|another_short|third]  # Efficient!
```

### Training Loop
```
Epoch 1: Learn basic patterns
Epoch 2: Refine understanding
Epoch 3: Polish outputs

Each step:
1. Forward pass (compute loss)
2. Backward pass (compute gradients)
3. Update LoRA weights only
4. Log metrics
```

### Expected Metrics
```
Step 100:  loss=2.5, lr=1.8e-4
Step 500:  loss=1.2, lr=1.5e-4
Step 1000: loss=0.8, lr=1.0e-4
Final:     loss=0.5, lr=0.0
```

Loss should decrease steadily!

## Run Command
```bash
modal run modal_step4.py
```

## Expected Output
- Training runs for 3 epochs
- Loss decreases from ~2.5 to ~0.5
- Checkpoints saved each epoch
- Final model saved to `/models/inframind-final`

## Training Time Estimate
- Dataset: 2000 samples
- Batch size: 16 (effective)
- Steps per epoch: ~125
- Total steps: ~375
- Time: ~15-30 minutes on A100

## Interview Questions

**Q: Why use SFTTrainer instead of regular Trainer?**
A: SFTTrainer has built-in support for packing, chat templates, and LoRA integration. Less boilerplate code.

**Q: What is packing and why use it?**
A: Packing combines multiple short sequences into one, reducing padding waste. Can improve throughput 2-3x.

**Q: Why cosine learning rate schedule?**
A: Smooth decay prevents sudden drops that can destabilize training. Better than linear for most tasks.

**Q: What's gradient accumulation?**
A: Simulates larger batch sizes by accumulating gradients over multiple forward passes before updating. Useful when VRAM limited.

**Q: Why bf16 instead of fp16?**
A: BFloat16 has larger dynamic range (same exponent bits as fp32), more stable for training. A100 has native bf16 support.

**Q: How do you know if training is working?**
A: Loss decreasing, not oscillating wildly, validation metrics improving. If loss plateaus early, learning rate might be too low.
