# Step 4: SFT Training with LoRA

## Objective
Train the model using Supervised Fine-Tuning (SFT) with LoRA adapters on the InfraMind dataset.

## What Happens in This Step

1. **Load Dataset**: 500 InfraMind tasks formatted for SFT
2. **Load Model**: Qwen2.5-0.5B with 4-bit quantization (QLoRA)
3. **Add LoRA**: Trainable adapters on attention & MLP layers
4. **Train**: 1 epoch with SFTTrainer from TRL
5. **Save**: Persist model to Modal volume
6. **Test**: Generate samples to verify learning

## Code: `modal_step4.py`

### Training Configuration
```python
training_args = TrainingArguments(
    output_dir="/models/checkpoints",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch = 16
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=True,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
)
```

### Key Parameters Explained

| Parameter | Value | Why |
|-----------|-------|-----|
| `num_train_epochs` | 1 | Start small, validate approach |
| `per_device_train_batch_size` | 4 | Fits in A100 memory with QLoRA |
| `gradient_accumulation_steps` | 4 | Effective batch size of 16 |
| `learning_rate` | 2e-4 | Standard for LoRA fine-tuning |
| `warmup_ratio` | 0.03 | 3% of steps for LR warmup |
| `bf16` | True | Use bfloat16 for A100 |
| `optim` | paged_adamw_8bit | Memory-efficient optimizer |
| `gradient_checkpointing` | True | Trade compute for memory |

## Run Command
```bash
cd /Users/saikiranrallabandi/Documents/git-repo/inframind
modal run modal_step4.py
```

## Expected Output

```
[1/5] Loading InfraMind dataset...
  - Loaded 500 tasks

[2/5] Loading model with QLoRA...
  - Model: Qwen/Qwen2.5-0.5B-Instruct
  - Parameters: 494,032,768

[3/5] Adding LoRA adapters...
trainable params: 2,359,296 || all params: 496,392,064 || trainable%: 0.48

[4/5] Setting up training...
  - Epochs: 1
  - Effective batch size: 16
  - Learning rate: 0.0002

[5/5] Starting training...
{'loss': 2.1234, 'learning_rate': 0.0002, 'epoch': 0.1}
{'loss': 1.8765, 'learning_rate': 0.00019, 'epoch': 0.2}
...

Training Complete!
  - Training loss: 1.2345
  - Training time: 180.00s
```

## Training Metrics

### Loss Curve (Expected)
```
Step   Loss
----   ----
10     2.5
50     2.0
100    1.7
150    1.5
200    1.3
250    1.2
```

### Memory Usage
- Base model (4-bit): ~0.25 GB
- LoRA adapters: ~0.01 GB
- Optimizer states: ~0.1 GB
- Activations (with checkpointing): ~2 GB
- **Total**: ~3 GB (fits easily on A100)

## Model Persistence

The trained model is saved to a Modal Volume:
```
/models/qwen-0.5b-inframind/
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

This persists across runs, so you can:
1. Resume training later
2. Run inference without retraining
3. Download the model locally

## Troubleshooting

### Out of Memory
- Reduce `per_device_train_batch_size` to 2
- Reduce `max_seq_length` to 512

### Slow Training
- Ensure GPU is being used (`device_map="auto"`)
- Check gradient accumulation isn't too high

### Poor Results
- Increase epochs to 3
- Check dataset quality
- Verify LoRA is attached to correct modules

## Next Step
[Step 5: Save and Test Fine-tuned Model](./step5_save_test.md)
