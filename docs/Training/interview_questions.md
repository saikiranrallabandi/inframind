# LLM Fine-tuning Interview Questions & Answers

## Step 1: Loading Base Models

### Q1: Why did you choose Qwen2.5-0.5B-Instruct as your base model?
**Answer:**
- **Size efficiency**: 0.5B parameters is small enough to iterate quickly during development
- **Instruction-tuned**: Already trained on instruction-following, so it understands our prompt format
- **Strong code understanding**: Qwen series has excellent code generation capabilities
- **Apache 2.0 license**: Can be used commercially and modified freely

### Q2: What does `device_map="auto"` do in `from_pretrained()`?
**Answer:**
`device_map="auto"` automatically distributes model layers across available GPUs. It:
- Analyzes available GPU memory
- Splits model layers optimally across devices
- Handles multi-GPU scenarios automatically
- Falls back to CPU for layers that don't fit

### Q3: Why use `torch_dtype=torch.bfloat16` instead of float32?
**Answer:**
- **Memory**: bfloat16 uses 2 bytes vs 4 bytes for float32 (50% reduction)
- **Speed**: Faster matrix operations on modern GPUs (A100, H100)
- **Range**: Unlike float16, bfloat16 has same exponent range as float32, preventing overflow
- **Quality**: Minimal accuracy loss for inference and training

### Q4: What was wrong with the base model's Terraform output before fine-tuning?
**Answer:**
The base model generated syntactically incorrect Terraform:
- Used `name` instead of `bucket` attribute
- Invented non-existent syntax: `aws_versioning::versioning()`
- Created fake resources: `aws_s3_bucket_versioning_policy`
- Confused AWS provider patterns with other languages

This shows the model has general coding knowledge but lacks domain-specific IaC expertise.

---

## Step 2: QLoRA Quantization

### Q5: What is quantization and why is it important for LLM training?
**Answer:**
Quantization reduces the precision of model weights to use less memory:
- **FP32**: 32 bits per weight (baseline)
- **FP16/BF16**: 16 bits per weight (2x reduction)
- **INT8**: 8 bits per weight (4x reduction)
- **INT4/NF4**: 4 bits per weight (8x reduction)

**Importance for training:**
- Allows training larger models on limited GPU memory
- Enables fine-tuning on consumer GPUs (RTX 3090, 4090)
- Reduces cloud GPU costs significantly

### Q6: Explain the difference between QLoRA and regular LoRA.
**Answer:**

| Aspect | LoRA | QLoRA |
|--------|------|-------|
| Base model precision | FP16/BF16 | 4-bit quantized |
| Memory for 7B model | ~14GB | ~4GB |
| Training speed | Faster | Slightly slower |
| Quality | Baseline | ~99% of LoRA |

**QLoRA = Quantized base model + LoRA adapters in FP16**

The key insight is that you can train FP16 LoRA adapters on top of a frozen 4-bit base model.

### Q7: What is NF4 (NormalFloat4) quantization? Why is it better than regular INT4?
**Answer:**
NF4 is specifically designed for normally-distributed neural network weights:

- **INT4**: Uniform quantization levels (equal spacing)
- **NF4**: Non-uniform levels matching Gaussian distribution of weights

Since neural network weights follow a normal distribution, NF4:
- Allocates more precision near zero (where most weights are)
- Uses fewer bits for extreme values
- Achieves better accuracy with same 4 bits

### Q8: What does `bnb_4bit_use_double_quant=True` do?
**Answer:**
Double quantization quantizes the quantization constants themselves:

1. **First quantization**: Weights → 4-bit with scaling constants
2. **Second quantization**: Scaling constants → 8-bit

This saves additional memory (~0.4 bits per parameter) with negligible quality loss. Especially important for large models where the constants add up.

### Q9: How much memory did QLoRA save in your project?
**Answer:**
For Qwen2.5-0.5B:
- **FP16**: ~0.92 GB
- **4-bit QLoRA**: ~0.23 GB
- **Savings**: ~75%

For larger models, savings are even more dramatic:
- 7B model: 14GB → 4GB
- 70B model: 140GB → 35GB

---

## Step 3: LoRA Adapters

### Q10: Explain how LoRA (Low-Rank Adaptation) works.
**Answer:**
LoRA adds trainable low-rank matrices alongside frozen pretrained weights:

```
Original forward pass: y = Wx
LoRA forward pass:     y = Wx + BAx

Where:
- W is frozen (d × k dimensions)
- A is trainable (r × k dimensions, r << d)
- B is trainable (d × r dimensions)
```

**Key insight**: Instead of updating all d×k parameters in W, we only train r×(d+k) parameters. With r=32 and d,k=4096, that's:
- Full: 4096 × 4096 = 16.7M params
- LoRA: 32 × (4096 + 4096) = 262K params (64x reduction)

### Q11: What does the "rank" (r) parameter control in LoRA?
**Answer:**
Rank controls the expressiveness of the adaptation:

| Rank | Parameters | Capacity | Use Case |
|------|------------|----------|----------|
| 4-8 | Very low | Limited | Simple style transfer |
| 16-32 | Low | Moderate | Domain adaptation |
| 64-128 | Medium | High | Complex new tasks |
| 256+ | High | Very high | Near full fine-tuning |

**Trade-offs:**
- Higher rank = more parameters = better quality but slower training
- Lower rank = fewer parameters = risk of underfitting

For IaC domain adaptation, rank=32 provides good balance.

### Q12: Why is `lora_alpha` typically set to 2× the rank?
**Answer:**
`lora_alpha` is a scaling factor applied to LoRA outputs:

```
ΔW = (lora_alpha / r) × BA
```

With alpha=2×r, the scaling becomes 2, which:
- Gives LoRA updates appropriate magnitude relative to base weights
- Was found empirically to work well across many tasks
- Higher alpha = stronger adaptation effect
- Lower alpha = more conservative updates

### Q13: What are the "target_modules" and why did you choose those specific ones?
**Answer:**
Target modules are the layers where LoRA adapters are inserted:

**Attention layers** (most important for learning patterns):
- `q_proj`: Query - "what to look for"
- `k_proj`: Key - "what to match against"
- `v_proj`: Value - "what information to retrieve"
- `o_proj`: Output - "how to combine attention heads"

**MLP layers** (store factual knowledge):
- `gate_proj`: Controls information flow
- `up_proj`: Expands to higher dimension
- `down_proj`: Projects back down

**Why all 7?** Research shows adapting both attention AND MLP gives best results for domain adaptation tasks like IaC generation.

### Q14: What does `prepare_model_for_kbit_training()` do?
**Answer:**
This function prepares a quantized model for backpropagation:

1. **Enables gradient checkpointing**: Trades compute for memory by recomputing activations during backward pass
2. **Casts LayerNorm to FP32**: Normalization layers need higher precision for stability
3. **Casts embeddings to FP32**: Input embeddings also need higher precision
4. **Enables input gradients**: Allows gradients to flow through the model

Without this, training a quantized model would fail or produce poor results.

### Q15: How many parameters are actually trained with your LoRA configuration?
**Answer:**
With rank=32 on Qwen2.5-0.5B:
- **Total parameters**: 494,032,768 (~494M)
- **Trainable (LoRA)**: ~2,359,296 (~2.4M)
- **Percentage**: 0.48%

We're training less than 0.5% of the model while achieving most of the benefit of full fine-tuning.

---

## General/Conceptual Questions

### Q16: What is the difference between SFT, RLHF, and GRPO?
**Answer:**

| Method | Training Signal | Use Case |
|--------|-----------------|----------|
| **SFT** | Input-output pairs | Teaching specific outputs |
| **RLHF** | Human preference rankings | Alignment with human values |
| **GRPO** | Reward function | Optimizing for verifiable metrics |

**Your project uses:**
- SFT for initial domain training (IaC examples)
- GRPO with reward functions for syntax/correctness optimization

### Q17: Why use Alpaca format for your dataset?
**Answer:**
Alpaca format (`instruction`, `input`, `output`) is:
- **Simple**: Easy to create and parse
- **Flexible**: `input` field optional for context
- **Compatible**: Works with most training frameworks
- **Standard**: Many models are pre-trained on this format

```json
{
  "instruction": "Create Terraform for S3 bucket",
  "input": "with versioning enabled",
  "output": "resource \"aws_s3_bucket\" {...}"
}
```

### Q18: How would you evaluate if your fine-tuned model is better than the base model?
**Answer:**
Multiple evaluation approaches:

1. **Automatic metrics** (your IaCReward):
   - Syntax correctness (valid HCL/YAML)
   - Keyword presence (AWS resources, K8s kinds)
   - Format completeness (balanced braces)

2. **LLM-as-Judge**:
   - Use GPT-4/Claude to rate outputs
   - More nuanced than keyword matching

3. **Human evaluation**:
   - DevOps engineers rate quality
   - Check if code actually works

4. **Execution tests**:
   - `terraform validate` on outputs
   - `kubectl apply --dry-run` for K8s

### Q19: What challenges did you face during this project?
**Answer:**
1. **Version compatibility**: Transformers + bitsandbytes versions must match. Fixed by pinning specific versions.

2. **Modal image building**: Needed `apt_install("git")` before pip installing from GitHub.

3. **Dataset quality**: Base model generates plausible but incorrect IaC. Need high-quality examples.

4. **Reward design**: Simple keyword matching isn't enough - need semantic understanding.

### Q20: How would you deploy this fine-tuned model in production?
**Answer:**
Options:
1. **Merged weights**: Merge LoRA into base model, deploy as single model
2. **Separate adapters**: Load base model once, swap LoRA adapters per task
3. **GGUF conversion**: Quantize for CPU inference with llama.cpp
4. **vLLM/TGI**: High-throughput GPU serving

For InfraMind:
```python
# Merge LoRA into base
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./qwen-inframind-merged")
```
