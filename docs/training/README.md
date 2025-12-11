# InfraMind Training Guide

Fine-tuning small language models for Infrastructure-as-Code tasks with multi-step reasoning.

## Training Pipeline

```
┌────────────────────────────────────────────────────────────────────┐
│                    InfraMind Training Pipeline                      │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Phase 1: SFT (Steps 1-4)        Phase 2: RL (Steps 5-7)           │
│  ─────────────────────────       ──────────────────────────        │
│                                                                     │
│  ┌─────┐   ┌─────┐   ┌─────┐   ┌─────────┐   ┌──────┐   ┌──────┐  │
│  │Load │──▶│QLoRA│──▶│LoRA │──▶│Reasoning│──▶│ GRPO │──▶│ DAPO │  │
│  │Model│   │     │   │     │   │ Dataset │   │      │   │      │  │
│  └─────┘   └─────┘   └─────┘   └─────────┘   └──────┘   └──────┘  │
│    (1)       (2)       (3)         (5)          (6)        (7)     │
│                 │                                   │               │
│                 ▼                                   ▼               │
│           ┌─────────┐                         ┌─────────┐          │
│           │   SFT   │                         │LLM Judge│          │
│           │Training │                         │  Eval   │          │
│           └─────────┘                         └─────────┘          │
│               (4)                                 (8)               │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

## Training Steps

### Phase 1: Supervised Fine-Tuning (SFT)

| Step | File | Status | Description |
|------|------|--------|-------------|
| 1 | [step1_load_model.md](step1_load_model.md) | ✅ Done | Load base model, verify setup |
| 2 | [step2_qlora_dataset.md](step2_qlora_dataset.md) | ✅ Done | QLoRA quantization + dataset |
| 3 | [step3_lora_adapters.md](step3_lora_adapters.md) | ✅ Done | Configure LoRA adapters |
| 4 | [step4_sft_training.md](step4_sft_training.md) | ✅ Done | Run SFT training |

### Phase 2: Reinforcement Learning (DAPO)

| Step | File | Status | Description |
|------|------|--------|-------------|
| 5 | [step5_multistep_reasoning.md](step5_multistep_reasoning.md) | ⏳ Next | Add `<think>` reasoning traces |
| 6 | [step6_grpo_training.md](step6_grpo_training.md) | ⏳ Pending | Group Relative Policy Optimization |
| 7 | [step7_dapo_training.md](step7_dapo_training.md) | ⏳ Pending | Full DAPO (Clip-Higher, Dynamic Sampling, etc.) |
| 8 | [step8_llm_judge.md](step8_llm_judge.md) | ⏳ Pending | LLM-as-Judge evaluation |

## Quick Start

```bash
# Phase 1: SFT Training (completed)
modal run modal_app.py

# Phase 2: DAPO Training (coming)
modal run modal_step5.py  # Reasoning dataset
modal run modal_step6.py  # GRPO
modal run modal_step7.py  # DAPO
python scripts/evaluate.py  # LLM-as-Judge
```

## Current Results

| Model | Params | Training | Accuracy |
|-------|--------|----------|----------|
| Qwen2.5-0.5B (base) | 0.5B | - | 28% |
| Qwen2.5-0.5B + SFT | 0.5B | 12s | 52% |
| Qwen2.5-0.5B + DAPO | 0.5B | ~6h | TBD |

## References

- [DAPO Paper (arXiv:2412.18279)](https://arxiv.org/html/2412.18279v1)
- [ByteDance DAPO (arXiv:2503.14476)](https://arxiv.org/abs/2503.14476)
- [DAPO GitHub](https://github.com/BytedTsinghua-SIA/DAPO)
- [DeepSeekMath GRPO](https://arxiv.org/abs/2402.03300)

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  InfraMind Training                  │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │  Base    │───▶│  QLoRA   │───▶│  LoRA    │      │
│  │  Model   │    │  4-bit   │    │ Adapters │      │
│  └──────────┘    └──────────┘    └──────────┘      │
│       │                               │             │
│       ▼                               ▼             │
│  Qwen2.5-0.5B              Trainable ~1% params    │
│  (500M params)             (~5M params)            │
│                                                      │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────────────────────────────────────────┐  │
│  │              SFT Training (TRL)               │  │
│  │                                                │  │
│  │  Dataset: InfraMind (2000+ IaC tasks)        │  │
│  │  • Terraform                                  │  │
│  │  • Kubernetes                                 │  │
│  │  • Docker                                     │  │
│  │  • CI/CD                                      │  │
│  │  • MLOps                                      │  │
│  └──────────────────────────────────────────────┘  │
│                                                      │
└─────────────────────────────────────────────────────┘
```

## Key Technologies

- **Modal.com**: Cloud GPU (A100) for training
- **QLoRA**: 4-bit quantization for memory efficiency
- **LoRA**: Parameter-efficient fine-tuning
- **TRL**: HuggingFace's training library for LLMs
- **PEFT**: Parameter-Efficient Fine-Tuning library

## Memory Requirements

| Component | VRAM Usage |
|-----------|------------|
| Base model (4-bit) | ~250 MB |
| LoRA adapters | ~20 MB |
| Optimizer states | ~100 MB |
| Activations | ~500 MB |
| **Total** | **~1 GB** |

Fits easily on A100 (40GB) with room for large batches!
