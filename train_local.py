#!/usr/bin/env python3
"""
InfraMind Local Training Script
================================
Platform-agnostic training that works on any GPU environment:
- Local GPU (NVIDIA)
- AWS (EC2, SageMaker)
- GCP (Compute Engine, Vertex AI)
- Azure (VMs, Azure ML)
- HuggingFace Spaces
- Google Colab
- Raspberry Pi (inference only)

Usage:
    # GRPO Training
    python train_local.py --method grpo --epochs 3 --output ./models/grpo

    # DAPO Training (from GRPO checkpoint)
    python train_local.py --method dapo --checkpoint ./models/grpo --epochs 2

    # Quick test
    python train_local.py --method grpo --samples 100 --epochs 1

    # With Accelerate (multi-GPU)
    accelerate launch train_local.py --method grpo --epochs 3
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Callable

import torch
import yaml
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from trl import GRPOConfig, GRPOTrainer

# Add inframind to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inframind import create_dataset, IaCReward


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    "model": {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "torch_dtype": "bfloat16",
        "trust_remote_code": True,
    },
    "lora": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    "grpo": {
        "learning_rate": 5e-6,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 3,
        "beta": 0.04,
        "num_generations": 4,
        "temperature": 0.8,
        "max_prompt_length": 256,
        "max_completion_length": 512,
    },
    "dapo": {
        "learning_rate": 5e-6,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": 2,
        "beta": 0.0,  # No KL penalty for pure DAPO
        "num_generations": 8,
        "temperature": 0.9,
        "max_prompt_length": 256,
        "max_completion_length": 768,
    },
}


def load_config(config_path: str = None) -> dict:
    """Load config from YAML file or use defaults"""
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)
    return DEFAULT_CONFIG


# =============================================================================
# REWARD FUNCTION
# =============================================================================

def create_reward_function(categories: List[str] = None) -> Callable:
    """Create reward function for IaC generation"""
    reward_calculator = IaCReward()

    def reward_fn(completions: List[str], prompts: List[str] = None, **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            # Detect category from completion content
            category = detect_category(completion)
            score, _ = reward_calculator.score(completion, category)
            rewards.append(score)
        return rewards

    return reward_fn


def detect_category(text: str) -> str:
    """Detect IaC category from text content"""
    text_lower = text.lower()

    if 'resource "' in text or 'terraform' in text_lower:
        return "terraform"
    elif 'apiVersion:' in text or 'kind:' in text:
        return "kubernetes"
    elif 'FROM ' in text.upper() or 'services:' in text:
        return "docker"
    elif 'jobs:' in text or 'stages:' in text or 'pipeline' in text_lower:
        return "cicd"
    elif any(k in text_lower for k in ['mlflow', 'sagemaker', 'kubeflow', 'ray']):
        return "mlops"
    else:
        return "terraform"  # default


# =============================================================================
# DATASET PREPARATION
# =============================================================================

def prepare_dataset(num_samples: int = 500, categories: List[str] = None) -> Dataset:
    """Prepare dataset in format expected by TRL"""
    bench = create_dataset(categories=categories, size=num_samples)

    # Convert to prompts
    data = []
    for task in bench:
        prompt = f"### Instruction:\n{task['instruction']}\n"
        if task.get('input'):
            prompt += f"### Input:\n{task['input']}\n"
        prompt += "### Response:\n"

        data.append({
            "prompt": prompt,
            "category": task.get("category", "unknown"),
        })

    return Dataset.from_list(data)


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(config: dict, checkpoint: str = None):
    """Load model with optional checkpoint"""
    model_config = config["model"]

    # Determine dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(model_config.get("torch_dtype", "bfloat16"), torch.bfloat16)

    # Check device capabilities
    if not torch.cuda.is_available():
        print("WARNING: No GPU detected. Training will be slow.")
        torch_dtype = torch.float32
        device_map = "cpu"
    else:
        device_map = "auto"
        # Check if bf16 is supported
        if torch_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            print("WARNING: bfloat16 not supported, using float16")
            torch_dtype = torch.float16

    print(f"Loading model: {model_config['model_id']}")
    print(f"  dtype: {torch_dtype}")
    print(f"  device_map: {device_map}")

    if checkpoint and os.path.exists(checkpoint):
        # Load from checkpoint (merge LoRA weights)
        print(f"Loading from checkpoint: {checkpoint}")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_config["model_id"],
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=model_config.get("trust_remote_code", True),
        )
        model = PeftModel.from_pretrained(base_model, checkpoint)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_config["model_id"],
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=model_config.get("trust_remote_code", True),
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_id"],
        trust_remote_code=model_config.get("trust_remote_code", True),
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def apply_lora(model, config: dict):
    """Apply LoRA adapters to model"""
    lora_config = config["lora"]

    peft_config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        bias=lora_config["bias"],
        task_type=lora_config["task_type"],
        target_modules=lora_config["target_modules"],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, peft_config


# =============================================================================
# TRAINING
# =============================================================================

def train_grpo(
    config: dict,
    dataset: Dataset,
    output_dir: str,
    checkpoint: str = None,
    num_epochs: int = None,
):
    """Train with GRPO (Group Relative Policy Optimization)"""
    print("\n" + "="*60)
    print("GRPO TRAINING")
    print("="*60)

    grpo_config = config["grpo"]

    # Load model
    model, tokenizer = load_model(config, checkpoint)
    model, peft_config = apply_lora(model, config)

    # Training arguments
    training_args = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs or grpo_config["num_train_epochs"],
        per_device_train_batch_size=grpo_config["per_device_train_batch_size"],
        gradient_accumulation_steps=grpo_config["gradient_accumulation_steps"],
        learning_rate=grpo_config["learning_rate"],
        beta=grpo_config["beta"],
        num_generations=grpo_config["num_generations"],
        temperature=grpo_config["temperature"],
        max_prompt_length=grpo_config["max_prompt_length"],
        max_completion_length=grpo_config["max_completion_length"],
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to="none",
        seed=42,
    )

    # Create reward function
    reward_fn = create_reward_function()

    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        peft_config=peft_config,
    )

    # Train
    print(f"\nStarting training...")
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")

    trainer.train()

    # Save
    print(f"\nSaving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return trainer


def train_dapo(
    config: dict,
    dataset: Dataset,
    output_dir: str,
    checkpoint: str = None,
    num_epochs: int = None,
):
    """Train with DAPO (Direct Advantage Policy Optimization)

    DAPO innovations:
    1. Clip-Higher: Asymmetric clipping for exploration
    2. Dynamic Sampling: Skip uniform reward batches
    3. Token-Level Loss: Per-token gradients
    4. Overlong Punishment: Soft length penalty
    """
    print("\n" + "="*60)
    print("DAPO TRAINING")
    print("="*60)

    dapo_config = config["dapo"]

    if not checkpoint:
        print("WARNING: DAPO typically starts from a GRPO checkpoint")
        print("         Training from base model instead...")

    # Load model
    model, tokenizer = load_model(config, checkpoint)
    model, peft_config = apply_lora(model, config)

    # Training arguments (DAPO uses GRPO trainer with different hyperparams)
    training_args = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs or dapo_config["num_train_epochs"],
        per_device_train_batch_size=dapo_config["per_device_train_batch_size"],
        gradient_accumulation_steps=dapo_config["gradient_accumulation_steps"],
        learning_rate=dapo_config["learning_rate"],
        beta=dapo_config["beta"],  # 0 for pure DAPO
        num_generations=dapo_config["num_generations"],
        temperature=dapo_config["temperature"],
        max_prompt_length=dapo_config["max_prompt_length"],
        max_completion_length=dapo_config["max_completion_length"],
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to="none",
        seed=42,
    )

    # Create reward function
    reward_fn = create_reward_function()

    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        peft_config=peft_config,
    )

    # Train
    print(f"\nStarting DAPO training...")
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Beta (KL): {training_args.beta} (0 = pure DAPO)")
    print(f"  Generations per prompt: {training_args.num_generations}")
    print(f"  Temperature: {training_args.temperature}")

    trainer.train()

    # Save
    print(f"\nSaving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return trainer


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(model_path: str, num_samples: int = 110):
    """Evaluate trained model on test set"""
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)

    config = load_config()

    # Load model
    model, tokenizer = load_model(config, model_path)
    model.eval()

    # Prepare test data
    dataset = prepare_dataset(num_samples=num_samples)
    reward_fn = create_reward_function()

    passed = 0
    total = 0

    print(f"\nEvaluating on {len(dataset)} samples...")

    for i, sample in enumerate(dataset):
        prompt = sample["prompt"]

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = response[len(prompt):]

        # Score
        reward = reward_fn([completion])[0]

        if reward >= 0.6:
            passed += 1
        total += 1

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(dataset)} - Accuracy: {passed/total*100:.1f}%")

    accuracy = passed / total * 100
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  Passed: {passed}/{total}")
    print(f"  Accuracy: {accuracy:.1f}%")

    return accuracy


# =============================================================================
# INFERENCE
# =============================================================================

def generate(model_path: str, prompt: str):
    """Generate IaC from prompt"""
    config = load_config()
    model, tokenizer = load_model(config, model_path)
    model.eval()

    # Format prompt
    formatted = f"### Instruction:\n{prompt}\n### Response:\n"

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(formatted):]


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="InfraMind Training - Platform Agnostic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # GRPO Training
  python train_local.py --method grpo --epochs 3 --output ./models/grpo

  # DAPO Training (from GRPO checkpoint)
  python train_local.py --method dapo --checkpoint ./models/grpo --output ./models/dapo

  # Quick test with 100 samples
  python train_local.py --method grpo --samples 100 --epochs 1

  # Evaluate trained model
  python train_local.py --evaluate ./models/grpo

  # Generate IaC
  python train_local.py --generate ./models/grpo --prompt "Create Terraform for AWS EC2"

  # With HuggingFace Accelerate (multi-GPU)
  accelerate launch train_local.py --method grpo --epochs 3
        """
    )

    # Training options
    parser.add_argument("--method", choices=["grpo", "dapo"], help="Training method")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--samples", type=int, default=500, help="Number of training samples")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint (for DAPO)")
    parser.add_argument("--output", type=str, default="./models/inframind", help="Output directory")
    parser.add_argument("--config", type=str, help="Path to config YAML file")

    # Evaluation options
    parser.add_argument("--evaluate", type=str, metavar="MODEL_PATH", help="Evaluate a trained model")
    parser.add_argument("--test-samples", type=int, default=110, help="Number of test samples")

    # Generation options
    parser.add_argument("--generate", type=str, metavar="MODEL_PATH", help="Generate with trained model")
    parser.add_argument("--prompt", type=str, help="Prompt for generation")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Evaluation mode
    if args.evaluate:
        evaluate(args.evaluate, args.test_samples)
        return

    # Generation mode
    if args.generate:
        if not args.prompt:
            parser.error("--prompt required with --generate")
        result = generate(args.generate, args.prompt)
        print("\n" + "="*60)
        print("GENERATED OUTPUT")
        print("="*60)
        print(result)
        return

    # Training mode
    if not args.method:
        parser.print_help()
        return

    # Prepare dataset
    print("Preparing dataset...")
    dataset = prepare_dataset(num_samples=args.samples)

    # Train
    if args.method == "grpo":
        train_grpo(
            config=config,
            dataset=dataset,
            output_dir=args.output,
            checkpoint=args.checkpoint,
            num_epochs=args.epochs,
        )
    elif args.method == "dapo":
        train_dapo(
            config=config,
            dataset=dataset,
            output_dir=args.output,
            checkpoint=args.checkpoint,
            num_epochs=args.epochs,
        )

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Model saved to: {args.output}")
    print(f"\nTo evaluate: python train_local.py --evaluate {args.output}")
    print(f"To generate: python train_local.py --generate {args.output} --prompt 'Create Terraform for S3'")


if __name__ == "__main__":
    main()
