# Step 4: Train with SFT (Supervised Fine-Tuning) using TRL Best Practices
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "transformers>=4.45.0",
        "accelerate>=0.34.0",
        "bitsandbytes",
        "peft>=0.12.0",
        "trl>=0.12.0",  # Newer TRL for SFTConfig
        "datasets",
        "git+https://github.com/saikiranrallabandi/inframind.git",
    )
)

app = modal.App("inframind-step4", image=image)

# Create volume to persist model
volume = modal.Volume.from_name("inframind-models", create_if_missing=True)


@app.function(gpu="A100", timeout=3600, volumes={"/models": volume})
def train_sft():
    """Train the model with SFT using LoRA - TRL Best Practices"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset
    from inframind import create_dataset

    print("=" * 60)
    print("STEP 4: SFT Training with LoRA")
    print("=" * 60)

    # =========================================
    # Part 1: Load and Prepare Dataset
    # =========================================
    print("\n[1/5] Loading InfraMind dataset...")
    dataset = create_dataset()  # Load ALL tasks (2000+)
    print(f"  - Loaded {len(dataset)} tasks")

    # Format for SFT
    def format_prompt(task):
        instruction = task.get("instruction", "")
        input_text = task.get("input", "")
        output = task.get("output", "")

        if input_text:
            return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

    hf_data = [{"text": format_prompt(t)} for t in dataset]
    hf_dataset = Dataset.from_list(hf_data)
    print(f"  - Formatted dataset: {len(hf_dataset)} examples")

    # =========================================
    # Part 2: Load Model with QLoRA
    # =========================================
    print("\n[2/5] Loading model with QLoRA...")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"  - Model: {model_name}")
    print(f"  - Parameters: {model.num_parameters():,}")

    # =========================================
    # Part 3: Configure LoRA (passed to SFTTrainer)
    # =========================================
    print("\n[3/5] Configuring LoRA adapters...")

    # LoRA config - will be applied by SFTTrainer automatically
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    print(f"  LoRA Config:")
    print(f"  - Rank (r): {peft_config.r}")
    print(f"  - Alpha: {peft_config.lora_alpha}")
    print(f"  - Target modules: {peft_config.target_modules}")

    # =========================================
    # Part 4: SFTConfig (TRL Best Practices)
    # =========================================
    print("\n[4/5] Setting up training with SFTConfig...")

    # Using SFTConfig instead of TrainingArguments (TRL recommended)
    sft_config = SFTConfig(
        output_dir="/models/checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,  # Higher LR for LoRA (TRL recommended)
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="none",
        max_grad_norm=1.0,
        # SFT specific settings
        max_seq_length=1024,
        packing=True,  # Pack multiple examples for efficiency (TRL recommended)
        dataset_text_field="text",  # Field containing formatted text
    )

    print(f"  Training Configuration:")
    print(f"  - Epochs: {sft_config.num_train_epochs}")
    print(f"  - Batch size: {sft_config.per_device_train_batch_size}")
    print(f"  - Gradient accumulation: {sft_config.gradient_accumulation_steps}")
    print(f"  - Effective batch size: {sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps}")
    print(f"  - Learning rate: {sft_config.learning_rate}")
    print(f"  - Packing: {sft_config.packing} (TRL best practice)")
    print(f"  - Max seq length: {sft_config.max_seq_length}")

    # =========================================
    # Part 5: Train with SFTTrainer!
    # =========================================
    print("\n[5/5] Starting training...")
    print("=" * 60)

    # SFTTrainer handles LoRA setup automatically when peft_config is passed
    trainer = SFTTrainer(
        model=model,
        train_dataset=hf_dataset,
        args=sft_config,
        processing_class=tokenizer,
        peft_config=peft_config,  # TRL applies LoRA automatically
    )

    # Train
    train_result = trainer.train()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"  - Training loss: {train_result.training_loss:.4f}")
    print(f"  - Training time: {train_result.metrics['train_runtime']:.2f}s")
    print(f"  - Samples/second: {train_result.metrics['train_samples_per_second']:.2f}")

    # =========================================
    # Save Model
    # =========================================
    print("\nSaving model...")
    output_path = "/models/qwen-0.5b-inframind"
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Commit volume to persist
    volume.commit()

    print(f"  - Model saved to {output_path}")

    # =========================================
    # Test Generation
    # =========================================
    print("\n" + "=" * 60)
    print("Testing fine-tuned model...")
    print("=" * 60)

    model.eval()
    test_prompts = [
        "Create Terraform for AWS S3 bucket with versioning",
        "Create Kubernetes Deployment with 3 replicas",
        "Write a Dockerfile for Python Flask app",
    ]

    for prompt in test_prompts:
        formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nPrompt: {prompt}")
        print(f"Response:\n{response[len(formatted):][:500]}...")
        print("-" * 40)

    print("\n" + "=" * 60)
    print("Step 4 Complete!")
    print("=" * 60)

    return {
        "status": "success",
        "training_loss": train_result.training_loss,
        "training_time": train_result.metrics['train_runtime'],
        "model_path": output_path,
    }


@app.local_entrypoint()
def main():
    result = train_sft.remote()
    print(f"\nResult: {result}")
