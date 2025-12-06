# Step 3: Add LoRA Adapters to the Quantized Model
import modal
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "transformers==4.45.0",
        "accelerate==0.34.0",
        "bitsandbytes",
        "peft==0.12.0",  # For LoRA
        "datasets",
        "git+https://github.com/saikiranrallabandi/inframind.git",
    )
)

app = modal.App("inframind-step3", image=image)


@app.function(gpu="A100", timeout=600)
def add_lora_adapters():
    """Add LoRA adapters to the quantized model"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset
    from inframind import create_dataset

    print("=" * 60)
    print("STEP 3: Add LoRA Adapters")
    print("=" * 60)

    # =========================================
    # Part 1: Load Dataset
    # =========================================
    print("\n[1/4] Loading InfraMind dataset...")
    dataset = create_dataset(size=100)
    print(f"  - Loaded {len(dataset)} tasks")

    # =========================================
    # Part 2: Load Model with QLoRA
    # =========================================
    print("\n[2/4] Loading model with QLoRA...")
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

    print(f"  - Model loaded: {model_name}")
    print(f"  - Total parameters: {model.num_parameters():,}")

    # =========================================
    # Part 3: Prepare for k-bit training
    # =========================================
    print("\n[3/4] Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)
    print("  - Gradient checkpointing enabled")
    print("  - Model prepared for training")

    # =========================================
    # Part 4: Configure and Apply LoRA
    # =========================================
    print("\n[4/4] Configuring LoRA adapters...")

    # LoRA Configuration
    lora_config = LoraConfig(
        r=32,                    # Rank - higher = more parameters, better quality
        lora_alpha=64,           # Scaling factor (usually 2x rank)
        lora_dropout=0.05,       # Dropout for regularization
        bias="none",             # Don't train bias terms
        task_type="CAUSAL_LM",   # Causal language modeling
        target_modules=[         # Which layers to adapt
            "q_proj",            # Query projection
            "k_proj",            # Key projection
            "v_proj",            # Value projection
            "o_proj",            # Output projection
            "gate_proj",         # MLP gate
            "up_proj",           # MLP up projection
            "down_proj",         # MLP down projection
        ],
    )

    print(f"  LoRA Configuration:")
    print(f"  - Rank (r): {lora_config.r}")
    print(f"  - Alpha: {lora_config.lora_alpha}")
    print(f"  - Dropout: {lora_config.lora_dropout}")
    print(f"  - Target modules: {lora_config.target_modules}")

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    print("\n" + "=" * 60)
    print("Trainable Parameters Summary")
    print("=" * 60)
    model.print_trainable_parameters()

    # Calculate actual numbers
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_percent = 100 * trainable_params / total_params

    print(f"\n  Breakdown:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable (LoRA): {trainable_params:,}")
    print(f"  - Frozen (base): {total_params - trainable_params:,}")
    print(f"  - Training: {trainable_percent:.2f}% of model")

    # =========================================
    # Test that model still works
    # =========================================
    print("\n" + "=" * 60)
    print("Testing model with LoRA adapters...")
    print("=" * 60)

    prompt = "### Instruction:\nCreate Terraform for AWS S3 bucket with versioning\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated (with untrained LoRA):\n{response}")

    print("\n" + "=" * 60)
    print("Step 3 Complete!")
    print("=" * 60)
    print("\nNote: LoRA adapters are added but NOT trained yet.")
    print("The output should be similar to base model.")
    print("Next step: Train the LoRA adapters!")

    return {
        "status": "success",
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percent": f"{trainable_percent:.2f}%",
        "lora_rank": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
    }


@app.local_entrypoint()
def main():
    result = add_lora_adapters.remote()
    print(f"\nResult: {result}")
