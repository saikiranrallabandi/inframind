# Step 2: QLoRA Quantization + Load InfraMind Dataset
import modal
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")  # Required for pip install from git
    .pip_install(
        "torch",
        "transformers==4.45.0",  # Newer version for bitsandbytes compatibility
        "accelerate==0.34.0",
        "bitsandbytes",
        "datasets",
        "git+https://github.com/saikiranrallabandi/inframind.git",
    )
)

app = modal.App("inframind-step2", image=image)


@app.function(gpu="A100", timeout=600)
def load_with_qlora():
    """Load model with QLoRA (4-bit quantization) and InfraMind dataset"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from datasets import Dataset
    from inframind import create_dataset

    print("=" * 60)
    print("STEP 2: QLoRA Quantization + InfraMind Dataset")
    print("=" * 60)

    # =========================================
    # Part 1: Load InfraMind Dataset
    # =========================================
    print("\n[1/3] Loading InfraMind dataset...")

    dataset = create_dataset(size=100)  # Start with 100 for testing
    print(f"  - Total tasks: {len(dataset)}")

    # Count categories
    categories = {}
    for task in dataset:
        cat = task.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    print(f"  - Categories: {categories}")
    print(f"  - Sample task:")
    print(f"    Instruction: {dataset[0]['instruction'][:60]}...")
    print(f"    Input: {dataset[0]['input'][:40]}...")

    # =========================================
    # Part 2: Format dataset for SFT
    # =========================================
    print("\n[2/3] Formatting dataset for SFT training...")

    def format_prompt(task):
        instruction = task.get("instruction", "")
        input_text = task.get("input", "")
        output = task.get("output", "")

        if input_text:
            text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        return text

    hf_data = [{"text": format_prompt(t), "category": t.get("category", "unknown")} for t in dataset]
    hf_dataset = Dataset.from_list(hf_data)

    print(f"  - HuggingFace Dataset: {hf_dataset}")
    print(f"  - Sample formatted text:")
    print(f"    {hf_data[0]['text'][:150]}...")

    # =========================================
    # Part 3: Load Model with QLoRA
    # =========================================
    print("\n[3/3] Loading model with QLoRA (4-bit quantization)...")

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    # QLoRA config - 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,              # Use 4-bit quantization
        bnb_4bit_quant_type="nf4",      # NormalFloat4 quantization
        bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in bfloat16
        bnb_4bit_use_double_quant=True, # Nested quantization for more savings
    )

    print(f"  - Model: {model_name}")
    print(f"  - Quantization: 4-bit NF4")
    print(f"  - Compute dtype: bfloat16")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Calculate memory savings
    param_count = model.num_parameters()
    # 4-bit = 0.5 bytes per param, vs 2 bytes for fp16
    memory_fp16 = (param_count * 2) / (1024**3)  # GB
    memory_4bit = (param_count * 0.5) / (1024**3)  # GB

    print(f"\n  Memory Comparison:")
    print(f"  - Parameters: {param_count:,}")
    print(f"  - FP16 would use: ~{memory_fp16:.2f} GB")
    print(f"  - 4-bit uses: ~{memory_4bit:.2f} GB")
    print(f"  - Savings: ~{(1 - memory_4bit/memory_fp16)*100:.0f}%")

    # =========================================
    # Test generation with quantized model
    # =========================================
    print("\n" + "=" * 60)
    print("Testing quantized model generation...")
    print("=" * 60)

    prompt = "### Instruction:\nCreate Terraform for AWS S3 bucket with versioning\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated (4-bit quantized):\n{response}")

    print("\n" + "=" * 60)
    print("Step 2 Complete!")
    print("=" * 60)

    return {
        "status": "success",
        "dataset_size": len(dataset),
        "categories": categories,
        "model_params": param_count,
        "quantization": "4-bit NF4",
    }


@app.local_entrypoint()
def main():
    result = load_with_qlora.remote()
    print(f"\nResult: {result}")
