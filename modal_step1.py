# Step 1: Load and test the base model on Modal
import modal

# Define image with dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "transformers==4.44.2",
    "accelerate",
)

app = modal.App("inframind-step1", image=image)


@app.function(gpu="A100", timeout=300)
def load_model():
    """Load Qwen model and test it on an IaC prompt"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading {model_name}...")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"✓ Model loaded!")
    print(f"  Parameters: {model.num_parameters():,}")
    print(f"  Device: {model.device}")

    # Test with an IaC prompt
    prompt = "### Instruction:\nCreate Terraform for AWS S3 bucket with versioning\n\n### Response:\n"

    print(f"\n--- Testing generation ---")
    print(f"Prompt: {prompt}")

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
    print(f"\nGenerated:\n{response}")

    return "Step 1 complete!"


@app.local_entrypoint()
def main():
    result = load_model.remote()
    print(result)
