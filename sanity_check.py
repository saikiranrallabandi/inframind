"""
Sanity check - Generate 5 samples from GRPO model for manual verification.

Usage:
    modal run sanity_check.py
"""

import modal

app = modal.App("inframind-sanity-check")

model_volume = modal.Volume.from_name("inframind-models", create_if_missing=True)
MODEL_DIR = "/models"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers==4.46.0",
        "peft==0.14.0",
        "accelerate>=0.34.0",
    )
)

# 5 diverse test prompts for manual verification
TEST_PROMPTS = [
    {"instruction": "Create a Kubernetes Deployment for nginx with 3 replicas", "category": "kubernetes"},
    {"instruction": "Write Terraform code for an AWS S3 bucket with versioning enabled", "category": "terraform"},
    {"instruction": "Create a Dockerfile for a Python FastAPI application", "category": "dockerfile"},
    {"instruction": "Write a GitHub Actions workflow that runs tests on push to main", "category": "github-actions"},
    {"instruction": "Create a docker-compose.yml for a web app with PostgreSQL and Redis", "category": "docker-compose"},
]


@app.function(gpu="A100", image=image, timeout=600, volumes={MODEL_DIR: model_volume})
def generate_samples():
    """Generate 5 samples for manual verification."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import os

    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    GRPO_PATH = "/models/inframind-grpo/final"

    print("Loading GRPO model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    if os.path.exists(GRPO_PATH):
        model = PeftModel.from_pretrained(model, GRPO_PATH)
        print("GRPO adapters loaded")
    else:
        print("WARNING: No GRPO model found!")
        return

    model.eval()

    results = []

    print("\n" + "=" * 80)
    print("SANITY CHECK - 5 Sample Outputs for Manual Verification")
    print("=" * 80)

    for i, sample in enumerate(TEST_PROMPTS, 1):
        prompt = sample["instruction"]
        category = sample["category"]

        messages = [
            {"role": "system", "content": "You are an Infrastructure-as-Code expert. Generate correct, production-ready code."},
            {"role": "user", "content": prompt}
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        print(f"\n{'='*80}")
        print(f"SAMPLE {i}: {category.upper()}")
        print(f"{'='*80}")
        print(f"PROMPT: {prompt}")
        print(f"\n--- GENERATED OUTPUT ---")
        print(response)
        print(f"--- END ---")

        results.append({
            "prompt": prompt,
            "category": category,
            "output": response
        })

    return results


@app.local_entrypoint()
def main():
    """Run sanity check."""
    print("\nGenerating 5 samples for manual verification...\n")
    results = generate_samples.remote()

    print("\n" + "=" * 80)
    print("SANITY CHECK COMPLETE")
    print("=" * 80)
    print("\nManually verify each output above:")
    print("1. Is it valid syntax for the target format?")
    print("2. Does it include the correct required fields?")
    print("3. Would it actually work if deployed?")
    print("\nIf outputs are 'gaming' the reward (e.g., just including keywords")
    print("without proper structure), the validators need tightening.")
