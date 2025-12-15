#!/usr/bin/env python3
"""Test InfraMind GRPO model locally (merged model version)."""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():
    print("Loading InfraMind GRPO merged model from HuggingFace...")

    model_name = "srallabandi0225/inframind-0.5b-grpo"

    # Load merged model directly (no PeftModel needed!)
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    print(f"Model loaded! Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Test prompts
    test_prompts = [
        ("Terraform EC2", "Create Terraform for AWS EC2 instance", "t3.micro instance type"),
        ("Kubernetes Deployment", "Create Kubernetes deployment", "nginx with 3 replicas"),
        ("Dockerfile", "Create a Dockerfile", "Python Flask application"),
        ("GitHub Actions", "Create GitHub Actions workflow", "Run pytest on push"),
    ]

    for name, instruction, input_text in test_prompts:
        print(f"\n{'='*60}")
        print(f"Test: {name}")
        print(f"{'='*60}")

        prompt = f"""### Instruction:
{instruction}
### Input:
{input_text}
### Response:
"""

        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the response part
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()

        print(response)

    # Interactive mode
    print(f"\n{'='*60}")
    print("Interactive Mode - Enter your prompts (type 'quit' to exit)")
    print(f"{'='*60}")

    while True:
        instruction = input("\nInstruction: ").strip()
        if instruction.lower() == 'quit':
            break

        input_text = input("Input (optional): ").strip()

        prompt = f"""### Instruction:
{instruction}
### Input:
{input_text}
### Response:
"""

        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()

        print(f"\nGenerated:\n{response}")

if __name__ == "__main__":
    main()
