"""
InfraMind Modal App - SFT Training on A100.

Self-contained training script for Modal.

Usage:
    modal run modal_app.py
"""

import modal

app = modal.App("inframind-sft")

# Persistent volume for trained models (survives deployments)
model_volume = modal.Volume.from_name("inframind-models", create_if_missing=True)
MODEL_DIR = "/models"

# Build image with all dependencies + training data
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "transformers==4.46.0",
        "accelerate>=0.34.0",
        "bitsandbytes>=0.43.0",
        "peft==0.14.0",
        "trl==0.12.2",
        "datasets",
        "fastapi",
    )
    .add_local_file("data/real_code_2k.json", "/root/training_data.json")
)


# =============================================================================
# TEMPLATES - Import from data module (kept inline for Modal compatibility)
# =============================================================================
# Note: Templates are defined in inframind/data/templates.py for local use.
# They're duplicated here because Modal needs self-contained functions.

OUT_OF_DOMAIN_RESPONSE = "I'm a DevOps and Infrastructure assistant. I can help with Terraform, Kubernetes, Docker, CI/CD, cloud infrastructure, monitoring, and debugging. For other topics, please use a general-purpose assistant."

# Embedded high-quality training data (fallback)
EMBEDDED_TRAINING_DATA = [
    {
        "instruction": "Create Kubernetes Service",
        "input": "LoadBalancer type for nginx on port 80",
        "output": """apiVersion: v1
kind: Service
metadata:
  name: nginx-loadbalancer
  labels:
    app: nginx
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 80
    protocol: TCP
    name: http
  selector:
    app: nginx""",
        "category": "kubernetes"
    },
    {
        "instruction": "Create Terraform for S3 bucket",
        "input": "with versioning enabled",
        "output": """terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

resource "aws_s3_bucket" "main" {
  bucket = var.bucket_name
  tags = {
    Name        = var.bucket_name
    Environment = var.environment
  }
}

resource "aws_s3_bucket_versioning" "main" {
  bucket = aws_s3_bucket.main.id
  versioning_configuration {
    status = "Enabled"
  }
}

variable "bucket_name" {
  type = string
}

variable "environment" {
  type    = string
  default = "dev"
}""",
        "category": "terraform"
    },
    {
        "instruction": "Create Dockerfile for Python application",
        "input": "with FastAPI and uvicorn",
        "output": """FROM python:3.11-slim

RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]""",
        "category": "docker"
    },
    {
        "instruction": "Create GitHub Actions workflow",
        "input": "run tests on push to main",
        "output": """name: Run Tests

on:
  push:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: '3.11'
    - run: pip install -r requirements.txt
    - run: pytest tests/ -v""",
        "category": "cicd"
    },
    {
        "instruction": "Create Kubernetes Deployment",
        "input": "nginx with 3 replicas",
        "output": """apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx
  labels:
    app: nginx
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.25-alpine
        ports:
        - containerPort: 80
        resources:
          requests:
            memory: "64Mi"
            cpu: "100m"
          limits:
            memory: "128Mi"
            cpu: "200m"""",
        "category": "kubernetes"
    },
    {
        "instruction": "Create docker-compose.yml",
        "input": "for web app with PostgreSQL and Redis",
        "output": """version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/app
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: app
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine

volumes:
  postgres_data:""",
        "category": "docker"
    },
]


def load_training_data_from_json():
    """Load training data from JSON file or fallback to embedded data."""
    import json
    import os

    paths_to_try = [
        "/root/training_data.json",
        "data/real_code_2k.json",
        "data/training_data.json",
    ]

    for path in paths_to_try:
        if os.path.exists(path):
            print(f"  Loading training data from: {path}")
            with open(path) as f:
                data = json.load(f)
            print(f"  Loaded {len(data)} samples")
            return data

    print("  Using embedded training data")
    return EMBEDDED_TRAINING_DATA


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

@app.function(gpu="A100", image=image, timeout=3600, volumes={MODEL_DIR: model_volume})
def train_sft():
    """SFT training with QLoRA."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    print("=" * 60)
    print("InfraMind SFT Training")
    print("=" * 60)

    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    OUTPUT_DIR = "/models/inframind-sft"

    # QLoRA config
    print("\n[1/4] Creating QLoRA config...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    print(f"\n[2/4] Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # LoRA config
    print("\n[3/4] Configuring LoRA...")
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # Load dataset
    print("\n[4/4] Loading dataset...")
    dataset = load_training_data_from_json()

    def format_sample(sample):
        messages = [
            {"role": "system", "content": "You are an Infrastructure-as-Code expert. Generate correct, production-ready code."},
            {"role": "user", "content": f"{sample['instruction']}\n\n{sample['input']}".strip()},
            {"role": "assistant", "content": sample['output']}
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    formatted = [format_sample(s) for s in dataset]
    hf_dataset = Dataset.from_list(formatted)
    split_data = hf_dataset.train_test_split(test_size=0.1, seed=42)

    # SFT config
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        packing=True,
        dataset_text_field="text",
        max_seq_length=2048,
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        report_to="none",
    )

    # Train
    print("\nStarting training...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=split_data["train"],
        eval_dataset=split_data["test"],
        args=sft_config,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    # Evaluate
    eval_result = trainer.evaluate()
    print(f"\nFinal eval loss: {eval_result['eval_loss']:.4f}")

    # Save
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    model_volume.commit()

    print(f"\nModel saved to: {OUTPUT_DIR}/final")
    return {"status": "success", "eval_loss": eval_result['eval_loss']}


@app.function(gpu="A100", image=image, timeout=3600, volumes={MODEL_DIR: model_volume})
def train_and_test(test_prompts: list[str] = None):
    """Train and run inference tests."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, PeftModel
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    OUTPUT_DIR = "/models/inframind-sft"

    # Train (same as train_sft)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    peft_config = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    dataset = load_training_data_from_json()

    def format_sample(sample):
        messages = [
            {"role": "system", "content": "You are an Infrastructure-as-Code expert. Generate correct, production-ready code."},
            {"role": "user", "content": f"{sample['instruction']}\n\n{sample['input']}".strip()},
            {"role": "assistant", "content": sample['output']}
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    formatted = [format_sample(s) for s in dataset]
    hf_dataset = Dataset.from_list(formatted)
    split_data = hf_dataset.train_test_split(test_size=0.1, seed=42)

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        packing=True,
        dataset_text_field="text",
        max_seq_length=2048,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=split_data["train"],
        eval_dataset=split_data["test"],
        args=sft_config,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    train_result = trainer.train()
    eval_result = trainer.evaluate()
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    model_volume.commit()

    # Test inference
    del model
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, f"{OUTPUT_DIR}/final")
    model.eval()

    if test_prompts is None:
        test_prompts = [
            "Create Terraform for S3 bucket",
            "Create Kubernetes Deployment for nginx with 3 replicas",
            "Create Dockerfile for Python FastAPI app",
        ]

    results = []
    for prompt in test_prompts:
        messages = [
            {"role": "system", "content": "You are an Infrastructure-as-Code expert. Generate correct, production-ready code."},
            {"role": "user", "content": prompt}
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        results.append({"prompt": prompt, "response": response})

    return {
        "train_loss": train_result.metrics.get("train_loss"),
        "eval_loss": eval_result['eval_loss'],
        "test_results": results
    }


# =============================================================================
# INFERENCE SERVER
# =============================================================================

@app.cls(
    gpu="A100",
    image=image,
    timeout=300,
    scaledown_window=300,
    volumes={MODEL_DIR: model_volume},
)
class InfraMindModel:
    """Persistent model class for fast inference."""

    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import os

        MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
        ADAPTER_PATH = "/models/inframind-sft/final"

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )

        if os.path.exists(ADAPTER_PATH):
            self.model = PeftModel.from_pretrained(self.model, ADAPTER_PATH)
            self.is_finetuned = True
        else:
            self.is_finetuned = False

        self.model.eval()
        print(f"Model ready! (finetuned={self.is_finetuned})")

    @modal.fastapi_endpoint(method="POST")
    def generate(self, request: dict):
        """Web API endpoint for inference."""
        import torch

        prompt = request.get("prompt", "")
        max_tokens = request.get("max_tokens", 512)
        temperature = request.get("temperature", 0.7)

        messages = [
            {"role": "system", "content": "You are an Infrastructure-as-Code expert. Generate correct, production-ready code."},
            {"role": "user", "content": prompt}
        ]

        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                repetition_penalty=1.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        return {
            "prompt": prompt,
            "response": response,
            "model": "qwen2.5-0.5b-inframind" if self.is_finetuned else "qwen2.5-0.5b-base",
        }


@app.local_entrypoint()
def main():
    """Run training and testing."""
    print("\nStarting InfraMind training on Modal (A100)...\n")
    result = train_and_test.remote()

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Train Loss: {result['train_loss']:.4f}")
    print(f"Eval Loss: {result['eval_loss']:.4f}")

    for i, test in enumerate(result['test_results'], 1):
        print(f"\n[Test {i}] {test['prompt']}")
        print(f"{test['response'][:200]}...")
