# Step 1: Load Base Model

## Objective
Load the base Qwen2.5-0.5B-Instruct model and test its IaC generation capabilities BEFORE fine-tuning.

## Why This Step?
- Establish a **baseline** to compare against fine-tuned model
- Verify GPU setup and model loading works
- Understand the model's current limitations

## Code: `modal_step1.py`

```python
import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "transformers==4.44.2",
    "accelerate",
)

app = modal.App("inframind-step1", image=image)

@app.function(gpu="A100", timeout=300)
def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Test generation
    prompt = "### Instruction:\nCreate Terraform for AWS S3 bucket with versioning\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Run Command
```bash
modal run modal_step1.py
```

## Results

### Model Info
- **Model**: Qwen/Qwen2.5-0.5B-Instruct
- **Parameters**: 494,032,768 (~0.5B)
- **Device**: cuda:0 (A100 GPU)

### Generated Output (Before Fine-tuning)
```hcl
resource "aws_s3_bucket" "my_bucket" {
  name                = "MyBucket"
  location            = "us-west-2"
  versioning          = aws_versioning::versioning(
    status = "Enabled"
  )
}

resource "aws_s3_bucket_versioning_policy" "public_access" {
  bucket           = aws_s3_bucket.my_bucket.name
  policy_type      = "PublicReadAccess"
}
```

### Issues Identified
| Issue | Problem |
|-------|---------|
| `name` attribute | Should be `bucket` in Terraform |
| `versioning` syntax | `aws_versioning::versioning()` is invalid |
| `location` | Should be handled via provider, not bucket |
| `aws_s3_bucket_versioning_policy` | This resource doesn't exist |

## Conclusion
The base model **attempts** Terraform but has significant syntax errors. Fine-tuning with InfraMind dataset should fix these issues.

## Next Step
[Step 2: QLoRA Quantization + Dataset](./step2_qlora_dataset.md)
