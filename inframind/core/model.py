"""
Model loading and configuration.

Step 1: Load base model for fine-tuning.
"""

from typing import Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

from inframind.utils.logging import get_logger

logger = get_logger(__name__)


def load_model(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    quantization_config=None,
    device_map: str = "auto",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load base model and tokenizer.

    Args:
        model_name: HuggingFace model name
        quantization_config: Optional BitsAndBytesConfig for QLoRA
        device_map: Device placement strategy

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model_kwargs = {
        "device_map": device_map,
        "trust_remote_code": True,
    }

    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
        logger.info("Using quantization config (QLoRA)")
    else:
        import torch
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {total_params/1e6:.1f}M params")

    return model, tokenizer
