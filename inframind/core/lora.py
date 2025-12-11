"""
LoRA adapter configuration.

Step 3: Configure Low-Rank Adaptation for parameter-efficient training.
"""

from peft import LoraConfig

from inframind.utils.logging import get_logger

logger = get_logger(__name__)


def get_lora_config(
    r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
) -> LoraConfig:
    """
    Get LoRA configuration for parameter-efficient fine-tuning.

    Args:
        r: LoRA rank (higher = more capacity)
        lora_alpha: Scaling factor (typically 2x rank)
        lora_dropout: Dropout for regularization

    Returns:
        LoraConfig for PEFT
    """
    logger.info(f"Creating LoRA config (r={r}, alpha={lora_alpha})")

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj"       # FFN
        ],
    )

    return config
