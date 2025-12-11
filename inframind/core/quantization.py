"""
QLoRA quantization configuration.

Step 2: Apply 4-bit quantization for memory efficiency.
"""

import torch
from transformers import BitsAndBytesConfig

from inframind.utils.logging import get_logger

logger = get_logger(__name__)


def get_qlora_config() -> BitsAndBytesConfig:
    """
    Get QLoRA configuration for 4-bit quantization.

    Returns:
        BitsAndBytesConfig for QLoRA
    """
    logger.info("Creating QLoRA config (4-bit NF4)")

    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    return config
