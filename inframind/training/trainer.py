"""
SFT Trainer for InfraMind.

Step 4: Supervised Fine-Tuning with TRL.
"""

from typing import Optional
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

from inframind.utils.logging import get_logger

logger = get_logger(__name__)


def get_sft_config(
    output_dir: str = "./inframind-sft",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
) -> SFTConfig:
    """
    Get SFT training configuration.

    Args:
        output_dir: Where to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Per-device batch size
        learning_rate: Learning rate for optimizer

    Returns:
        SFTConfig for TRL trainer
    """
    logger.info(f"Creating SFT config (epochs={num_epochs}, lr={learning_rate})")

    config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        packing=True,
        dataset_text_field="text",
        max_seq_length=1024,
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    return config


def create_trainer(
    model,
    tokenizer,
    dataset: Dataset,
    peft_config,
    sft_config: Optional[SFTConfig] = None,
) -> SFTTrainer:
    """
    Create SFT trainer.

    Args:
        model: Base model (with quantization)
        tokenizer: Tokenizer
        dataset: Training dataset
        peft_config: LoRA configuration
        sft_config: Optional SFT configuration

    Returns:
        SFTTrainer ready for training
    """
    if sft_config is None:
        sft_config = get_sft_config()

    logger.info("Creating SFT trainer")

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    return trainer
