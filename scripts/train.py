#!/usr/bin/env python3
"""InfraMind Training Script - SFT/LoRA and GRPO fine-tuning for IaC"""
import argparse
from inframind import InfraMindTrainer, create_dataset


def main():
    parser = argparse.ArgumentParser(description="Fine-tune model with InfraMind")

    # Model settings
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct",
                        help="Base model (default: Qwen2.5-Coder-7B-Instruct)")
    parser.add_argument("--no-qlora", action="store_true",
                        help="Disable QLoRA (4-bit quantization)")

    # LoRA settings
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")

    # Training settings
    parser.add_argument("--method", choices=["sft", "grpo"], default="sft",
                        help="Training method: sft or grpo (default: sft)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--gradient-accumulation", type=int, default=4,
                        help="Gradient accumulation steps")

    # Dataset settings
    parser.add_argument("--category", type=str, help="Filter by category")
    parser.add_argument("--size", type=int, help="Dataset size")

    # Output settings
    parser.add_argument("--output", default="./inframind-model", help="Output path")
    parser.add_argument("--logging-steps", type=int, default=10, help="Log every N steps")
    parser.add_argument("--save-steps", type=int, default=100, help="Save checkpoint every N steps")

    args = parser.parse_args()

    # Load dataset
    categories = [args.category] if args.category else None
    dataset = create_dataset(categories=categories, size=args.size)
    print(f"Loaded {len(dataset)} tasks")

    # Initialize trainer
    trainer = InfraMindTrainer(
        model_name=args.model,
        use_qlora=not args.no_qlora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lr=args.lr,
    )

    # Train
    if args.method == "sft":
        print(f"Starting SFT training with {'QLoRA' if not args.no_qlora else 'LoRA'}...")
        trainer.train_sft(
            dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            max_seq_length=args.max_seq_length,
            output_dir=args.output,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
        )
    else:
        print("Starting GRPO training...")
        history = trainer.train_grpo(dataset, epochs=args.epochs)
        print(f"Final mean reward: {history[-1]['mean_reward']:.3f}")

    # Save final model
    trainer.save(args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
