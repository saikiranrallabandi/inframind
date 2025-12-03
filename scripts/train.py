#!/usr/bin/env python3
"""IAPO Training Script"""
import argparse
from iapo import IaCBench, IAPOTrainer, create_dataset


def main():
    parser = argparse.ArgumentParser(description="Train IAPO model")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--category", type=str, help="Filter by category")
    parser.add_argument("--size", type=int, help="Dataset size")
    parser.add_argument("--output", default="./iapo-model", help="Output path")
    args = parser.parse_args()

    # Load dataset
    categories = [args.category] if args.category else None
    dataset = create_dataset(categories=categories, size=args.size)
    print(f"Loaded {len(dataset)} tasks")

    # Train
    trainer = IAPOTrainer(model_name=args.model, lr=args.lr)
    trainer.train(dataset, epochs=args.epochs)
    trainer.save(args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
