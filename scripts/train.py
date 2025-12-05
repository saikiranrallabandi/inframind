#!/usr/bin/env python3
"""InfraMind Training Script"""
import argparse
from inframind import IaCBench, InfraMindTrainer, create_dataset


def main():
    parser = argparse.ArgumentParser(description="Fine-tune model with InfraMind")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--category", type=str, help="Filter by category")
    parser.add_argument("--size", type=int, help="Dataset size")
    parser.add_argument("--output", default="./inframind-model", help="Output path")
    args = parser.parse_args()

    # Load dataset
    categories = [args.category] if args.category else None
    dataset = create_dataset(categories=categories, size=args.size)
    print(f"Loaded {len(dataset)} tasks")

    # Train
    trainer = InfraMindTrainer(model_name=args.model, lr=args.lr)
    trainer.train(dataset, epochs=args.epochs)
    trainer.save(args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
