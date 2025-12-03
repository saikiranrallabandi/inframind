"""IAPO Quick Start Example"""
from iapo import IaCBench, IAPOTrainer, create_dataset

# 1. Load IaC-Bench dataset
dataset = create_dataset(size=50)  # Start small
print(f"Dataset: {len(dataset)} tasks")

# Show sample task
task = dataset[0]
print(f"\nSample task:")
print(f"  Instruction: {task['instruction']}")
print(f"  Input: {task['input']}")
print(f"  Category: {task['category']}")

# 2. Initialize trainer with Qwen
trainer = IAPOTrainer(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    lr=1e-5,
    group_size=2
)

# 3. Train (1 epoch for demo)
print("\nTraining...")
history = trainer.train(dataset, epochs=1)

# 4. Save model
trainer.save("./iapo-quickstart-model")
print("\nModel saved!")
