"""InfraMind Quick Start Example"""
from inframind import IaCBench, InfraMindTrainer, create_dataset

# 1. Load InfraMind-Bench dataset
dataset = create_dataset(size=50)  # Start small
print(f"Dataset: {len(dataset)} tasks")

# Show sample task
task = dataset[0]
print(f"\nSample task:")
print(f"  Instruction: {task['instruction']}")
print(f"  Input: {task['input']}")
print(f"  Category: {task['category']}")

# 2. Initialize trainer with Qwen
trainer = InfraMindTrainer(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    lr=1e-5,
    group_size=2
)

# 3. Train (1 epoch for demo)
print("\nFine-tuning...")
history = trainer.train(dataset, epochs=1)

# 4. Save fine-tuned model
trainer.save("./qwen-0.5b-inframind")
print("\nModel saved!")
