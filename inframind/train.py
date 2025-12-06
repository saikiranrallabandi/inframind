"""InfraMind Trainer - SFT/LoRA and GRPO fine-tuning for Infrastructure-as-Code"""
import torch
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer
from datasets import Dataset

from .dataset import IaCBench, create_dataset
from .rewards import IaCReward


class InfraMindTrainer:
    """InfraMind: Fine-tuning trainer for IaC generation

    Supports:
    - SFT (Supervised Fine-Tuning) with LoRA/QLoRA
    - GRPO (Group Relative Policy Optimization)

    Example:
        trainer = InfraMindTrainer(
            model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
            use_qlora=True,
        )
        trainer.train_sft(dataset, epochs=3)
        trainer.save("./qwen-7b-inframind")
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        use_qlora: bool = True,
        lora_rank: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        lr: float = 2e-4,
        device: str = None,
    ):
        self.model_name = model_name
        self.use_qlora = use_qlora
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # Quantization config for QLoRA
        if use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

        # LoRA config
        self.lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()

        # For GRPO
        self.reward_fn = IaCReward()

    def _format_prompt(self, task: Dict) -> str:
        """Format task as chat-style prompt"""
        instruction = task.get("instruction", "")
        input_text = task.get("input", "")

        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

        return prompt

    def _format_for_sft(self, task: Dict) -> str:
        """Format task for SFT training (prompt + response)"""
        prompt = self._format_prompt(task)
        output = task.get("output", "")

        if output:
            return prompt + output
        else:
            # For DevOps tasks without output, we'll generate during training
            return prompt

    def _prepare_dataset(self, dataset: IaCBench) -> Dataset:
        """Convert IaCBench to HuggingFace Dataset"""
        data = []
        for task in dataset:
            text = self._format_for_sft(task)
            data.append({"text": text, "category": task.get("category", "unknown")})

        return Dataset.from_list(data)

    def train_sft(
        self,
        dataset: IaCBench,
        epochs: int = 3,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        max_seq_length: int = 1024,
        output_dir: str = "./inframind-checkpoints",
        logging_steps: int = 10,
        save_steps: int = 100,
    ) -> None:
        """Train using Supervised Fine-Tuning (SFT) with LoRA

        Args:
            dataset: IaCBench dataset
            epochs: Number of training epochs
            batch_size: Per-device batch size
            gradient_accumulation_steps: Gradient accumulation steps
            max_seq_length: Maximum sequence length
            output_dir: Directory for checkpoints
            logging_steps: Log every N steps
            save_steps: Save checkpoint every N steps
        """
        print(f"Preparing dataset with {len(dataset)} examples...")
        hf_dataset = self._prepare_dataset(dataset)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=self.lr,
            weight_decay=0.01,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=3,
            bf16=True,
            optim="paged_adamw_8bit" if self.use_qlora else "adamw_torch",
            gradient_checkpointing=True,
            report_to="none",  # Set to "wandb" if you want W&B logging
        )

        # SFT Trainer
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=hf_dataset,
            args=training_args,
            processing_class=self.tokenizer,
            max_seq_length=max_seq_length,
        )

        print(f"Starting SFT training for {epochs} epochs...")
        trainer.train()
        print("Training complete!")

    def train_grpo(
        self,
        dataset: IaCBench,
        epochs: int = 1,
        group_size: int = 4,
    ) -> List[Dict]:
        """Train using Group Relative Policy Optimization (GRPO)

        This is a lightweight GRPO implementation for RL fine-tuning.

        Args:
            dataset: IaCBench dataset
            epochs: Number of training epochs
            group_size: Number of samples per prompt for comparison

        Returns:
            Training history
        """
        from torch.optim import AdamW

        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        history = []

        for epoch in range(epochs):
            epoch_rewards = []

            for task in dataset:
                prompt = self._format_prompt(task)

                # Generate multiple responses
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.8,
                        num_return_sequences=group_size,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

                # Score responses
                category = task.get("category", "terraform")
                rewards = np.array([self.reward_fn.score(r, category)[0] for r in responses])

                # GRPO advantage
                mean_r, std_r = rewards.mean(), rewards.std() + 1e-8
                advantages = (rewards - mean_r) / std_r

                # Policy update
                self.model.train()
                total_loss = 0.0

                for i, response in enumerate(responses):
                    resp_inputs = self.tokenizer(response, return_tensors="pt", truncation=True, max_length=512)
                    resp_inputs = {k: v.to(self.model.device) for k, v in resp_inputs.items()}

                    outputs = self.model(**resp_inputs, labels=resp_inputs["input_ids"])
                    loss = -outputs.loss * advantages[i]

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item()

                epoch_rewards.append(rewards.mean())
                print(f"Task: {category[:8]} | Reward: {rewards.mean():.3f}")

            history.append({"epoch": epoch + 1, "mean_reward": np.mean(epoch_rewards)})
            print(f"Epoch {epoch+1}: Mean Reward = {np.mean(epoch_rewards):.3f}")

        return history

    def save(self, path: str) -> None:
        """Save the fine-tuned LoRA adapter

        Args:
            path: Directory to save the model
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save LoRA adapter
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        # Save config
        config = {
            "base_model": self.model_name,
            "use_qlora": self.use_qlora,
            "lora_config": {
                "r": self.lora_config.r,
                "lora_alpha": self.lora_config.lora_alpha,
                "lora_dropout": self.lora_config.lora_dropout,
            }
        }

        import json
        with open(path / "inframind_config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, device: str = None) -> "InfraMindTrainer":
        """Load a saved InfraMind model

        Args:
            path: Path to saved model
            device: Device to load on

        Returns:
            Loaded trainer
        """
        import json

        path = Path(path)

        with open(path / "inframind_config.json", "r") as f:
            config = json.load(f)

        # Create trainer with base model
        trainer = cls(
            model_name=config["base_model"],
            use_qlora=config["use_qlora"],
            device=device,
        )

        # Load LoRA weights
        trainer.model = PeftModel.from_pretrained(
            trainer.model,
            path,
            is_trainable=False,
        )

        return trainer

    def generate(self, instruction: str, input_text: str = "", max_tokens: int = 512) -> str:
        """Generate IaC code for a given instruction

        Args:
            instruction: The task instruction
            input_text: Optional additional input
            max_tokens: Maximum tokens to generate

        Returns:
            Generated IaC code
        """
        task = {"instruction": instruction, "input": input_text}
        prompt = self._format_prompt(task)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the response part
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()

        return response
