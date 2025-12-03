"""IAPO Trainer - GRPO with Infrastructure-Aware Optimization"""
import torch
import numpy as np
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional
from .dataset import IaCBench
from .rewards import IaCReward


class IAPOTrainer:
    """IAPO: Infrastructure-Aware Policy Optimization Trainer"""

    def __init__(self, model_name: str = "HuggingFaceTB/SmolLM-135M-Instruct",
                 lr: float = 1e-5, group_size: int = 2, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.reward_fn = IaCReward()
        self.group_size = group_size
        for p in self.model.parameters():
            p.requires_grad = True

    def _format_prompt(self, task: Dict) -> str:
        """Format task as Alpaca-style prompt"""
        if task.get("input"):
            return f"### Instruction:\n{task['instruction']}\n\n### Input:\n{task['input']}\n\n### Response:\n"
        return f"### Instruction:\n{task['instruction']}\n\n### Response:\n"

    def _generate(self, prompt: str, num_samples: int = 2) -> List[str]:
        """Generate multiple responses for GRPO"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=256, do_sample=True,
                                          temperature=0.8, num_return_sequences=num_samples,
                                          pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def train_step(self, task: Dict) -> Dict:
        """Single IAPO training step with GRPO"""
        prompt = self._format_prompt(task)
        responses = self._generate(prompt, self.group_size)

        # Score responses (IAPO reward)
        rewards = np.array([self.reward_fn.score(r, task["category"])[0] for r in responses])

        # GRPO advantage: normalize within group
        mean_r, std_r = rewards.mean(), rewards.std() + 1e-8
        advantages = (rewards - mean_r) / std_r

        # Policy update
        self.model.train()
        total_loss = 0.0
        for i, response in enumerate(responses):
            inputs = self.tokenizer(response, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = -outputs.loss * advantages[i]  # GRPO loss
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()

        return {"loss": total_loss / len(responses), "mean_reward": rewards.mean(), "best_reward": rewards.max()}

    def train(self, dataset: IaCBench, epochs: int = 1) -> List[Dict]:
        """Train on IaC-Bench dataset"""
        history = []
        for epoch in range(epochs):
            epoch_rewards = []
            for task in dataset:
                metrics = self.train_step(task)
                epoch_rewards.append(metrics["mean_reward"])
                print(f"Task: {task['category'][:4]} | Reward: {metrics['mean_reward']:.3f}")
            history.append({"epoch": epoch + 1, "mean_reward": np.mean(epoch_rewards)})
            print(f"Epoch {epoch+1}: Mean Reward = {np.mean(epoch_rewards):.3f}")
        return history

    def save(self, path: str):
        """Save trained model"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
