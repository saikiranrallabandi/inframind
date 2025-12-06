torch — PyTorch deep learning framework. Handles tensors, GPU acceleration, and is the backbone for model operations.


AutoModelForCausalLM — Loads any causal language model (GPT-style, left-to-right generation) from Hugging Face. Automatically detects architecture from the model name.



AutoTokenizer — Loads the matching tokenizer for your model. Converts text ↔ token IDs.


TrainingArguments — Configuration container for training: batch size, learning rate, epochs, save steps, logging, evaluation strategy, etc.


BitsAndBytesConfig — Configures quantization (4-bit or 8-bit) to reduce VRAM usage. Essential for QLoRA on consumer GPUs.


LoraConfig — Defines LoRA hyperparameters: rank (r), alpha scaling, dropout, and which layers to adapt (e.g., q_proj, v_proj).

get_peft_model — Wraps your base model with LoRA adapters, making only the adapter weights trainable.
prepare_model_for_kbit_training — Prepares a quantized model for training by handling gradient checkpointing and layer normalization in fp32.

SFTTrainer — Supervised Fine-Tuning trainer from TRL. Simplifies instruction/chat fine-tuning with built-in dataset formatting, packing, and PEFT integration.

Dataset — Hugging Face dataset class. Create datasets from dicts, lists, or files for feeding into the trainer.



