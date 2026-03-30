"""
LoRA fine-tuning with layer-specific targeting.

Usage:
    python training/train_lora.py \
        --model_name meta-llama/Llama-3.1-8B \
        --dataset_path data/train.jsonl \
        --target_layers 0,1,2,3 \
        --output_dir outputs/lora-early-layers

    # Target only the last 8 layers:
    python training/train_lora.py \
        --model_name meta-llama/Llama-3.1-8B \
        --target_layers 24,25,26,27,28,29,30,31

    # Target all layers (default behavior):
    python training/train_lora.py \
        --model_name meta-llama/Llama-3.1-8B

    # Target a range of layers with --layer_range:
    python training/train_lora.py \
        --model_name meta-llama/Llama-3.1-8B \
        --layer_range 0-15
"""

import argparse
import json
import os
import sys

import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)


def parse_layers(target_layers: str | None, layer_range: str | None, num_layers: int) -> list[int] | None:
    """Parse layer specification into a list of layer indices. Returns None for all layers."""
    if target_layers and layer_range:
        print("Error: specify --target_layers or --layer_range, not both.")
        sys.exit(1)

    if target_layers:
        layers = [int(x.strip()) for x in target_layers.split(",")]
        for l in layers:
            if l < 0 or l >= num_layers:
                print(f"Error: layer {l} out of range [0, {num_layers - 1}]")
                sys.exit(1)
        return layers

    if layer_range:
        parts = layer_range.split("-")
        start, end = int(parts[0]), int(parts[1])
        if start < 0 or end >= num_layers or start > end:
            print(f"Error: invalid range {start}-{end} for {num_layers} layers")
            sys.exit(1)
        return list(range(start, end + 1))

    return None  # all layers


def build_target_modules(model, layers: list[int] | None) -> list[str]:
    """Build the list of LoRA target module names for specific layers.

    If layers is None, returns the standard module names (applied to all layers).
    If layers is specified, returns fully qualified names like
    'model.layers.0.self_attn.q_proj' to target only those layers.
    """
    standard_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    if layers is None:
        return standard_targets

    # Build explicit module paths for the requested layers
    targets = []
    for name, _ in model.named_modules():
        for target in standard_targets:
            if name.endswith(target):
                # Extract layer index from name like "model.layers.5.self_attn.q_proj"
                parts = name.split(".")
                try:
                    layer_idx_pos = parts.index("layers") + 1
                    layer_idx = int(parts[layer_idx_pos])
                    if layer_idx in layers:
                        targets.append(name)
                except (ValueError, IndexError):
                    continue

    if not targets:
        print("Warning: no matching modules found for the specified layers.")
        print("The model may use a different naming convention.")
        print("Named modules in the model:")
        for name, _ in model.named_modules():
            if any(t in name for t in standard_targets):
                print(f"  {name}")
        sys.exit(1)

    return targets


def load_training_data(dataset_path: str, tokenizer, max_length: int) -> Dataset:
    """Load and tokenize training data.

    Supports:
      - HuggingFace dataset name (e.g. 'wikitext/wikitext-2-raw-v1')
      - JSONL file with 'text' field
      - Text file (one document per line)
    """
    if os.path.exists(dataset_path):
        if dataset_path.endswith(".jsonl") or dataset_path.endswith(".json"):
            dataset = load_dataset("json", data_files=dataset_path, split="train")
        elif dataset_path.endswith(".txt"):
            dataset = load_dataset("text", data_files=dataset_path, split="train")
        else:
            dataset = load_dataset("json", data_files=dataset_path, split="train")
    else:
        # Treat as HuggingFace dataset name
        parts = dataset_path.split("/")
        if len(parts) >= 2:
            dataset = load_dataset(parts[0], "/".join(parts[1:]), split="train")
        else:
            dataset = load_dataset(dataset_path, split="train")

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    return tokenized


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning with layer targeting")

    # Model
    parser.add_argument("--model_name", type=str, required=True, help="HuggingFace model name or path")
    parser.add_argument("--tokenizer_name", type=str, default=None, help="Tokenizer name (defaults to model_name)")

    # Data
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to training data or HF dataset name")
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length")

    # Layer targeting
    parser.add_argument("--target_layers", type=str, default=None, help="Comma-separated layer indices (e.g. '0,1,2,3')")
    parser.add_argument("--layer_range", type=str, default=None, help="Layer range (e.g. '0-15')")

    # LoRA config
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    # Training
    parser.add_argument("--output_dir", type=str, default="outputs/lora", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit quantization")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit quantization")

    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.model_name}")
    model_kwargs = {"torch_dtype": torch.bfloat16 if args.bf16 else torch.float32}

    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif args.load_in_8bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name or args.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine number of layers
    num_layers = model.config.num_hidden_layers
    print(f"Model has {num_layers} layers")

    # Parse layer targets
    layers = parse_layers(args.target_layers, args.layer_range, num_layers)
    if layers is not None:
        print(f"Targeting layers: {layers}")
    else:
        print("Targeting all layers")

    # Build target modules
    target_modules = build_target_modules(model, layers)
    if layers is not None:
        print(f"LoRA applied to {len(target_modules)} modules")
    else:
        print(f"LoRA target module types: {target_modules}")

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load data
    print(f"Loading dataset: {args.dataset_path}")
    dataset = load_training_data(args.dataset_path, tokenizer, args.max_length)
    print(f"Training examples: {len(dataset)}")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Save config for reproducibility
    os.makedirs(args.output_dir, exist_ok=True)
    config_record = {
        "model_name": args.model_name,
        "target_layers": layers,
        "num_model_layers": num_layers,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "num_target_modules": len(target_modules),
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }
    with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
        json.dump(config_record, f, indent=2)

    # Train
    print("Starting training...")
    trainer.train()

    # Save
    print(f"Saving adapter to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
