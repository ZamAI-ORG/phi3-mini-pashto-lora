#!/usr/bin/env python
"""LoRA fine-tuning script for ZamAI Phi-3 Pashto model."""

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional

import torch
import yaml
from datasets import Dataset, load_dataset
from huggingface_hub import login
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def prepare_prompt(item: Dict[str, Any], template: str) -> str:
    """Prepare prompt from template and data item."""
    # Simple template substitution
    prompt = template
    for key, value in item.items():
        placeholder = "{" + key + "}"
        if placeholder in prompt:
            prompt = prompt.replace(placeholder, str(value))
    return prompt


def preprocess_function(examples: Dict[str, List], tokenizer, max_seq_length: int, template: str):
    """Preprocess examples for training."""
    inputs = []
    targets = []
    
    for i in range(len(examples["instruction"])):
        item = {k: v[i] for k, v in examples.items()}
        
        # Create full prompt with instruction and expected output
        full_text = prepare_prompt(item, template)
        
        inputs.append(full_text)
        targets.append(full_text)  # For causal LM, input and target are the same
    
    # Tokenize
    model_inputs = tokenizer(
        inputs,
        max_length=max_seq_length,
        truncation=True,
        padding=False,
        return_tensors=None,
    )
    
    # For causal LM, labels are the same as input_ids
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    
    return model_inputs


def create_dataset(data_path: str, tokenizer, max_seq_length: int, template: str) -> Optional[Dataset]:
    """Create dataset from data file."""
    if not data_path or not os.path.exists(data_path):
        logger.warning(f"Data file not found: {data_path}")
        return None
    
    logger.info(f"Loading data from: {data_path}")
    
    if data_path.endswith(".jsonl"):
        data = load_jsonl(data_path)
        dataset = Dataset.from_list(data)
    else:
        # Try to load as HF dataset
        dataset = load_dataset("json", data_files=data_path, split="train")
    
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Preprocess
    dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, max_seq_length, template),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Preprocessing data",
    )
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for ZamAI Phi-3 Pashto")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--output_dir", help="Output directory (overrides config)")
    parser.add_argument("--push_to_hub", action="store_true", help="Push to Hugging Face Hub")
    parser.add_argument("--hub_model_id", help="Hub model ID for upload")
    parser.add_argument("--hf_token", help="Hugging Face token")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded config from: {args.config}")
    
    # Override config with command line args if provided
    if args.output_dir:
        config["output_dir"] = args.output_dir
    
    # Set random seed
    set_seed(config.get("seed", 42))
    
    # Login to HF Hub if token provided
    if args.hf_token:
        login(token=args.hf_token)
    
    # Load tokenizer and model
    logger.info(f"Loading model: {config['base_model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model_name"], trust_remote_code=True)
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with quantization if specified
    model_kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
    
    if config.get("use_4bit", False):
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
    
    model = AutoModelForCausalLM.from_pretrained(config["base_model_name"], **model_kwargs)
    
    # Prepare LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.get("lora_r", 64),
        lora_alpha=config.get("lora_alpha", 16),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=config.get("lora_target_modules", ["q_proj", "v_proj"]),
        bias="none",
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load datasets
    template = config.get("prompt_template", "{instruction}\n{input}\n{output}")
    max_seq_length = config.get("max_seq_length", 2048)
    
    train_dataset = create_dataset(config.get("train_file"), tokenizer, max_seq_length, template)
    eval_dataset = create_dataset(config.get("eval_file"), tokenizer, max_seq_length, template)
    
    if train_dataset is None:
        raise ValueError("Training dataset is required but could not be loaded")
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt",
    )
    
    # Training arguments
    output_dir = config.get("output_dir", "outputs/zamai-phi3-pashto")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 16),
        learning_rate=config.get("learning_rate", 2e-4),
        num_train_epochs=config.get("num_train_epochs", 3),
        warmup_ratio=config.get("warmup_ratio", 0.03),
        weight_decay=config.get("weight_decay", 0.0),
        logging_steps=config.get("logging_steps", 25),
        eval_steps=config.get("eval_steps", 400) if eval_dataset else None,
        evaluation_strategy="steps" if eval_dataset else "no",
        save_steps=config.get("save_steps", 400),
        save_total_limit=config.get("save_total_limit", 3),
        max_grad_norm=config.get("max_grad_norm", 1.0),
        fp16=config.get("fp16", False),
        bf16=config.get("bf16", True),
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        dataloader_drop_last=True,
        report_to=config.get("report_to", []),
        run_name="zamai-phi3-pashto-lora",
        seed=config.get("seed", 42),
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    logger.info(f"Saving final model to: {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Push to hub if requested
    if args.push_to_hub:
        if not args.hub_model_id:
            raise ValueError("--hub_model_id required when --push_to_hub is set")
        
        logger.info(f"Pushing to Hub: {args.hub_model_id}")
        trainer.model.push_to_hub(args.hub_model_id)
        tokenizer.push_to_hub(args.hub_model_id)
    
    logger.info("✅ Training completed successfully!")


if __name__ == "__main__":
    main()