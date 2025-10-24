#!/usr/bin/env python3
"""
TRAINING SCRIPT - FINAL WORKING VERSION
"""

import os
import json
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from datetime import datetime

print("🚀 TRAINING - FINAL VERSION")
print("="*80)

# Load config
with open('config.json') as f:
    config = json.load(f)

print(f"Model: {config['base_model']}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load datasets
print("\n📥 Loading datasets...")
tokenized_datasets = load_from_disk('data/processed/tokenized_datasets')

# CRITICAL: Verify data structure
sample = tokenized_datasets['train'][0]
print(f"✅ Data loaded")
print(f"   Sample input_ids length: {len(sample['input_ids'])}")
print(f"   Sample labels length: {len(sample['labels'])}")
print(f"   Columns: {tokenized_datasets['train'].column_names}")

# Limit for faster training
if config.get('max_train_samples'):
    tokenized_datasets['train'] = tokenized_datasets['train'].select(range(min(config['max_train_samples'], len(tokenized_datasets['train']))))
    tokenized_datasets['validation'] = tokenized_datasets['validation'].select(range(min(config['max_val_samples'], len(tokenized_datasets['validation']))))

print(f"   Train: {len(tokenized_datasets['train']):,}")
print(f"   Val: {len(tokenized_datasets['validation']):,}")

# Load model
print("\n📥 Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    config['base_model'],
    torch_dtype=torch.float32,
    device_map="auto",
    low_cpu_mem_usage=True,
)

print(f"✅ Model loaded ({model.num_parameters():,} params)")

# LoRA
print("\n🔧 Applying LoRA...")
lora_config = LoraConfig(
    r=config['lora_r'],
    lora_alpha=config['lora_alpha'],
    target_modules=config['lora_target_modules'],
    lora_dropout=config['lora_dropout'],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ LoRA applied ({trainable:,} trainable params)")

# Training args
output_dir = "./models/checkpoints/experiment_1_baseline"
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=config['num_epochs'],
    per_device_train_batch_size=1,  # Start with batch size 1
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=config['learning_rate'],
    weight_decay=config['weight_decay'],
    warmup_ratio=config['warmup_ratio'],
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    fp16=False,
    bf16=False,
    seed=42,
    report_to="none",
    dataloader_pin_memory=False,
    remove_unused_columns=True,  # Remove all non-standard columns
)

# Data collator - CRITICAL FIX
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,  # Pad to multiple of 8 for efficiency
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=data_collator,
)

print("\n✅ Trainer created")

# Train
print("\n" + "="*80)
print("🚀 STARTING TRAINING")
print("="*80)
print(f"Start: {datetime.now().strftime('%H:%M:%S')}")
print("Estimated: 45-60 minutes")
print("="*80 + "\n")

try:
    result = trainer.train()
    
    print("\n✅ TRAINING COMPLETE!")
    print(f"Train loss: {result.metrics['train_loss']:.4f}")
    print(f"Runtime: {result.metrics['train_runtime']/60:.1f} min")
    
    # Save
    trainer.save_model(output_dir)
    trainer.save_metrics("train", result.metrics)
    
    # Eval
    eval_results = trainer.evaluate()
    print(f"Eval loss: {eval_results['eval_loss']:.4f}")
    trainer.save_metrics("eval", eval_results)
    
    # Save info
    info = {
        "model": config['base_model'],
        "train_loss": float(result.metrics['train_loss']),
        "eval_loss": float(eval_results['eval_loss']),
        "perplexity": float(np.exp(eval_results['eval_loss'])),
        "completed": datetime.now().isoformat()
    }
    
    with open(f"{output_dir}/experiment_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\n💾 Saved to: {output_dir}")
    print("="*80)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
