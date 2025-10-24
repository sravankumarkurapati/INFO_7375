#!/usr/bin/env python3
"""
STEP 5: TRAIN MODEL - BASELINE (Fixed)
"""

import os
import sys
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
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
import numpy as np
from datetime import datetime

print("="*80)
print("🚀 TRAINING: BASELINE EXPERIMENT")
print("="*80)
print(f"\n⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)

print(f"✅ Model: {config['base_model']}")

# Setup device
if torch.backends.mps.is_available():
    device = "mps"
    print("✅ Using MPS (Apple Silicon)")
else:
    device = "cpu"
    print("✅ Using CPU")

# Load tokenizer
print("\n📥 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    config['base_model'],
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

tokenizer.padding_side = "right"
print("✅ Tokenizer loaded")

# Load datasets
print("\n📥 Loading datasets...")
tokenized_datasets = load_from_disk('data/processed/tokenized_datasets')

# IMPORTANT FIX: Remove extra columns that cause issues
print("🔧 Removing extra columns...")
columns_to_remove = []
for col in tokenized_datasets['train'].column_names:
    if col not in ['input_ids', 'attention_mask', 'labels']:
        columns_to_remove.append(col)

if columns_to_remove:
    print(f"   Removing: {columns_to_remove}")
    tokenized_datasets = tokenized_datasets.remove_columns(columns_to_remove)

print("✅ Dataset cleaned")
print(f"   Columns: {tokenized_datasets['train'].column_names}")

# Limit dataset
if config.get('max_train_samples'):
    tokenized_datasets['train'] = tokenized_datasets['train'].select(
        range(min(config['max_train_samples'], len(tokenized_datasets['train'])))
    )
    tokenized_datasets['validation'] = tokenized_datasets['validation'].select(
        range(min(config['max_val_samples'], len(tokenized_datasets['validation'])))
    )

print(f"   Train: {len(tokenized_datasets['train']):,} examples")
print(f"   Val: {len(tokenized_datasets['validation']):,} examples")

# Load model
print("\n📥 Loading model...")

try:
    model = AutoModelForCausalLM.from_pretrained(
        config['base_model'],
        torch_dtype=torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print("✅ Model loaded successfully")
    print(f"   Parameters: {model.num_parameters():,}")
    
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)

# Setup LoRA
print("\n🔧 Setting up LoRA...")
lora_config = LoraConfig(
    r=config['lora_r'],
    lora_alpha=config['lora_alpha'],
    target_modules=config['lora_target_modules'],
    lora_dropout=config['lora_dropout'],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

print(f"✅ LoRA applied")
print(f"   Trainable: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

# Training arguments
experiment_name = "experiment_1_baseline"
output_dir = os.path.join(config['output_dir'], experiment_name)
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=config['num_epochs'],
    per_device_train_batch_size=config['batch_size'],
    per_device_eval_batch_size=config['batch_size'],
    gradient_accumulation_steps=config['gradient_accumulation_steps'],
    learning_rate=config['learning_rate'],
    weight_decay=config['weight_decay'],
    warmup_ratio=config['warmup_ratio'],
    max_grad_norm=config['max_grad_norm'],
    optim=config['optimizer'],
    lr_scheduler_type=config['lr_scheduler'],
    fp16=False,
    bf16=False,
    logging_steps=config['logging_steps'],
    logging_first_step=True,
    eval_strategy="steps",
    eval_steps=config['eval_steps'],
    save_strategy="steps",
    save_steps=config['save_steps'],
    save_total_limit=config['save_total_limit'],
    load_best_model_at_end=False,
    seed=config['seed'],
    report_to="none",
    dataloader_pin_memory=False,  # Fix for MPS
)

steps_per_epoch = len(tokenized_datasets['train']) // (config['batch_size'] * config['gradient_accumulation_steps'])
total_steps = steps_per_epoch * config['num_epochs']

print(f"\n✅ Training config:")
print(f"   Steps per epoch: {steps_per_epoch}")
print(f"   Total steps: {total_steps}")

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=data_collator,
)

print(f"\n✅ Trainer created")

# Train
print("\n" + "="*80)
print("🚀 STARTING TRAINING")
print("="*80)
print(f"⏰ Start: {datetime.now().strftime('%H:%M:%S')}")
print(f"⏱️  Estimated: 45-60 minutes")
print("="*80 + "\n")

try:
    train_result = trainer.train()
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    
    metrics = train_result.metrics
    print(f"\n📊 Training Metrics:")
    print(f"   Loss: {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"   Runtime: {metrics.get('train_runtime', 0)/60:.1f} minutes")
    print(f"   Samples/sec: {metrics.get('train_samples_per_second', 'N/A'):.2f}")
    
    trainer.save_model(output_dir)
    trainer.save_metrics("train", metrics)
    print(f"\n💾 Model saved to: {output_dir}")
    
    # Evaluate
    print("\n📊 Evaluating...")
    eval_results = trainer.evaluate()
    print(f"   Eval loss: {eval_results.get('eval_loss', 'N/A'):.4f}")
    print(f"   Perplexity: {np.exp(eval_results.get('eval_loss', 0)):.2f}")
    trainer.save_metrics("eval", eval_results)
    
    # Save experiment info
    experiment_info = {
        "experiment": experiment_name,
        "model": config['base_model'],
        "config": {
            "lora_r": config['lora_r'],
            "learning_rate": config['learning_rate'],
            "batch_size": config['batch_size'],
            "epochs": config['num_epochs']
        },
        "results": {
            "train_loss": float(metrics.get('train_loss', 0)),
            "eval_loss": float(eval_results.get('eval_loss', 0)),
            "perplexity": float(np.exp(eval_results.get('eval_loss', 0))),
            "runtime_minutes": float(metrics.get('train_runtime', 0) / 60)
        },
        "completed": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(output_dir, 'experiment_info.json'), 'w') as f:
        json.dump(experiment_info, f, indent=2)
    
    print("\n" + "="*80)
    print("🎉 EXPERIMENT 1 COMPLETE!")
    print("="*80)
    print(f"\n📊 Summary:")
    print(f"   Train loss: {metrics.get('train_loss', 0):.4f}")
    print(f"   Eval loss: {eval_results.get('eval_loss', 0):.4f}")
    print(f"   Perplexity: {np.exp(eval_results.get('eval_loss', 0)):.2f}")
    print(f"   Duration: {metrics.get('train_runtime', 0)/60:.1f} minutes")
    print(f"\n💾 Saved to: {output_dir}")
    print("="*80)
    
except KeyboardInterrupt:
    print("\n⚠️  Training interrupted")
    trainer.save_model(os.path.join(output_dir, "interrupted"))
    print("💾 Checkpoint saved")
    sys.exit(0)
    
except Exception as e:
    print(f"\n❌ Training error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
