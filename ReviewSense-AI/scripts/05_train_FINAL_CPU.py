#!/usr/bin/env python3
"""
TRAINING SCRIPT - CPU VERSION (STABLE)
Works reliably on Mac, just slower
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

print("🚀 TRAINING - CPU VERSION (STABLE)")
print("="*80)
print("⚠️  Using CPU (slower but reliable)")
print("   Estimated time: 2-3 hours")
print("="*80)

# Load config
with open('config.json') as f:
    config = json.load(f)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("✅ Tokenizer loaded")

# Load datasets
tokenized_datasets = load_from_disk('data/processed/tokenized_datasets')

# Limit dataset
tokenized_datasets['train'] = tokenized_datasets['train'].select(range(min(config['max_train_samples'], len(tokenized_datasets['train']))))
tokenized_datasets['validation'] = tokenized_datasets['validation'].select(range(min(config['max_val_samples'], len(tokenized_datasets['validation']))))

print(f"✅ Data: {len(tokenized_datasets['train']):,} train, {len(tokenized_datasets['validation']):,} val")

# Load model - FORCE CPU
print("\n📥 Loading model on CPU...")
model = AutoModelForCausalLM.from_pretrained(
    config['base_model'],
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)

# Move model to CPU explicitly
model = model.to('cpu')

print(f"✅ Model on CPU ({model.num_parameters():,} params)")

# LoRA
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

print(f"✅ LoRA applied ({trainable:,} trainable)")

# Training args - FORCE CPU
output_dir = "./models/checkpoints/experiment_1_baseline"
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=config['num_epochs'],
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=config['learning_rate'],
    weight_decay=config['weight_decay'],
    warmup_ratio=config['warmup_ratio'],
    logging_steps=25,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    fp16=False,
    bf16=False,
    seed=42,
    report_to="none",
    use_cpu=True,  # FORCE CPU
    no_cuda=True,  # DISABLE CUDA/MPS
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=data_collator,
)

print("✅ Trainer ready")

# Train
print("\n" + "="*80)
print("🚀 STARTING TRAINING")
print("="*80)
start_time = datetime.now()
print(f"Start: {start_time.strftime('%H:%M:%S')}")
print("⏱️  CPU training is slower but stable")
print("   Estimated: 2-3 hours")
print("   Feel free to use other apps while training")
print("="*80 + "\n")

try:
    result = trainer.train()
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"\nTrain loss: {result.metrics['train_loss']:.4f}")
    print(f"Runtime: {duration/60:.1f} minutes ({duration/3600:.2f} hours)")
    print(f"Samples/sec: {result.metrics.get('train_samples_per_second', 0):.2f}")
    
    # Save
    trainer.save_model(output_dir)
    trainer.save_metrics("train", result.metrics)
    
    # Eval
    print("\n📊 Evaluating...")
    eval_results = trainer.evaluate()
    print(f"Eval loss: {eval_results['eval_loss']:.4f}")
    print(f"Perplexity: {np.exp(eval_results['eval_loss']):.2f}")
    trainer.save_metrics("eval", eval_results)
    
    # Save info
    info = {
        "experiment": "experiment_1_baseline",
        "model": config['base_model'],
        "device": "cpu",
        "config": {
            "lora_r": config['lora_r'],
            "learning_rate": config['learning_rate'],
            "batch_size": 1,
            "epochs": config['num_epochs']
        },
        "results": {
            "train_loss": float(result.metrics['train_loss']),
            "eval_loss": float(eval_results['eval_loss']),
            "perplexity": float(np.exp(eval_results['eval_loss'])),
            "runtime_hours": duration / 3600
        },
        "completed": end_time.isoformat()
    }
    
    with open(f"{output_dir}/experiment_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\n💾 Model saved to: {output_dir}")
    
    print("\n" + "="*80)
    print("🎉 EXPERIMENT 1 COMPLETE!")
    print("="*80)
    print(f"\n📊 Summary:")
    print(f"   Train loss: {result.metrics['train_loss']:.4f}")
    print(f"   Eval loss: {eval_results['eval_loss']:.4f}")
    print(f"   Perplexity: {np.exp(eval_results['eval_loss']):.2f}")
    print(f"   Duration: {duration/3600:.2f} hours")
    print(f"\n💡 Model is ready for evaluation and deployment!")
    print("="*80)
    
except KeyboardInterrupt:
    print("\n\n⚠️  Training interrupted")
    print("💾 Saving checkpoint...")
    trainer.save_model(f"{output_dir}/interrupted")
    print("✅ Saved. You can resume later if needed.")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
