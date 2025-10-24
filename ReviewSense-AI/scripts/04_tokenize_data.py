#!/usr/bin/env python3
"""
STEP 4: TOKENIZE DATA FOR TRAINING (FIXED)
Properly handles variable-length sequences
"""

import os
import json
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from tqdm import tqdm

print("="*80)
print("🔤 TOKENIZE DATA FOR TRAINING (FIXED)")
print("="*80)

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)

print(f"✅ Model: {config['base_model']}")
print(f"   Max length: {config['model_max_length']}")

# Load tokenizer
print("\n📥 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    config['base_model'],
    trust_remote_code=True,
    use_fast=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

tokenizer.padding_side = "right"
print("✅ Tokenizer loaded")

# Load data
print("\n📥 Loading data splits...")

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

train_data = load_jsonl('data/splits/train.jsonl')
val_data = load_jsonl('data/splits/val.jsonl')
test_data = load_jsonl('data/splits/test.jsonl')

print(f"✅ Loaded:")
print(f"   Train: {len(train_data):,}")
print(f"   Val: {len(val_data):,}")
print(f"   Test: {len(test_data):,}")

# Format for training
print("\n🔄 Formatting for instruction tuning...")

def format_instruction(example):
    """Format in chat template"""
    prompt = f"<s>[INST] {example['instruction']} [/INST] {example['response']}</s>"
    return {"text": prompt}

train_formatted = [format_instruction(ex) for ex in tqdm(train_data, desc="Train")]
val_formatted = [format_instruction(ex) for ex in tqdm(val_data, desc="Val")]
test_formatted = [format_instruction(ex) for ex in tqdm(test_data, desc="Test")]

print("✅ Formatted")

# Create datasets
print("\n🔄 Creating HuggingFace datasets...")
train_dataset = Dataset.from_list(train_formatted)
val_dataset = Dataset.from_list(val_formatted)
test_dataset = Dataset.from_list(test_formatted)

dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset
})

print("✅ Datasets created")

# Tokenize with FIXED padding strategy
print("\n🔄 Tokenizing (this may take 5-10 minutes)...")

def tokenize_function(examples):
    """
    Tokenize with proper truncation and NO padding
    Padding will be done dynamically during training
    """
    result = tokenizer(
        examples['text'],
        truncation=True,
        max_length=config['model_max_length'],
        padding=False,  # DO NOT pad here - let data collator handle it
        return_tensors=None,
    )
    
    # For causal LM, labels = input_ids
    result['labels'] = result['input_ids'].copy()
    
    return result

# Tokenize all splits
tokenized_datasets = dataset_dict.map(
    tokenize_function,
    batched=True,
    remove_columns=['text'],
    desc="Tokenizing",
    num_proc=1,  # Single process for stability
)

print("✅ Tokenization complete")

# Verify
print("\n🔍 Verifying tokenized data...")
sample = tokenized_datasets['train'][0]
print(f"   Sample length: {len(sample['input_ids'])} tokens")
print(f"   Has labels: {('labels' in sample)}")
print(f"   Columns: {tokenized_datasets['train'].column_names}")

# Check length distribution
lengths = [len(ex['input_ids']) for ex in tokenized_datasets['train'].select(range(min(100, len(tokenized_datasets['train']))))]
print(f"\n📊 Token length stats (first 100 examples):")
print(f"   Min: {min(lengths)}")
print(f"   Max: {max(lengths)}")
print(f"   Avg: {sum(lengths)/len(lengths):.0f}")

# Save
print("\n💾 Saving tokenized datasets...")
output_path = 'data/processed/tokenized_datasets'
os.makedirs(output_path, exist_ok=True)

tokenized_datasets.save_to_disk(output_path)

print(f"✅ Saved to: {output_path}")

# Save stats
stats = {
    'tokenizer': config['base_model'],
    'max_length': config['model_max_length'],
    'train_examples': len(tokenized_datasets['train']),
    'val_examples': len(tokenized_datasets['validation']),
    'test_examples': len(tokenized_datasets['test']),
    'avg_length': sum(lengths)/len(lengths),
    'min_length': min(lengths),
    'max_length': max(lengths)
}

with open(f'{output_path}/tokenization_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print("\n" + "="*80)
print("✅ TOKENIZATION COMPLETE!")
print("="*80)
print(f"\n📊 Summary:")
print(f"   Train: {len(tokenized_datasets['train']):,} examples")
print(f"   Val: {len(tokenized_datasets['validation']):,} examples")
print(f"   Test: {len(tokenized_datasets['test']):,} examples")
print(f"   Avg length: {sum(lengths)/len(lengths):.0f} tokens")
print(f"\n🚀 Ready for training!")
print("="*80)
