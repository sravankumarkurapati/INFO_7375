#!/usr/bin/env python3
"""
TOKENIZE DATA - FINAL WORKING VERSION
"""

import os
import json
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

print("="*80)
print("🔤 TOKENIZING DATA (FINAL FIX)")
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

print("✅ Tokenizer loaded")

# Load JSONL data
def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

train_data = load_jsonl('data/splits/train.jsonl')
val_data = load_jsonl('data/splits/val.jsonl')
test_data = load_jsonl('data/splits/test.jsonl')

print(f"✅ Loaded {len(train_data):,} train, {len(val_data):,} val, {len(test_data):,} test")

# Format examples
def format_example(ex):
    text = f"<s>[INST] {ex['instruction']} [/INST] {ex['response']}</s>"
    return {"text": text}

print("🔄 Formatting...")
train_formatted = [format_example(ex) for ex in train_data]
val_formatted = [format_example(ex) for ex in val_data]
test_formatted = [format_example(ex) for ex in test_data]

# Create datasets
dataset_dict = DatasetDict({
    'train': Dataset.from_list(train_formatted),
    'validation': Dataset.from_list(val_formatted),
    'test': Dataset.from_list(test_formatted)
})

print("✅ Formatted")

# Tokenize function - CRITICAL FIX
def tokenize_function(examples):
    """Tokenize WITHOUT creating nested lists"""
    
    # Tokenize
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        max_length=config['model_max_length'],
        padding=False,  # No padding - will be done dynamically
    )
    
    # CRITICAL: Copy input_ids to labels (NOT as nested list)
    # The labels should be the same structure as input_ids
    tokenized['labels'] = tokenized['input_ids']
    
    return tokenized

print("🔄 Tokenizing...")

# Tokenize all splits
tokenized_datasets = dataset_dict.map(
    tokenize_function,
    batched=True,
    remove_columns=['text'],
    desc="Tokenizing"
)

print("✅ Tokenized")

# Verify structure
sample = tokenized_datasets['train'][0]
print(f"\n🔍 Verification:")
print(f"   input_ids type: {type(sample['input_ids'])}")
print(f"   input_ids length: {len(sample['input_ids'])}")
print(f"   labels type: {type(sample['labels'])}")
print(f"   labels length: {len(sample['labels'])}")
print(f"   Columns: {tokenized_datasets['train'].column_names}")

# Save
output_path = 'data/processed/tokenized_datasets'
os.makedirs(output_path, exist_ok=True)
tokenized_datasets.save_to_disk(output_path)

print(f"\n💾 Saved to: {output_path}")
print("="*80)
print("✅ TOKENIZATION COMPLETE!")
print("="*80)
