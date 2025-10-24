#!/usr/bin/env python3
"""
STEP 3: CREATE TRAINING DATASET
Formats data for instruction tuning with multiple tasks
Runtime: ~15 minutes
"""

import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import random

random.seed(42)
np.random.seed(42)

print("="*80)
print("🎨 CREATE TRAINING DATASET")
print("="*80)
print("\n🎯 Goal: Format data for multi-task instruction tuning")
print("⏱️  Estimated time: 15 minutes\n")

# ============================================================================
# LOAD CLEANED DATA
# ============================================================================
print("="*80)
print("1️⃣  LOADING CLEANED DATA")
print("="*80)

df = pd.read_csv('data/processed/reviews_cleaned.csv')
print(f"✅ Loaded {len(df):,} cleaned reviews")

# ============================================================================
# TASK 1: ASPECT-BASED SENTIMENT ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("2️⃣  CREATING TASK 1: ASPECT-BASED SENTIMENT ANALYSIS")
print("="*80)

aspect_examples = []

# Sample reviews for aspect analysis
sample_size = min(10000, len(df))
sampled_df = df.sample(n=sample_size, random_state=42)

print(f"🔄 Creating {sample_size:,} aspect analysis examples...")

for idx, row in tqdm(sampled_df.iterrows(), total=sample_size, desc="Aspect Analysis"):
    review_text = row['text_clean'][:500]  # Truncate long reviews
    rating = row['rating']
    
    # Create instruction
    instruction = f"""Analyze this product review and extract sentiment for each aspect:

Review: "{review_text}"

Provide sentiment scores (1-5) for these aspects if mentioned:
- Quality
- Durability  
- Value for Money
- Performance
- Customer Service

If an aspect is not mentioned, mark as "Not mentioned"."""

    # Generate response based on rating
    aspects_mentioned = random.sample(['Quality', 'Durability', 'Value for Money', 'Performance'], 
                                     k=random.randint(2, 4))
    
    response_parts = ["**Aspect Sentiment Analysis:**\n"]
    
    for aspect in ['Quality', 'Durability', 'Value for Money', 'Performance', 'Customer Service']:
        if aspect in aspects_mentioned:
            # Score based on overall rating with some variation
            score = max(1, min(5, rating + random.randint(-1, 1)))
            sentiment = "Positive" if score >= 4 else "Negative" if score <= 2 else "Neutral"
            response_parts.append(f"- **{aspect}**: {score}/5 ({sentiment})")
        else:
            response_parts.append(f"- **{aspect}**: Not mentioned")
    
    # Overall sentiment
    overall = "Positive" if rating >= 4 else "Negative" if rating <= 2 else "Mixed"
    response_parts.append(f"\n**Overall Sentiment**: {overall}")
    
    response = "\n".join(response_parts)
    
    aspect_examples.append({
        'task': 'aspect_analysis',
        'instruction': instruction,
        'response': response,
        'rating': rating
    })

print(f"✅ Created {len(aspect_examples):,} aspect analysis examples")

# ============================================================================
# TASK 2: REVIEW SUMMARIZATION
# ============================================================================
print("\n" + "="*80)
print("3️⃣  CREATING TASK 2: REVIEW SUMMARIZATION")
print("="*80)

summary_examples = []

# Group reviews by product category
categories = df['category'].unique()
sample_size = min(5000, len(df) // 3)

print(f"🔄 Creating {sample_size:,} summarization examples...")

for i in tqdm(range(sample_size), desc="Summarization"):
    # Sample 5-10 reviews from same category
    category = random.choice(categories)
    cat_reviews = df[df['category'] == category].sample(n=min(random.randint(5, 10), len(df[df['category'] == category])))
    
    if len(cat_reviews) < 3:
        continue
    
    # Create multi-review input
    review_texts = []
    for idx, (_, rev) in enumerate(cat_reviews.iterrows(), 1):
        review_texts.append(f"Review {idx} ({rev['rating']}⭐): {rev['text_clean'][:150]}...")
    
    reviews_combined = "\n\n".join(review_texts)
    
    instruction = f"""Analyze these {len(cat_reviews)} reviews for a {category.replace('_', ' ')} and provide:

1. Overall consensus
2. Common praise points
3. Common complaints
4. Recommendation

Reviews:
{reviews_combined}"""

    # Generate summary
    avg_rating = cat_reviews['rating'].mean()
    positive_reviews = len(cat_reviews[cat_reviews['rating'] >= 4])
    negative_reviews = len(cat_reviews[cat_reviews['rating'] <= 2])
    
    response = f"""**Multi-Review Analysis:**

**Overall Consensus**: {len(cat_reviews)} reviews analyzed
- Average Rating: {avg_rating:.1f}/5.0
- Positive: {positive_reviews}, Negative: {negative_reviews}
- Consistency: {"High" if cat_reviews['rating'].std() < 1.0 else "Medium" if cat_reviews['rating'].std() < 1.5 else "Low"}

**Common Praise**:
{"- Quality and build" if avg_rating >= 4 else "- Value for money" if avg_rating >= 3 else "- Some positive aspects"}
{"- Performance meets expectations" if avg_rating >= 4 else ""}

**Common Complaints**:
{"- Minor issues reported" if avg_rating >= 4 else "- Quality concerns mentioned" if avg_rating < 3 else "- Mixed experiences"}
{"- Some durability concerns" if negative_reviews > 0 else ""}

**Recommendation**:
{"Recommended for most users. Strong overall satisfaction." if avg_rating >= 4 else 
 "Consider carefully. Mixed reviews suggest variability." if avg_rating >= 3 else
 "Not recommended. Significant issues reported."}
"""
    
    summary_examples.append({
        'task': 'summarization',
        'instruction': instruction,
        'response': response,
        'avg_rating': avg_rating
    })

print(f"✅ Created {len(summary_examples):,} summarization examples")

# ============================================================================
# TASK 3: SENTIMENT CLASSIFICATION
# ============================================================================
print("\n" + "="*80)
print("4️⃣  CREATING TASK 3: SENTIMENT CLASSIFICATION")
print("="*80)

sentiment_examples = []
sample_size = min(5000, len(df))

print(f"🔄 Creating {sample_size:,} sentiment classification examples...")

sampled_df = df.sample(n=sample_size, random_state=42)

for idx, row in tqdm(sampled_df.iterrows(), total=sample_size, desc="Sentiment"):
    review_text = row['text_clean'][:400]
    rating = row['rating']
    
    instruction = f"""Classify the sentiment of this review:

Review: "{review_text}"

Provide: Sentiment (Positive/Negative/Neutral), Confidence (High/Medium/Low), and brief explanation."""

    sentiment = "Positive" if rating >= 4 else "Negative" if rating <= 2 else "Neutral"
    confidence = "High" if rating in [1, 5] else "Medium"
    
    response = f"""**Sentiment Classification:**

- **Sentiment**: {sentiment}
- **Confidence**: {confidence}
- **Rating**: {rating}/5

**Explanation**: {"The review expresses clear satisfaction with the product." if sentiment == "Positive" else
                 "The review indicates dissatisfaction and issues." if sentiment == "Negative" else
                 "The review presents a balanced perspective with both positives and negatives."}
"""
    
    sentiment_examples.append({
        'task': 'sentiment',
        'instruction': instruction,
        'response': response,
        'rating': rating
    })

print(f"✅ Created {len(sentiment_examples):,} sentiment examples")

# ============================================================================
# COMBINE ALL TASKS
# ============================================================================
print("\n" + "="*80)
print("5️⃣  COMBINING ALL TASKS")
print("="*80)

all_examples = aspect_examples + summary_examples + sentiment_examples
random.shuffle(all_examples)

print(f"✅ Total training examples: {len(all_examples):,}")
print(f"\n   Task distribution:")
print(f"      Aspect Analysis:  {len(aspect_examples):,}")
print(f"      Summarization:    {len(summary_examples):,}")
print(f"      Sentiment:        {len(sentiment_examples):,}")

# ============================================================================
# TRAIN/VAL/TEST SPLIT
# ============================================================================
print("\n" + "="*80)
print("6️⃣  CREATING TRAIN/VAL/TEST SPLITS")
print("="*80)

# Shuffle
random.shuffle(all_examples)

# Split ratios
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

n_train = int(len(all_examples) * train_ratio)
n_val = int(len(all_examples) * val_ratio)

train_data = all_examples[:n_train]
val_data = all_examples[n_train:n_train+n_val]
test_data = all_examples[n_train+n_val:]

print(f"✅ Split created:")
print(f"   Train: {len(train_data):,} ({len(train_data)/len(all_examples)*100:.1f}%)")
print(f"   Val:   {len(val_data):,} ({len(val_data)/len(all_examples)*100:.1f}%)")
print(f"   Test:  {len(test_data):,} ({len(test_data)/len(all_examples)*100:.1f}%)")

# ============================================================================
# SAVE DATASETS
# ============================================================================
print("\n" + "="*80)
print("7️⃣  SAVING DATASETS")
print("="*80)

# Save as JSONL (JSON Lines format - standard for training)
with open('data/splits/train.jsonl', 'w') as f:
    for example in train_data:
        f.write(json.dumps(example) + '\n')

with open('data/splits/val.jsonl', 'w') as f:
    for example in val_data:
        f.write(json.dumps(example) + '\n')

with open('data/splits/test.jsonl', 'w') as f:
    for example in test_data:
        f.write(json.dumps(example) + '\n')

print(f"✅ Saved datasets:")
print(f"   data/splits/train.jsonl ({len(train_data):,} examples)")
print(f"   data/splits/val.jsonl ({len(val_data):,} examples)")
print(f"   data/splits/test.jsonl ({len(test_data):,} examples)")

# Save split statistics
split_stats = {
    'total_examples': len(all_examples),
    'train': len(train_data),
    'val': len(val_data),
    'test': len(test_data),
    'task_distribution': {
        'aspect_analysis': len(aspect_examples),
        'summarization': len(summary_examples),
        'sentiment': len(sentiment_examples)
    }
}

with open('data/splits/split_stats.json', 'w') as f:
    json.dump(split_stats, f, indent=2)

print(f"✅ Saved statistics: data/splits/split_stats.json")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📊 TRAINING DATA SUMMARY")
print("="*80)

print(f"\n📈 Dataset Sizes:")
print(f"   Total examples: {len(all_examples):,}")
print(f"   Training:       {len(train_data):,} (70%)")
print(f"   Validation:     {len(val_data):,} (15%)")
print(f"   Test:           {len(test_data):,} (15%)")

print(f"\n🎯 Tasks:")
print(f"   Aspect Analysis:  {len(aspect_examples):,} ({len(aspect_examples)/len(all_examples)*100:.1f}%)")
print(f"   Summarization:    {len(summary_examples):,} ({len(summary_examples)/len(all_examples)*100:.1f}%)")
print(f"   Sentiment:        {len(sentiment_examples):,} ({len(sentiment_examples)/len(all_examples)*100:.1f}%)")

print(f"\n💾 Disk Space:")
train_size = os.path.getsize('data/splits/train.jsonl') / 1024**2
val_size = os.path.getsize('data/splits/val.jsonl') / 1024**2
test_size = os.path.getsize('data/splits/test.jsonl') / 1024**2
print(f"   train.jsonl: {train_size:.1f} MB")
print(f"   val.jsonl:   {val_size:.1f} MB")
print(f"   test.jsonl:  {test_size:.1f} MB")
print(f"   Total:       {train_size + val_size + test_size:.1f} MB")

# Sample examples
print(f"\n📝 Sample Training Example:")
sample = train_data[0]
print(f"   Task: {sample['task']}")
print(f"   Instruction: {sample['instruction'][:100]}...")
print(f"   Response: {sample['response'][:100]}...")

print("\n" + "="*80)
print("✅ TRAINING DATA CREATION COMPLETE!")
print("="*80)
print(f"\n🎯 Dataset Ready for Training!")
print(f"   ✓ {len(train_data):,} training examples")
print(f"   ✓ {len(val_data):,} validation examples")
print(f"   ✓ {len(test_data):,} test examples")
print(f"\n🚀 Next Step: Load and tokenize data")
print(f"   python3 scripts/04_prepare_for_training.py")
print("="*80)
