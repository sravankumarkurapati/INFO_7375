#!/usr/bin/env python3
"""
STEP 2: DATA PREPROCESSING & CLEANING
Cleans, filters, and prepares data for training
Runtime: ~10 minutes
"""

import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Enable progress bars for pandas
tqdm.pandas()

print("="*80)
print("🧹 DATA PREPROCESSING & CLEANING")
print("="*80)
print("\n🎯 Goal: Clean and prepare data for training")
print("⏱️  Estimated time: 10 minutes\n")

# ============================================================================
# LOAD RAW DATA
# ============================================================================
print("="*80)
print("1️⃣  LOADING RAW DATA")
print("="*80)

df = pd.read_csv('data/raw/all_reviews_combined.csv')
print(f"✅ Loaded {len(df):,} reviews")
print(f"   Columns: {list(df.columns)}")
print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

initial_count = len(df)

# ============================================================================
# CLEANING FUNCTIONS
# ============================================================================

def clean_text(text):
    """Clean and normalize review text"""
    if pd.isna(text) or not isinstance(text, str):
        return None
    
    # Convert to string
    text = str(text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove excessive punctuation (more than 3 in a row)
    text = re.sub(r'([!?.]){4,}', r'\1\1\1', text)
    
    # Remove non-ASCII characters (keep basic punctuation)
    text = ''.join(char for char in text if ord(char) < 128)
    
    # Strip whitespace
    text = text.strip()
    
    return text if len(text) > 10 else None

def is_valid_review(text, min_length=20, max_length=2000):
    """Check if review meets quality criteria"""
    if not text or not isinstance(text, str):
        return False
    
    # Length check
    if not (min_length <= len(text) <= max_length):
        return False
    
    # Minimum word count
    words = text.split()
    if len(words) < 5:
        return False
    
    # Check for spam patterns
    spam_patterns = [
        r'^(.)\1{20,}',  # Repeated character spam (e.g., "aaaaaaaa...")
        r'(?i)(buy now|click here|visit our website|limited time)',  # Spam phrases
        r'(?i)(viagra|cialis|lottery|casino)',  # Common spam words
    ]
    
    for pattern in spam_patterns:
        if re.search(pattern, text):
            return False
    
    # Check for minimum unique words (avoid repetitive spam)
    unique_words = set(words)
    if len(unique_words) / len(words) < 0.3:  # Less than 30% unique words
        return False
    
    return True

# ============================================================================
# APPLY CLEANING
# ============================================================================
print("\n" + "="*80)
print("2️⃣  CLEANING TEXT")
print("="*80)

print("🔄 Cleaning review text...")
df['text_clean'] = df['text'].progress_apply(clean_text)

# Remove null cleaned texts
before_null = len(df)
df = df[df['text_clean'].notna()]
removed_null = before_null - len(df)
print(f"✅ Cleaned text")
print(f"   Removed {removed_null:,} reviews with invalid text")

# ============================================================================
# QUALITY FILTERING
# ============================================================================
print("\n" + "="*80)
print("3️⃣  FILTERING QUALITY")
print("="*80)

print("🔍 Applying quality filters...")
before_filter = len(df)
df = df[df['text_clean'].apply(is_valid_review)]
removed_filter = before_filter - len(df)

print(f"✅ Quality filtering complete")
print(f"   Removed {removed_filter:,} low-quality reviews")
print(f"   Retained {len(df):,} reviews ({len(df)/initial_count*100:.1f}% of original)")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*80)
print("4️⃣  ENGINEERING FEATURES")
print("="*80)

print("⚙️  Creating text features...")

# Text statistics
df['text_length'] = df['text_clean'].str.len()
df['word_count'] = df['text_clean'].str.split().str.len()
df['avg_word_length'] = df['text_length'] / df['word_count']

# Punctuation features
df['exclamation_count'] = df['text_clean'].str.count('!')
df['question_count'] = df['text_clean'].str.count('\?')
df['period_count'] = df['text_clean'].str.count('\.')

# Capital letters ratio
df['capital_ratio'] = df['text_clean'].apply(
    lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
)

# Sentiment category
df['sentiment'] = df['rating'].apply(
    lambda x: 'positive' if x >= 4 else 'negative' if x <= 2 else 'neutral'
)

print("✅ Features created:")
print(f"   - text_length, word_count, avg_word_length")
print(f"   - exclamation_count, question_count, period_count")
print(f"   - capital_ratio, sentiment")

# ============================================================================
# REMOVE DUPLICATES
# ============================================================================
print("\n" + "="*80)
print("5️⃣  REMOVING DUPLICATES")
print("="*80)

before_dedup = len(df)
df = df.drop_duplicates(subset=['text_clean'], keep='first')
duplicates_removed = before_dedup - len(df)

print(f"✅ Removed {duplicates_removed:,} duplicate reviews")

# ============================================================================
# BALANCE DATASET
# ============================================================================
print("\n" + "="*80)
print("6️⃣  BALANCING DATASET")
print("="*80)

print("⚖️  Current rating distribution:")
print(df['rating'].value_counts().sort_index())

# Balance by rating (cap each rating at 8000 samples)
max_per_rating = 8000

balanced_dfs = []
for rating in sorted(df['rating'].unique()):
    rating_df = df[df['rating'] == rating]
    
    if len(rating_df) > max_per_rating:
        rating_df = rating_df.sample(n=max_per_rating, random_state=42)
    
    balanced_dfs.append(rating_df)

df_balanced = pd.concat(balanced_dfs, ignore_index=True)

print(f"\n⚖️  Balanced rating distribution:")
print(df_balanced['rating'].value_counts().sort_index())

print(f"\n✅ Balanced dataset: {len(df_balanced):,} reviews")

# ============================================================================
# SAVE PREPROCESSED DATA
# ============================================================================
print("\n" + "="*80)
print("7️⃣  SAVING PREPROCESSED DATA")
print("="*80)

df_balanced.to_csv('data/processed/reviews_cleaned.csv', index=False)
print(f"✅ Saved to: data/processed/reviews_cleaned.csv")

# Save basic statistics
stats = {
    'initial_count': initial_count,
    'after_cleaning': len(df),
    'after_balancing': len(df_balanced),
    'retention_rate': f"{len(df_balanced)/initial_count*100:.1f}%",
    'avg_text_length': df_balanced['text_length'].mean(),
    'avg_word_count': df_balanced['word_count'].mean(),
    'rating_distribution': df_balanced['rating'].value_counts().to_dict(),
    'source_distribution': df_balanced['source'].value_counts().to_dict()
}

import json
with open('data/processed/preprocessing_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print(f"✅ Saved statistics to: data/processed/preprocessing_stats.json")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📊 PREPROCESSING SUMMARY")
print("="*80)

print(f"\n📈 Data Flow:")
print(f"   Initial reviews:      {initial_count:,}")
print(f"   After cleaning:       {len(df):,} ({len(df)/initial_count*100:.1f}%)")
print(f"   After balancing:      {len(df_balanced):,} ({len(df_balanced)/initial_count*100:.1f}%)")

print(f"\n📝 Text Statistics:")
print(f"   Avg text length:      {df_balanced['text_length'].mean():.0f} characters")
print(f"   Avg word count:       {df_balanced['word_count'].mean():.0f} words")
print(f"   Avg word length:      {df_balanced['avg_word_length'].mean():.1f} characters")

print(f"\n⭐ Rating Distribution:")
for rating in sorted(df_balanced['rating'].unique()):
    count = len(df_balanced[df_balanced['rating'] == rating])
    pct = count / len(df_balanced) * 100
    bar = '█' * int(pct / 2)
    print(f"   {rating}⭐: {count:,} ({pct:.1f}%) {bar}")

print(f"\n📦 Source Distribution:")
for source in df_balanced['source'].unique():
    count = len(df_balanced[df_balanced['source'] == source])
    pct = count / len(df_balanced) * 100
    print(f"   {source:12s}: {count:,} ({pct:.1f}%)")

print(f"\n💾 Disk Space:")
print(f"   Raw data:             {initial_count * 0.001:.1f} MB")
print(f"   Processed data:       {df_balanced.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

print("\n" + "="*80)
print("✅ PREPROCESSING COMPLETE!")
print("="*80)
print(f"\n🎯 Next Step: Create training dataset")
print(f"   python3 scripts/03_create_training_data.py")
print("="*80)
