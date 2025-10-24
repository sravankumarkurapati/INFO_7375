#!/usr/bin/env python3
"""
STEP 1: DATA COLLECTION
Collects 50,000+ product reviews from multiple sources
Runtime: ~30 minutes
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

print("="*80)
print("📦 REVIEWSENSE AI - DATA COLLECTION")
print("="*80)
print("\n🎯 Goal: Collect 50,000+ product reviews")
print("⏱️  Estimated time: 30 minutes\n")

# Create directories
os.makedirs('data/raw', exist_ok=True)

# ============================================================================
# SOURCE 1: Amazon Reviews (30,000 reviews)
# ============================================================================
print("\n" + "="*80)
print("1️⃣  COLLECTING AMAZON REVIEWS")
print("="*80)

try:
    print("📥 Loading Amazon Polarity dataset...")
    amazon_data = load_dataset("amazon_polarity", split="train[:30000]", trust_remote_code=True)
    
    amazon_reviews = []
    print("🔄 Processing Amazon reviews...")
    
    for idx, review in enumerate(tqdm(amazon_data, desc="Amazon")):
        amazon_reviews.append({
            'review_id': f"AMZ_{idx:06d}",
            'text': review['content'],
            'rating': 5 if review['label'] == 1 else 1,  # Binary: 5-star or 1-star
            'title': review['title'],
            'category': 'electronics',
            'verified': True,
            'helpful_votes': random.randint(0, 100),
            'source': 'amazon'
        })
    
    df_amazon = pd.DataFrame(amazon_reviews)
    df_amazon.to_csv('data/raw/amazon_reviews.csv', index=False)
    
    print(f"✅ Collected {len(amazon_reviews):,} Amazon reviews")
    print(f"💾 Saved to: data/raw/amazon_reviews.csv")
    
except Exception as e:
    print(f"⚠️  Error loading Amazon data: {e}")
    print("Continuing with other sources...")
    df_amazon = pd.DataFrame()

# ============================================================================
# SOURCE 2: Yelp Reviews (20,000 reviews)
# ============================================================================
print("\n" + "="*80)
print("2️⃣  COLLECTING YELP REVIEWS")
print("="*80)

try:
    print("📥 Loading Yelp dataset...")
    yelp_data = load_dataset("yelp_review_full", split="train[:20000]", trust_remote_code=True)
    
    yelp_reviews = []
    print("🔄 Processing Yelp reviews...")
    
    for idx, review in enumerate(tqdm(yelp_data, desc="Yelp")):
        yelp_reviews.append({
            'review_id': f"YELP_{idx:06d}",
            'text': review['text'],
            'rating': review['label'] + 1,  # Convert 0-4 to 1-5
            'title': '',
            'category': 'restaurant',
            'verified': True,
            'helpful_votes': random.randint(0, 50),
            'source': 'yelp'
        })
    
    df_yelp = pd.DataFrame(yelp_reviews)
    df_yelp.to_csv('data/raw/yelp_reviews.csv', index=False)
    
    print(f"✅ Collected {len(yelp_reviews):,} Yelp reviews")
    print(f"💾 Saved to: data/raw/yelp_reviews.csv")
    
except Exception as e:
    print(f"⚠️  Error loading Yelp data: {e}")
    print("Continuing with synthetic generation...")
    df_yelp = pd.DataFrame()

# ============================================================================
# SOURCE 3: Generate Synthetic Multi-Aspect Reviews (10,000 reviews)
# ============================================================================
print("\n" + "="*80)
print("3️⃣  GENERATING SYNTHETIC MULTI-ASPECT REVIEWS")
print("="*80)

print("🎨 Creating synthetic reviews with aspect annotations...")

# Product categories
products = [
    'laptop', 'smartphone', 'headphones', 'tablet', 'smartwatch',
    'camera', 'speaker', 'keyboard', 'mouse', 'monitor',
    'coffee_maker', 'blender', 'vacuum', 'air_purifier', 'printer'
]

# Aspect templates
aspect_templates = {
    'quality': {
        'positive': [
            "Excellent build quality. Materials feel premium and well-constructed.",
            "Outstanding quality. Far exceeds expectations for the price point.",
            "Top-notch quality. You can tell this was made with attention to detail."
        ],
        'negative': [
            "Poor quality. Feels cheap and flimsy right out of the box.",
            "Disappointing quality. Materials feel low-grade and poorly assembled.",
            "Subpar quality. Not what I expected based on the product description."
        ]
    },
    'durability': {
        'positive': [
            "Very durable. Been using it daily for 6 months without any issues.",
            "Built to last. Shows no signs of wear despite heavy use.",
            "Impressive durability. Withstood drops and rough handling."
        ],
        'negative': [
            "Not durable. Broke within 2 weeks of normal use.",
            "Poor durability. Parts started failing after a month.",
            "Fragile construction. Doesn't hold up to regular use."
        ]
    },
    'value': {
        'positive': [
            "Great value for money. Comparable products cost twice as much.",
            "Excellent price-to-performance ratio. Worth every penny.",
            "Outstanding value. Features usually found in much more expensive models."
        ],
        'negative': [
            "Overpriced for what you get. Better alternatives at this price point.",
            "Poor value. Not worth the money given the quality issues.",
            "Too expensive. Similar products offer more for less."
        ]
    },
    'performance': {
        'positive': [
            "Excellent performance. Handles everything I throw at it smoothly.",
            "Outstanding performance. Fast, responsive, and reliable.",
            "Superior performance. Exceeds specifications in real-world use."
        ],
        'negative': [
            "Poor performance. Struggles with basic tasks.",
            "Disappointing performance. Doesn't live up to advertised specs.",
            "Sluggish performance. Noticeable lag and slowdowns."
        ]
    },
    'design': {
        'positive': [
            "Beautiful design. Sleek and modern aesthetic.",
            "Excellent design. Thoughtful layout and intuitive controls.",
            "Attractive design. Looks more expensive than it is."
        ],
        'negative': [
            "Ugly design. Looks cheap and outdated.",
            "Poor design. Awkward layout and confusing controls.",
            "Bland design. Nothing special about the appearance."
        ]
    }
}

synthetic_reviews = []

for i in tqdm(range(10000), desc="Generating synthetic"):
    product = random.choice(products)
    
    # Generate rating (skewed towards positive)
    rating = random.choices([1, 2, 3, 4, 5], weights=[5, 10, 15, 30, 40])[0]
    
    # Select 2-4 aspects to mention
    num_aspects = random.randint(2, 4)
    selected_aspects = random.sample(list(aspect_templates.keys()), num_aspects)
    
    # Generate review text
    review_parts = []
    aspect_scores = {}
    
    for aspect in selected_aspects:
        # Mostly align with overall rating, sometimes contradict (realistic!)
        if random.random() < 0.85:  # 85% alignment
            sentiment = 'positive' if rating >= 4 else 'negative'
        else:  # 15% contradiction
            sentiment = 'negative' if rating >= 4 else 'positive'
        
        text = random.choice(aspect_templates[aspect][sentiment])
        review_parts.append(text)
        
        # Score this aspect
        aspect_scores[aspect] = 5 if sentiment == 'positive' else 2
    
    # Combine into full review
    review_text = f"I purchased this {product.replace('_', ' ')}. " + " ".join(review_parts)
    
    synthetic_reviews.append({
        'review_id': f"SYN_{i:06d}",
        'text': review_text,
        'rating': rating,
        'title': f"Review of {product.replace('_', ' ')}",
        'category': product,
        'verified': random.choice([True, False]),
        'helpful_votes': random.randint(0, 100),
        'aspects': json.dumps(aspect_scores),
        'source': 'synthetic'
    })

df_synthetic = pd.DataFrame(synthetic_reviews)
df_synthetic.to_csv('data/raw/synthetic_reviews.csv', index=False)

print(f"✅ Generated {len(synthetic_reviews):,} synthetic reviews")
print(f"💾 Saved to: data/raw/synthetic_reviews.csv")

# ============================================================================
# COMBINE ALL DATA
# ============================================================================
print("\n" + "="*80)
print("4️⃣  COMBINING ALL DATA SOURCES")
print("="*80)

# Combine all dataframes
all_dfs = []
if not df_amazon.empty:
    all_dfs.append(df_amazon)
if not df_yelp.empty:
    all_dfs.append(df_yelp)
if not df_synthetic.empty:
    all_dfs.append(df_synthetic)

if not all_dfs:
    print("❌ No data collected! Check your internet connection.")
    sys.exit(1)

df_combined = pd.concat(all_dfs, ignore_index=True)

# Standardize columns
df_combined = df_combined.fillna({
    'title': '',
    'helpful_votes': 0,
    'verified': False,
    'aspects': '{}'
})

# Basic statistics
print(f"\n📊 COLLECTION SUMMARY:")
print(f"   Total reviews: {len(df_combined):,}")
print(f"\n   By source:")
for source in df_combined['source'].unique():
    count = len(df_combined[df_combined['source'] == source])
    print(f"      {source}: {count:,}")

print(f"\n   By rating:")
for rating in sorted(df_combined['rating'].unique()):
    count = len(df_combined[df_combined['rating'] == rating])
    pct = count / len(df_combined) * 100
    print(f"      {rating}⭐: {count:,} ({pct:.1f}%)")

# Save combined data
df_combined.to_csv('data/raw/all_reviews_combined.csv', index=False)
print(f"\n💾 Saved combined data to: data/raw/all_reviews_combined.csv")

# ============================================================================
# COMPLETION
# ============================================================================
print("\n" + "="*80)
print("✅ DATA COLLECTION COMPLETE!")
print("="*80)
print(f"\n📊 Total Dataset Size: {len(df_combined):,} reviews")
print(f"💾 Total Disk Space: ~{df_combined.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
print(f"\n🎯 Next Step: Run preprocessing")
print(f"   python3 scripts/02_preprocess_data.py")
print("="*80)
