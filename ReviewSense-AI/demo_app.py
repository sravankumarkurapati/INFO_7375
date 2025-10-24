"""
ReviewSense AI - Demo Application
Interactive demo for review summarization using exp2 (Best Model)
Achievement: 54.8% improvement over baseline (ROUGE-1: 0.5079)
"""

import torch
import re
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path.home() / "Documents/INFO_7375/Assigment-FineTuningLLM/ReviewSense-AI"
MODELS_DIR = BASE_DIR / "all_training_results" / "models"
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Use best model from evaluation
BEST_MODEL = "exp2"  # 0.5079 ROUGE-1, 54.8% improvement

# Sample reviews for demo
SAMPLE_REVIEWS = [
    {
        "id": 1,
        "text": "This restaurant was absolutely amazing! The food was delicious, service was impeccable, and the atmosphere was perfect for a romantic dinner. Our waiter was very attentive and made great recommendations. The pasta was cooked to perfection and the dessert was heavenly. Will definitely come back!",
        "rating": 5,
        "type": "Positive"
    },
    {
        "id": 2,
        "text": "Very disappointing experience. Waited over an hour for our food, and when it finally arrived, it was cold. The staff seemed overwhelmed and disorganized. The only positive was that the drinks were decent. Not worth the price at all.",
        "rating": 2,
        "type": "Negative"
    },
    {
        "id": 3,
        "text": "Decent place for a quick lunch. Nothing special but nothing terrible either. Food was okay, prices were reasonable. Service was a bit slow but friendly. Would go back if in the area but wouldn't go out of my way for it.",
        "rating": 3,
        "type": "Neutral"
    }
]

# ============================================================================
# POST-PROCESSING
# ============================================================================

def clean_summary(text):
    """Clean up model output by removing format markers and fixing truncation"""
    
    # Remove common format markers
    markers_to_remove = [
        r'Review:\s*',
        r'Summary:\s*',
        r'Task:\s*',
        r'Provide:\s*',
        r'\[/?\|user\|\]',
        r'\[/?\|assistant\|\]',
        r'\*/\|user\|\*/',
        r'\*\*/\|assistant\|\*\*',
        r'Positives?:\s*',
        r'Negatives?:\s*',
        r'Key points?:\s*',
        r'Sentiment:\s*',
        r'Overall:\s*',
        r'Generate.*?summary.*?\.',
    ]
    
    cleaned = text
    for marker in markers_to_remove:
        cleaned = re.sub(marker, '', cleaned, flags=re.IGNORECASE)
    
    # Remove bullet points and dashes at start
    cleaned = re.sub(r'^\s*[-•]\s*', '', cleaned, flags=re.MULTILINE)
    
    # Remove excessive newlines
    cleaned = re.sub(r'\n\s*\n', ' ', cleaned)
    
    # Remove quotes at start/end if present
    cleaned = cleaned.strip('"\'')
    
    # If text ends mid-word or sentence, try to complete or truncate properly
    if cleaned and not cleaned[-1] in '.!?':
        # Find last complete sentence
        last_period = max(cleaned.rfind('.'), cleaned.rfind('!'), cleaned.rfind('?'))
        if last_period > len(cleaned) * 0.5:  # If at least halfway through
            cleaned = cleaned[:last_period + 1]
        else:
            # Add period if reasonable length
            if len(cleaned.split()) > 10:
                cleaned = cleaned.rstrip(',;: ') + '.'
    
    # Clean up multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    return cleaned.strip()

# ============================================================================
# LOAD MODEL
# ============================================================================

def load_model(model_name=BEST_MODEL):
    """Load the fine-tuned model"""
    
    print("\n" + "="*70)
    print("REVIEWSENSE AI - INTERACTIVE DEMO")
    print("="*70)
    print(f"\n📊 Model Performance:")
    print(f"   • ROUGE-1: 0.5079 (54.8% improvement over baseline)")
    print(f"   • ROUGE-2: 0.4724")
    print(f"   • ROUGE-L: 0.4942")
    print(f"   • BLEU: 0.2940")
    print(f"\n📥 Loading {model_name} (best model)...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=DEVICE,
        trust_remote_code=True
    )
    
    # Load LoRA adapter
    adapter_path = MODELS_DIR / model_name
    if adapter_path.exists():
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print(f"✅ Model loaded successfully!")
    else:
        print(f"⚠️  Adapter not found at {adapter_path}, using base model")
    
    model.eval()
    print()
    
    return model, tokenizer

# ============================================================================
# GENERATE SUMMARY
# ============================================================================

def generate_summary(model, tokenizer, review_text, temperature=0.7, max_tokens=200):
    """Generate summary for a review with post-processing"""
    
    prompt = f"""<|system|>
You are a helpful assistant that analyzes customer reviews and generates summaries.
<|user|>
Review: {review_text}

Task: Generate a concise summary of this review highlighting key points and sentiment.
<|assistant|>
"""
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant's response
    if "<|assistant|>" in generated:
        summary = generated.split("<|assistant|>")[-1].strip()
    else:
        summary = generated[len(prompt):].strip()
    
    # Clean up the summary
    summary = clean_summary(summary)
    
    return summary

# ============================================================================
# INTERACTIVE DEMO
# ============================================================================

def display_menu():
    """Display menu options"""
    print("\n" + "="*70)
    print("REVIEWSENSE AI - DEMO MENU")
    print("="*70)
    print("\nOptions:")
    print("  1. Try sample review #1 (⭐⭐⭐⭐⭐ Positive)")
    print("  2. Try sample review #2 (⭐⭐ Negative)")
    print("  3. Try sample review #3 (⭐⭐⭐ Neutral)")
    print("  4. Enter custom review")
    print("  5. Exit")
    print("\n" + "="*70)

def run_demo():
    """Run interactive demo"""
    
    # Load model
    model, tokenizer = load_model(BEST_MODEL)
    
    while True:
        display_menu()
        
        choice = input("\nSelect an option (1-5): ").strip()
        
        if choice == "5":
            print("\n" + "="*70)
            print("Thank you for trying ReviewSense AI!")
            print("Model: exp2 | Performance: 54.8% improvement over baseline")
            print("="*70)
            break
        
        # Get review text
        if choice in ["1", "2", "3"]:
            sample_idx = int(choice) - 1
            review = SAMPLE_REVIEWS[sample_idx]
            review_text = review['text']
            rating = review['rating']
            review_type = review['type']
            
            print(f"\n{'='*70}")
            print(f"SAMPLE REVIEW #{choice} - {review_type.upper()}")
            print(f"{'='*70}")
            print(f"\n📝 Original Review:")
            print(f"   {review_text}")
            print(f"\n⭐ Rating: {rating}/5")
            
        elif choice == "4":
            print("\n" + "="*70)
            print("CUSTOM REVIEW INPUT")
            print("="*70)
            print("\n📝 Enter your review (press Enter twice when done):")
            lines = []
            empty_count = 0
            while empty_count < 2:
                line = input()
                if line == "":
                    empty_count += 1
                else:
                    empty_count = 0
                    lines.append(line)
            
            review_text = " ".join(lines)
            rating = None
            review_type = "Custom"
            
            if not review_text:
                print("❌ No review entered. Try again.")
                continue
                
            print(f"\n📝 Your Review:")
            print(f"   {review_text[:200]}{'...' if len(review_text) > 200 else ''}")
            
        else:
            print("❌ Invalid option. Please select 1-5.")
            continue
        
        # Generate summary
        print(f"\n🔮 Generating summary...")
        summary = generate_summary(model, tokenizer, review_text)
        
        print(f"\n{'='*70}")
        print("✨ GENERATED SUMMARY")
        print(f"{'='*70}")
        print(f"\n{summary}\n")
        print(f"{'='*70}")
        
        # Ask to continue
        continue_demo = input("\nTry another review? (y/n): ").strip().lower()
        if continue_demo != 'y':
            print("\n" + "="*70)
            print("Thank you for trying ReviewSense AI!")
            print("Model: exp2 | Performance: 54.8% improvement over baseline")
            print("="*70)
            break

# ============================================================================
# BATCH DEMO (for screenshot/video)
# ============================================================================

def run_batch_demo():
    """Run batch demo of all samples (for screenshot/video recording)"""
    
    print("\n" + "="*70)
    print("REVIEWSENSE AI - BATCH DEMO")
    print("="*70)
    print("\n🎯 Demonstrating Fine-Tuned Model Performance")
    print(f"   Model: exp2 (LR=1e-4, LoRA rank=8)")
    print(f"   Achievement: 54.8% improvement over baseline")
    print(f"   ROUGE-1: 0.5079 | BLEU: 0.2940")
    
    # Load model
    model, tokenizer = load_model(BEST_MODEL)
    
    for idx, review in enumerate(SAMPLE_REVIEWS, 1):
        print(f"\n{'='*70}")
        print(f"EXAMPLE {idx}: {review['type'].upper()} REVIEW")
        print(f"{'='*70}")
        print(f"\n📝 Original Review:")
        print(f"   {review['text']}")
        print(f"\n⭐ Rating: {review['rating']}/5")
        print(f"\n🔮 Generating summary...")
        
        summary = generate_summary(model, tokenizer, review['text'])
        
        print(f"\n✨ Generated Summary:")
        print(f"   {summary}")
    
    print(f"\n{'='*70}")
    print("✅ DEMO COMPLETE")
    print("="*70)
    print("\n📊 Key Achievements:")
    print("   ✓ Successfully fine-tuned TinyLlama-1.1B")
    print("   ✓ 54.8% improvement in ROUGE-1 score")
    print("   ✓ Handles positive, negative, and neutral reviews")
    print("   ✓ Generates concise, accurate summaries")
    print(f"\n{'='*70}\n")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function"""
    
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        run_batch_demo()
    else:
        run_demo()

if __name__ == "__main__":
    main()