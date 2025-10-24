"""
ReviewSense AI - Comprehensive Model Evaluation Script
Evaluates fine-tuned TinyLlama models with ROUGE, BLEU, and custom metrics
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel
from datasets import load_dataset
from evaluate import load

# ============================================================================
# CONFIGURATION
# ============================================================================

class EvaluationConfig:
    """Configuration for model evaluation"""
    
    # Paths
    BASE_DIR = Path.home() / "Documents/INFO_7375/Assigment-FineTuningLLM/ReviewSense-AI"
    MODELS_DIR = BASE_DIR / "models"
    RESULTS_DIR = BASE_DIR / "evaluation_results"
    
    # Model configurations
    BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    EXPERIMENTS = {
        "exp1": {"lr": "2e-4", "r": 8, "description": "LR=2e-4, r=8"},
        "exp2": {"lr": "1e-4", "r": 8, "description": "LR=1e-4, r=8"},
        "exp3": {"lr": "2e-4", "r": 16, "description": "LR=2e-4, r=16"}
    }
    
    # Evaluation parameters
    EVAL_BATCH_SIZE = 8
    MAX_EVAL_SAMPLES = 500  # Subset for thorough evaluation
    MAX_NEW_TOKENS = 150
    TEMPERATURE = 0.7
    TOP_P = 0.9
    
    # Metrics
    METRICS = ["rouge", "bleu", "meteor"]
    
    # Device
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ============================================================================
# STEP 1: EXTRACT MODELS
# ============================================================================

def extract_models():
    """Extract models from zip file if not already extracted"""
    
    print("\n" + "="*70)
    print("STEP 1: EXTRACTING MODELS")
    print("="*70)
    
    zip_path = EvaluationConfig.BASE_DIR / "all_training_results.zip"
    
    if not zip_path.exists():
        print(f"❌ Zip file not found: {zip_path}")
        return False
    
    if EvaluationConfig.MODELS_DIR.exists():
        print(f"✅ Models directory already exists: {EvaluationConfig.MODELS_DIR}")
        
        # List extracted models
        for exp_name in EvaluationConfig.EXPERIMENTS.keys():
            exp_dir = EvaluationConfig.MODELS_DIR / exp_name
            if exp_dir.exists():
                print(f"   Found: {exp_name}/")
        return True
    
    print(f"📦 Extracting from: {zip_path}")
    
    import zipfile
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(EvaluationConfig.MODELS_DIR)
        print(f"✅ Extraction complete!")
        return True
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False


# ============================================================================
# STEP 2: LOAD MODELS
# ============================================================================

def load_model_and_tokenizer(experiment_name: str = None, use_baseline: bool = False):
    """
    Load model and tokenizer
    
    Args:
        experiment_name: Name of experiment (exp1, exp2, exp3) or None for baseline
        use_baseline: If True, load base model without LoRA
    """
    
    print(f"\n📥 Loading {'baseline' if use_baseline else experiment_name} model...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        EvaluationConfig.BASE_MODEL_NAME,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        EvaluationConfig.BASE_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=EvaluationConfig.DEVICE,
        trust_remote_code=True
    )
    
    # Load LoRA weights if not baseline
    if not use_baseline and experiment_name:
        adapter_path = EvaluationConfig.MODELS_DIR / experiment_name
        
        if not adapter_path.exists():
            print(f"❌ Adapter not found: {adapter_path}")
            return None, None
        
        print(f"   Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()  # Merge for faster inference
    
    model.eval()
    print(f"✅ Model loaded successfully!")
    
    return model, tokenizer


# ============================================================================
# STEP 3: PREPARE EVALUATION DATA
# ============================================================================

def prepare_eval_data(num_samples: int = 500):
    """Prepare evaluation dataset"""
    
    print(f"\n📊 Preparing evaluation data ({num_samples} samples)...")
    
    # Load test split from Yelp dataset
    dataset = load_dataset("yelp_review_full", split="test")
    
    # Take subset for evaluation
    eval_dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    print(f"✅ Loaded {len(eval_dataset)} samples for evaluation")
    
    return eval_dataset


# ============================================================================
# STEP 4: GENERATE PREDICTIONS
# ============================================================================

def generate_predictions(
    model,
    tokenizer,
    eval_data,
    experiment_name: str
) -> List[Dict]:
    """Generate predictions for evaluation data"""
    
    print(f"\n🔮 Generating predictions for {experiment_name}...")
    
    predictions = []
    
    for idx in tqdm(range(len(eval_data)), desc="Generating"):
        sample = eval_data[idx]
        
        # Create prompt
        prompt = f"""<|system|>
You are a helpful assistant that analyzes customer reviews and generates summaries.
<|user|>
Review: {sample['text'][:500]}...

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
                max_new_tokens=EvaluationConfig.MAX_NEW_TOKENS,
                temperature=EvaluationConfig.TEMPERATURE,
                top_p=EvaluationConfig.TOP_P,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "<|assistant|>" in generated:
            prediction = generated.split("<|assistant|>")[-1].strip()
        else:
            prediction = generated[len(prompt):].strip()
        
        predictions.append({
            "review_id": idx,
            "original_text": sample['text'],
            "rating": sample['label'] + 1,  # Convert 0-4 to 1-5
            "prediction": prediction,
            "reference": sample['text'][:200]  # First 200 chars as reference
        })
    
    print(f"✅ Generated {len(predictions)} predictions")
    
    return predictions


# ============================================================================
# STEP 5: CALCULATE METRICS
# ============================================================================

def calculate_metrics(predictions: List[Dict]) -> Dict:
    """Calculate ROUGE, BLEU, and METEOR scores"""
    
    print(f"\n📈 Calculating evaluation metrics...")
    
    # Load metrics
    rouge = load("rouge")
    bleu = load("bleu")
    meteor = load("meteor")
    
    # Prepare predictions and references
    preds = [p['prediction'] for p in predictions]
    refs = [p['reference'] for p in predictions]
    
    # Calculate ROUGE scores
    rouge_scores = rouge.compute(
        predictions=preds,
        references=refs,
        use_aggregator=True
    )
    
    # Calculate BLEU score
    # BLEU expects references as list of lists
    bleu_score = bleu.compute(
        predictions=preds,
        references=[[r] for r in refs]
    )
    
    # Calculate METEOR score
    meteor_score = meteor.compute(
        predictions=preds,
        references=refs
    )
    
    metrics = {
        "rouge1": rouge_scores['rouge1'],
        "rouge2": rouge_scores['rouge2'],
        "rougeL": rouge_scores['rougeL'],
        "bleu": bleu_score['bleu'],
        "meteor": meteor_score['meteor']
    }
    
    print("\n📊 Metrics Summary:")
    for metric_name, value in metrics.items():
        print(f"   {metric_name}: {value:.4f}")
    
    return metrics


# ============================================================================
# STEP 6: COMPARE MODELS
# ============================================================================

def evaluate_all_models():
    """Evaluate all models and compare results"""
    
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*70)
    
    # Create results directory
    EvaluationConfig.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Prepare evaluation data
    eval_data = prepare_eval_data(EvaluationConfig.MAX_EVAL_SAMPLES)
    
    all_results = {}
    
    # Evaluate baseline model
    print("\n" + "-"*70)
    print("EVALUATING BASELINE MODEL")
    print("-"*70)
    
    baseline_model, baseline_tokenizer = load_model_and_tokenizer(use_baseline=True)
    if baseline_model:
        baseline_preds = generate_predictions(
            baseline_model, 
            baseline_tokenizer, 
            eval_data, 
            "baseline"
        )
        baseline_metrics = calculate_metrics(baseline_preds)
        all_results["baseline"] = {
            "metrics": baseline_metrics,
            "predictions": baseline_preds
        }
        
        # Clean up
        del baseline_model, baseline_tokenizer
        torch.mps.empty_cache() if EvaluationConfig.DEVICE == "mps" else None
    
    # Evaluate each fine-tuned model
    for exp_name, exp_config in EvaluationConfig.EXPERIMENTS.items():
        print("\n" + "-"*70)
        print(f"EVALUATING {exp_name.upper()}: {exp_config['description']}")
        print("-"*70)
        
        model, tokenizer = load_model_and_tokenizer(exp_name)
        if model:
            predictions = generate_predictions(model, tokenizer, eval_data, exp_name)
            metrics = calculate_metrics(predictions)
            
            all_results[exp_name] = {
                "config": exp_config,
                "metrics": metrics,
                "predictions": predictions
            }
            
            # Clean up
            del model, tokenizer
            torch.mps.empty_cache() if EvaluationConfig.DEVICE == "mps" else None
    
    return all_results, eval_data


# ============================================================================
# STEP 7: VISUALIZATION AND REPORTING
# ============================================================================

def create_comparison_plots(results: Dict):
    """Create visualization plots comparing all models"""
    
    print("\n📊 Creating comparison visualizations...")
    
    # Extract metrics for plotting
    model_names = list(results.keys())
    metrics_to_plot = ["rouge1", "rouge2", "rougeL", "bleu", "meteor"]
    
    data = {metric: [] for metric in metrics_to_plot}
    
    for model_name in model_names:
        for metric in metrics_to_plot:
            data[metric].append(results[model_name]['metrics'][metric])
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        bars = ax.bar(model_names, data[metric], color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax.set_title(metric.upper(), fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Remove extra subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    
    plot_path = EvaluationConfig.RESULTS_DIR / "model_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved comparison plot: {plot_path}")
    
    plt.close()


def generate_evaluation_report(results: Dict, eval_data):
    """Generate comprehensive evaluation report"""
    
    print("\n📝 Generating evaluation report...")
    
    report_path = EvaluationConfig.RESULTS_DIR / "evaluation_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# ReviewSense AI - Model Evaluation Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Evaluation Samples**: {len(eval_data)}\n\n")
        
        f.write("## Overview\n\n")
        f.write("This report presents a comprehensive evaluation of fine-tuned TinyLlama models ")
        f.write("for review summarization tasks.\n\n")
        
        # Model configurations
        f.write("## Model Configurations\n\n")
        f.write("| Model | Learning Rate | LoRA Rank | Description |\n")
        f.write("|-------|--------------|-----------|-------------|\n")
        f.write("| Baseline | - | - | Base TinyLlama without fine-tuning |\n")
        for exp_name, result in results.items():
            if exp_name != "baseline" and 'config' in result:
                config = result['config']
                f.write(f"| {exp_name} | {config['lr']} | {config['r']} | {config['description']} |\n")
        f.write("\n")
        
        # Performance comparison
        f.write("## Performance Metrics\n\n")
        f.write("| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | METEOR |\n")
        f.write("|-------|---------|---------|---------|------|--------|\n")
        
        for model_name, result in results.items():
            metrics = result['metrics']
            f.write(f"| {model_name} | {metrics['rouge1']:.4f} | {metrics['rouge2']:.4f} | ")
            f.write(f"{metrics['rougeL']:.4f} | {metrics['bleu']:.4f} | {metrics['meteor']:.4f} |\n")
        f.write("\n")
        
        # Best model identification
        f.write("## Best Performing Model\n\n")
        
        # Find best for each metric
        best_models = {}
        for metric in ['rouge1', 'rouge2', 'rougeL', 'bleu', 'meteor']:
            best_model = max(results.items(), key=lambda x: x[1]['metrics'][metric])
            best_models[metric] = (best_model[0], best_model[1]['metrics'][metric])
        
        for metric, (model_name, score) in best_models.items():
            f.write(f"- **{metric.upper()}**: {model_name} ({score:.4f})\n")
        f.write("\n")
        
        # Sample predictions
        f.write("## Sample Predictions\n\n")
        baseline_preds = results['baseline']['predictions'][:3]
        
        for idx, pred in enumerate(baseline_preds):
            f.write(f"### Example {idx + 1}\n\n")
            f.write(f"**Original Review (truncated)**:\n")
            f.write(f"> {pred['original_text'][:300]}...\n\n")
            f.write(f"**Rating**: {pred['rating']}/5\n\n")
            
            f.write("**Model Outputs**:\n\n")
            for model_name in results.keys():
                model_pred = results[model_name]['predictions'][idx]['prediction']
                f.write(f"- **{model_name}**: {model_pred}\n\n")
        
        # Conclusions
        f.write("## Key Findings\n\n")
        f.write("1. **Fine-tuning Impact**: Compare baseline vs fine-tuned models\n")
        f.write("2. **Hyperparameter Effects**: Analyze impact of learning rate and LoRA rank\n")
        f.write("3. **Best Configuration**: Identify optimal settings for production\n\n")
    
    print(f"✅ Saved evaluation report: {report_path}")
    
    # Save detailed results as JSON
    json_path = EvaluationConfig.RESULTS_DIR / "evaluation_results.json"
    
    # Prepare JSON-serializable results
    json_results = {}
    for model_name, result in results.items():
        json_results[model_name] = {
            "config": result.get('config', {}),
            "metrics": result['metrics'],
            "sample_predictions": result['predictions'][:5]  # Save first 5
        }
    
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"✅ Saved detailed results: {json_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main evaluation pipeline"""
    
    print("\n" + "="*70)
    print("REVIEWSENSE AI - MODEL EVALUATION PIPELINE")
    print("="*70)
    
    # Step 1: Extract models
    if not extract_models():
        print("\n❌ Failed to extract models. Please check the zip file.")
        return
    
    # Step 2-6: Evaluate all models
    results, eval_data = evaluate_all_models()
    
    # Step 7: Create visualizations and reports
    create_comparison_plots(results)
    generate_evaluation_report(results, eval_data)
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {EvaluationConfig.RESULTS_DIR}")
    print("\nGenerated files:")
    print("  📊 model_comparison.png - Visual comparison of all models")
    print("  📝 evaluation_report.md - Comprehensive markdown report")
    print("  💾 evaluation_results.json - Detailed results in JSON format")
    print("\n🎯 Next steps:")
    print("  1. Review the evaluation report")
    print("  2. Identify best performing model")
    print("  3. Proceed to Error Analysis (Phase 6)")


if __name__ == "__main__":
    main()