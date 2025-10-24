"""
ReviewSense AI - Checkpoint-Enabled Model Evaluation
Saves progress after each model and can resume from where it left off
"""

import os
import json
import torch
import pickle
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
    MODELS_DIR = BASE_DIR / "all_training_results" / "models"  # Fixed path
    RESULTS_DIR = BASE_DIR / "evaluation_results"
    CHECKPOINT_DIR = BASE_DIR / "evaluation_checkpoints"
    
    # Model configurations
    BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    EXPERIMENTS = {
        "baseline": {"description": "Base TinyLlama without fine-tuning"},
        "exp1": {"lr": "2e-4", "r": 8, "description": "LR=2e-4, r=8"},
        "exp2": {"lr": "1e-4", "r": 8, "description": "LR=1e-4, r=8"},
        "exp3": {"lr": "2e-4", "r": 16, "description": "LR=2e-4, r=16"}
    }
    
    # Evaluation parameters
    EVAL_BATCH_SIZE = 8
    MAX_EVAL_SAMPLES = 200  # Balanced for thorough yet efficient evaluation
    MAX_NEW_TOKENS = 150
    TEMPERATURE = 0.7
    TOP_P = 0.9
    
    # Metrics
    METRICS = ["rouge", "bleu"]
    
    # Device
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def save_checkpoint(model_name: str, predictions: List[Dict], metrics: Dict):
    """Save checkpoint after each model evaluation"""
    
    EvaluationConfig.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    checkpoint_data = {
        "model_name": model_name,
        "predictions": predictions,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
        "num_samples": len(predictions)
    }
    
    checkpoint_path = EvaluationConfig.CHECKPOINT_DIR / f"{model_name}_checkpoint.pkl"
    
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    print(f"✅ Checkpoint saved: {checkpoint_path}")


def load_checkpoint(model_name: str) -> Dict:
    """Load checkpoint if exists"""
    
    checkpoint_path = EvaluationConfig.CHECKPOINT_DIR / f"{model_name}_checkpoint.pkl"
    
    if checkpoint_path.exists():
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        print(f"✅ Loaded checkpoint for {model_name} ({checkpoint_data['num_samples']} samples)")
        return checkpoint_data
    
    return None


def list_checkpoints() -> Dict[str, bool]:
    """List which models have checkpoints"""
    
    checkpoint_status = {}
    
    for model_name in EvaluationConfig.EXPERIMENTS.keys():
        checkpoint_path = EvaluationConfig.CHECKPOINT_DIR / f"{model_name}_checkpoint.pkl"
        checkpoint_status[model_name] = checkpoint_path.exists()
    
    return checkpoint_status


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model_and_tokenizer(experiment_name: str = None, use_baseline: bool = False):
    """Load model and tokenizer"""
    
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
    if not use_baseline and experiment_name and experiment_name != "baseline":
        adapter_path = EvaluationConfig.MODELS_DIR / experiment_name
        
        if not adapter_path.exists():
            print(f"❌ Adapter not found: {adapter_path}")
            return None, None
        
        print(f"   Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    
    model.eval()
    print(f"✅ Model loaded successfully!")
    
    return model, tokenizer


# ============================================================================
# EVALUATION DATA
# ============================================================================

def prepare_eval_data(num_samples: int = 200):
    """Prepare evaluation dataset"""
    
    print(f"\n📊 Preparing evaluation data ({num_samples} samples)...")
    
    # Load test split
    dataset = load_dataset("yelp_review_full", split="test")
    
    # Take subset
    eval_dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    print(f"✅ Loaded {len(eval_dataset)} samples for evaluation")
    
    return eval_dataset


# ============================================================================
# PREDICTION GENERATION
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
        
        # Extract assistant's response
        if "<|assistant|>" in generated:
            prediction = generated.split("<|assistant|>")[-1].strip()
        else:
            prediction = generated[len(prompt):].strip()
        
        predictions.append({
            "review_id": idx,
            "original_text": sample['text'],
            "rating": sample['label'] + 1,
            "prediction": prediction,
            "reference": sample['text'][:200]
        })
    
    print(f"✅ Generated {len(predictions)} predictions")
    
    return predictions


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_metrics(predictions: List[Dict]) -> Dict:
    """Calculate ROUGE and BLEU scores"""
    
    print(f"\n📈 Calculating evaluation metrics...")
    
    # Load metrics
    rouge = load("rouge")
    bleu = load("bleu")
    
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
    bleu_score = bleu.compute(
        predictions=preds,
        references=[[r] for r in refs]
    )
    
    metrics = {
        "rouge1": rouge_scores['rouge1'],
        "rouge2": rouge_scores['rouge2'],
        "rougeL": rouge_scores['rougeL'],
        "bleu": bleu_score['bleu']
    }
    
    print("\n📊 Metrics Summary:")
    for metric_name, value in metrics.items():
        print(f"   {metric_name}: {value:.4f}")
    
    return metrics


# ============================================================================
# MAIN EVALUATION WITH CHECKPOINTS
# ============================================================================

def evaluate_single_model(model_name: str, eval_data) -> Dict:
    """Evaluate a single model with checkpoint support"""
    
    # Check for existing checkpoint
    checkpoint = load_checkpoint(model_name)
    
    if checkpoint:
        print(f"✅ Using cached results for {model_name}")
        return {
            "metrics": checkpoint["metrics"],
            "predictions": checkpoint["predictions"]
        }
    
    print(f"\n{'='*70}")
    print(f"EVALUATING {model_name.upper()}")
    print(f"{'='*70}")
    
    # Load model
    if model_name == "baseline":
        model, tokenizer = load_model_and_tokenizer(use_baseline=True)
    else:
        model, tokenizer = load_model_and_tokenizer(model_name)
    
    if not model:
        print(f"❌ Failed to load {model_name}")
        return None
    
    # Generate predictions
    predictions = generate_predictions(model, tokenizer, eval_data, model_name)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions)
    
    # Save checkpoint
    save_checkpoint(model_name, predictions, metrics)
    
    # Cleanup
    del model, tokenizer
    if EvaluationConfig.DEVICE == "mps":
        torch.mps.empty_cache()
    
    return {
        "metrics": metrics,
        "predictions": predictions
    }


def evaluate_all_models():
    """Evaluate all models with checkpoint resume capability"""
    
    print("\n" + "="*70)
    print("CHECKPOINT-ENABLED MODEL EVALUATION")
    print("="*70)
    
    # Create directories
    EvaluationConfig.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EvaluationConfig.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check checkpoint status
    print("\n📋 Checkpoint Status:")
    checkpoint_status = list_checkpoints()
    for model_name, has_checkpoint in checkpoint_status.items():
        status = "✅ Complete" if has_checkpoint else "⏳ Pending"
        print(f"   {model_name}: {status}")
    
    # Prepare evaluation data
    eval_data = prepare_eval_data(EvaluationConfig.MAX_EVAL_SAMPLES)
    
    all_results = {}
    
    # Evaluate each model
    for model_name in EvaluationConfig.EXPERIMENTS.keys():
        result = evaluate_single_model(model_name, eval_data)
        
        if result:
            config = EvaluationConfig.EXPERIMENTS[model_name]
            all_results[model_name] = {
                "config": config,
                "metrics": result["metrics"],
                "predictions": result["predictions"]
            }
    
    return all_results, eval_data


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_comparison_plots(results: Dict):
    """Create visualization plots"""
    
    print("\n📊 Creating comparison visualizations...")
    
    model_names = list(results.keys())
    metrics_to_plot = ["rouge1", "rouge2", "rougeL", "bleu"]
    
    data = {metric: [] for metric in metrics_to_plot}
    
    for model_name in model_names:
        for metric in metrics_to_plot:
            data[metric].append(results[model_name]['metrics'][metric])
    
    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        bars = ax.bar(model_names, data[metric], 
                     color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax.set_title(metric.upper(), fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    plot_path = EvaluationConfig.RESULTS_DIR / "model_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {plot_path}")
    
    plt.close()


def generate_evaluation_report(results: Dict, eval_data):
    """Generate comprehensive report"""
    
    print("\n📝 Generating evaluation report...")
    
    report_path = EvaluationConfig.RESULTS_DIR / "evaluation_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# ReviewSense AI - Model Evaluation Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Evaluation Samples**: {len(eval_data)}\n\n")
        
        # Overview
        f.write("## Overview\n\n")
        f.write("Comprehensive evaluation of TinyLlama models for review summarization.\n\n")
        
        # Configurations
        f.write("## Model Configurations\n\n")
        f.write("| Model | Learning Rate | LoRA Rank | Description |\n")
        f.write("|-------|--------------|-----------|-------------|\n")
        
        for model_name, result in results.items():
            if 'config' in result:
                config = result['config']
                if model_name == "baseline":
                    f.write(f"| {model_name} | - | - | {config['description']} |\n")
                else:
                    f.write(f"| {model_name} | {config.get('lr', '-')} | {config.get('r', '-')} | {config['description']} |\n")
        f.write("\n")
        
        # Performance metrics
        f.write("## Performance Metrics\n\n")
        f.write("| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU |\n")
        f.write("|-------|---------|---------|---------|------|\n")
        
        for model_name, result in results.items():
            metrics = result['metrics']
            f.write(f"| {model_name} | {metrics['rouge1']:.4f} | {metrics['rouge2']:.4f} | ")
            f.write(f"{metrics['rougeL']:.4f} | {metrics['bleu']:.4f} |\n")
        f.write("\n")
        
        # Best models
        f.write("## Best Performing Model\n\n")
        
        best_models = {}
        for metric in ['rouge1', 'rouge2', 'rougeL', 'bleu']:
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
            f.write(f"**Original Review**:\n")
            f.write(f"> {pred['original_text'][:300]}...\n\n")
            f.write(f"**Rating**: {pred['rating']}/5\n\n")
            f.write("**Model Outputs**:\n\n")
            
            for model_name in results.keys():
                model_pred = results[model_name]['predictions'][idx]['prediction']
                f.write(f"- **{model_name}**: {model_pred}\n\n")
    
    print(f"✅ Saved: {report_path}")
    
    # Save JSON
    json_path = EvaluationConfig.RESULTS_DIR / "evaluation_results.json"
    json_results = {}
    
    for model_name, result in results.items():
        json_results[model_name] = {
            "config": result.get('config', {}),
            "metrics": result['metrics'],
            "sample_predictions": result['predictions'][:5]
        }
    
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"✅ Saved: {json_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main evaluation pipeline with checkpoint support"""
    
    print("\n" + "="*70)
    print("REVIEWSENSE AI - CHECKPOINT-ENABLED EVALUATION")
    print("="*70)
    
    # Evaluate all models (will resume from checkpoints)
    results, eval_data = evaluate_all_models()
    
    # Generate visualizations and reports
    create_comparison_plots(results)
    generate_evaluation_report(results, eval_data)
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nResults: {EvaluationConfig.RESULTS_DIR}")
    print(f"Checkpoints: {EvaluationConfig.CHECKPOINT_DIR}")
    print("\nFiles generated:")
    print("  📊 model_comparison.png")
    print("  📝 evaluation_report.md")
    print("  💾 evaluation_results.json")


if __name__ == "__main__":
    main()