# Reproduction Instructions

**Step-by-Step Guide to Reproduce ReviewSense AI Results**

This document provides clear instructions to reproduce all experimental results from the ReviewSense AI project.

---

## Prerequisites

Before starting, ensure you have completed:
- ✅ **Environment Setup** (see ENVIRONMENT_SETUP.md)
- ✅ **Virtual environment activated**: Run `source venv/bin/activate`
- ✅ **All dependencies installed**

---

## Overview

**What You Will Reproduce:**
1. Model evaluation on 200 test samples
2. Performance metrics (ROUGE, BLEU)
3. Error analysis
4. Interactive demo

**What You Will NOT Reproduce** (already completed):
- Data collection (60K reviews collected)
- Data preprocessing (33K cleaned reviews)
- Model training (6 GPU-hours on Google Colab)

**Why**: Training requires GPU access and significant time. Pre-trained models are provided.

---

## Step 1: Verify Project Structure

Ensure your directory looks like this:

```bash
cd ~/Documents/INFO_7375/Assigment-FineTuningLLM/ReviewSense-AI/

# Check structure
ls -la
```

**Required directories:**
```
ReviewSense-AI/
├── venv/                      # Virtual environment
├── all_training_results/
│   └── models/
│       ├── exp1/              # Must contain adapter files
│       ├── exp2/              # Must contain adapter files
│       └── exp3/              # Must contain adapter files
├── data/
│   └── splits/
│       ├── train.jsonl        # Training data
│       ├── val.jsonl          # Validation data
│       └── test.jsonl         # Test data
├── checkpoint_model_evaluation.py
├── error_analysis.py
├── demo_app.py
└── requirements.txt
```

---

## Step 2: Verify Model Files

Check that trained models are present:

```bash
# Verify exp1
ls -lh all_training_results/models/exp1/
# Should see: adapter_config.json, adapter_model.safetensors

# Verify exp2 (best model)
ls -lh all_training_results/models/exp2/
# Should see: adapter_config.json, adapter_model.safetensors

# Verify exp3
ls -lh all_training_results/models/exp3/
# Should see: adapter_config.json, adapter_model.safetensors
```

**If files are missing:**
Models must be extracted from training results. Contact project maintainer for model files.

---

## Step 3: Run Model Evaluation

### 3.1 Activate Environment

```bash
# Ensure virtual environment is active
source venv/bin/activate

# Verify
which python  # Should show path with 'venv'
```

### 3.2 Run Evaluation Script

```bash
# Run comprehensive evaluation
python checkpoint_model_evaluation.py
```

**What This Does:**
1. Loads baseline TinyLlama (no fine-tuning)
2. Loads exp1, exp2, exp3 (fine-tuned models)
3. Generates predictions on 200 test reviews
4. Calculates ROUGE-1, ROUGE-2, ROUGE-L, BLEU metrics
5. Compares all models
6. Saves results to `evaluation_results/`

**Expected Duration:**
- **Total Time**: ~60-70 minutes
- Baseline: ~17 minutes (200 samples)
- exp1: ~15 minutes
- exp2: ~15 minutes
- exp3: ~15 minutes

**Progress Indicators:**
```
======================================================================
EVALUATING BASELINE MODEL
======================================================================
📥 Loading baseline model...
✅ Model loaded successfully!

🔮 Generating predictions for baseline...
Generating: 100%|███████████| 200/200 [17:23<00:00]
✅ Generated 200 predictions

📈 Calculating evaluation metrics...
📊 Metrics Summary:
   rouge1: 0.3281
   rouge2: 0.1632
   rougeL: 0.2599
   bleu: 0.1025

======================================================================
EVALUATING EXP1
======================================================================
[... continues for exp1, exp2, exp3 ...]

✅ EVALUATION COMPLETE!
```

### 3.3 Verify Results

```bash
# Check output files were created
ls -lh evaluation_results/

# Should see:
# - evaluation_report.md
# - model_comparison.png
# - evaluation_results.json
```

**View the report:**
```bash
# Open report in terminal
cat evaluation_results/evaluation_report.md | head -100

# Or open with text editor
open evaluation_results/evaluation_report.md  # Mac
nano evaluation_results/evaluation_report.md  # Linux
notepad evaluation_results/evaluation_report.md  # Windows
```

### 3.4 Expected Results

**Performance Metrics:**

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU |
|-------|---------|---------|---------|------|
| baseline | 0.3281 ±0.02 | 0.1632 ±0.02 | 0.2599 ±0.02 | 0.1025 ±0.01 |
| exp1 | 0.4767 ±0.02 | 0.4342 ±0.02 | 0.4635 ±0.02 | 0.2682 ±0.01 |
| **exp2** | **0.5079 ±0.02** | **0.4724 ±0.02** | **0.4942 ±0.02** | **0.2940 ±0.01** |
| exp3 | 0.4715 ±0.02 | 0.4177 ±0.02 | 0.4517 ±0.02 | 0.2607 ±0.01 |

**Note**: Small variations (±2%) are normal due to randomness in generation.

**Best Model**: exp2 (LR=1e-4, LoRA rank=8)

**Improvement**: +54.8% ROUGE-1 over baseline

---

## Step 4: Run Error Analysis

### 4.1 Execute Error Analysis

```bash
# Run error analysis on best model (exp2)
python error_analysis.py
```

**What This Does:**
1. Loads exp2 predictions from evaluation
2. Categorizes errors (truncation, format issues, etc.)
3. Identifies 15 worst-performing cases
4. Creates error distribution visualization
5. Generates comprehensive error report

**Expected Duration:** ~30 seconds (uses cached predictions)

**Expected Output:**
```
======================================================================
REVIEWSENSE AI - ERROR ANALYSIS (Phase 6)
======================================================================

Analyzing: exp2 (Best performing model)

======================================================================
LOADING PREDICTIONS FOR EXP2
======================================================================
✅ Loaded 200 predictions

Model Performance:
  ROUGE-1: 0.5079
  ROUGE-2: 0.4724
  ROUGE-L: 0.4942
  BLEU: 0.2940

======================================================================
ANALYZING ERROR PATTERNS
======================================================================

📊 Error Distribution:
  Truncated: 160 cases (80.0%)
  Verbatim Copy: 140 cases (70.0%)
  Repetitive: 59 cases (29.5%)
  Format Issues: 200 cases (100.0%)

✅ Well-Performing: 0 cases (0.0%)

======================================================================
IDENTIFYING FAILURE CASES
======================================================================
📋 Selected 15 worst cases for detailed analysis

📊 Creating visualizations...
✅ Saved: error_analysis_results/error_distribution.png

📝 Generating evaluation report...
✅ Saved: error_analysis_results/error_analysis_report.md
✅ Saved: error_analysis_results/error_analysis.json

======================================================================
✅ ERROR ANALYSIS COMPLETE!
======================================================================
```

### 4.2 Verify Error Analysis Results

```bash
# Check files were created
ls -lh error_analysis_results/

# Should see:
# - error_analysis_report.md
# - error_distribution.png
# - error_analysis.json

# View the report
cat error_analysis_results/error_analysis_report.md | head -50
```

### 4.3 Expected Error Patterns

**Error Distribution:**
- **Format Leakage**: 100% (all predictions contain template markers)
- **Truncation**: 80% (predictions cut off mid-sentence)
- **Verbatim Copying**: 70% (model copies instead of summarizing)
- **Repetition**: 29.5% (excessive word repetition)

**Key Insight**: Model learned the task (understands reviews) but has output formatting challenges.

---

## Step 5: Run Interactive Demo

### 5.1 Test Batch Demo (Fastest)

```bash
# Run all 3 examples automatically
python demo_app.py --batch
```

**What This Shows:**
- Example 1: Positive 5-star review
- Example 2: Negative 2-star review
- Example 3: Neutral 3-star review

**Expected Duration:** ~30 seconds

**Expected Output:**
```
======================================================================
REVIEWSENSE AI - BATCH DEMO
======================================================================

🎯 Demonstrating Fine-Tuned Model Performance
   Model: exp2 (LR=1e-4, LoRA rank=8)
   Achievement: 54.8% improvement over baseline
   ROUGE-1: 0.5079 | BLEU: 0.2940

📥 Loading exp2 (best model)...
✅ Model loaded successfully!

======================================================================
EXAMPLE 1: POSITIVE REVIEW
======================================================================

📝 Original Review:
   This restaurant was absolutely amazing! The food was delicious...

⭐ Rating: 5/5

🔮 Generating summary...

✨ Generated Summary:
   [Model-generated summary appears here]

[... continues for examples 2 and 3 ...]

======================================================================
✅ DEMO COMPLETE
======================================================================
```

### 5.2 Test Interactive Demo (Optional)

```bash
# Run interactive mode
python demo_app.py
```

**Follow prompts:**
1. Select option `1` (Positive review)
2. When asked "Try another? (y/n)", type `y`
3. Select option `3` (Neutral review)
4. Type `y` again
5. Select option `4` (Custom input)
6. Paste this sample review:
   ```
   Great coffee shop with excellent espresso. Service was friendly but slow. 
   Prices are reasonable. Would visit again.
   ```
7. Press Enter twice
8. Type `n` to exit

**Expected Duration:** 2-3 minutes

---

## Step 6: Verify Complete Reproduction

### 6.1 Check All Output Files

```bash
# Evaluation results
ls -lh evaluation_results/
# Should see: evaluation_report.md, model_comparison.png, evaluation_results.json

# Error analysis
ls -lh error_analysis_results/
# Should see: error_analysis_report.md, error_distribution.png, error_analysis.json

# Checkpoints (cached predictions)
ls -lh evaluation_checkpoints/
# Should see: baseline_checkpoint.pkl, exp1_checkpoint.pkl, exp2_checkpoint.pkl, exp3_checkpoint.pkl
```

### 6.2 Verification Checklist

Confirm all steps completed:

- [ ] Environment activated
- [ ] Evaluation ran successfully (~70 minutes)
- [ ] All 4 models evaluated (baseline + exp1 + exp2 + exp3)
- [ ] Results match expected metrics (±2%)
- [ ] Error analysis completed
- [ ] Error reports generated
- [ ] Demo runs without errors
- [ ] All output files present

---

## Troubleshooting

### Issue: "ModuleNotFoundError"

**Cause**: Dependency not installed or virtual environment not activated

**Solution:**
```bash
# Activate environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: "Model files not found"

**Cause**: Models not in correct directory

**Solution:**
```bash
# Verify model paths
ls -R all_training_results/models/

# Should see exp1/, exp2/, exp3/ directories
# Each should contain: adapter_config.json, adapter_model.safetensors
```

### Issue: Evaluation very slow

**Cause**: Normal on CPU-only systems

**Solution:**
- Expected: 60-70 minutes total
- Faster on machines with GPU/MPS acceleration
- Can reduce samples in code if needed (not recommended for official reproduction)

### Issue: "Out of memory"

**Cause**: Insufficient RAM

**Solution:**
```bash
# Close other applications
# Restart evaluation

# If persists, reduce evaluation samples:
# In checkpoint_model_evaluation.py, change:
# MAX_EVAL_SAMPLES = 200  # to  100
```

### Issue: Demo fails to load model

**Cause**: Model path incorrect

**Solution:**
```bash
# Verify you're in project root
pwd  # Should end with /ReviewSense-AI

# Check model exists
ls all_training_results/models/exp2/
```

---

## Expected Outputs Summary

**After completing all steps, you should have:**

1. **Evaluation Results**
   - Metrics table comparing 4 models
   - Bar charts showing performance
   - JSON file with detailed results

2. **Error Analysis**
   - Error pattern identification
   - Failure case examples
   - Distribution visualization
   - Recommendations for improvement

3. **Demo Output**
   - Working inference pipeline
   - Example summaries for 3 review types
   - Proof of functional model

---

## Time Required

| Step | Duration |
|------|----------|
| Environment verification | 2 minutes |
| Model file verification | 2 minutes |
| Run evaluation | 60-70 minutes |
| Run error analysis | 1 minute |
| Test demo | 5 minutes |
| **Total** | **~1.5 hours** |

**Note**: Most time is spent on evaluation (model inference). This is automated and requires no interaction.

---

## Success Criteria

**Your reproduction is successful if:**

1. ✅ exp2 achieves ROUGE-1 ≥ 0.49 (within 2% of 0.5079)
2. ✅ exp2 is the best-performing model
3. ✅ Error analysis identifies 100% format issues
4. ✅ Demo generates summaries without crashing
5. ✅ All output files are created

**Small variations (±2%) in metrics are normal** due to:
- Hardware differences (CPU vs GPU/MPS)
- PyTorch version differences
- Random sampling in generation

---

## Next Steps After Reproduction

1. 📖 **Read**: `TECHNICAL_REPORT.md` for complete methodology
2. 📊 **Review**: Evaluation and error analysis reports
3. 🎬 **Watch**: Video walkthrough (if available)
4. 💡 **Understand**: Why exp2 outperforms other configurations

---

## Contact for Issues

If you encounter problems during reproduction:

1. Check Troubleshooting section above
2. Verify environment setup (ENVIRONMENT_SETUP.md)
3. Ensure all prerequisite steps completed
4. Check that virtual environment is activated

---

**Document Version**: 1.0  
**Last Updated**: October 2025  
**Maintained By**: Sravan Kumar Kurapati  
**Estimated Reproduction Time**: 1.5 hours
