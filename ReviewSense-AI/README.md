# ReviewSense AI - Fine-Tuning TinyLlama for Review Analysis

**Student**: Sravan Kumar Kurapati  
**Course**: INFO 7375 - Fine-Tuning Large Language Models  
**Institution**: Northeastern University  
**Date**: October 2025

---

## 🎯 Project Overview

ReviewSense AI demonstrates comprehensive fine-tuning of TinyLlama-1.1B-Chat for specialized review analysis tasks using LoRA (Low-Rank Adaptation). The project achieved **54.8% improvement** over baseline on ROUGE-1 metric through systematic hyperparameter optimization.

**Key Achievements:**
- ✅ 60,000 multi-source reviews collected and processed
- ✅ 3 experimental configurations with rigorous evaluation
- ✅ Best model: exp2 (LR=1e-4, rank=8) - ROUGE-1: 0.5079
- ✅ Comprehensive error analysis with actionable recommendations
- ✅ Production-ready inference pipeline

---

## 📁 Repository Structure

```
ReviewSense-AI/
├── data/                              # Dataset files
│   ├── raw/                          # Original collected data
│   ├── processed/                    # Cleaned and tokenized data
│   └── splits/                       # Train/validation/test splits
│       ├── train.jsonl               # 14,000 training examples
│       ├── val.jsonl                 # 3,000 validation examples
│       └── test.jsonl                # 3,000 test examples
│
├── scripts/                           # Data preparation scripts
│   ├── 01_collect_data.py            # Multi-source data collection
│   ├── 02_preprocess_data.py         # Text cleaning and filtering
│   ├── 03_create_training_data.py    # Format for instruction tuning
│   └── 04_tokenize_data.py           # Tokenization for training
│
├── all_training_results/             # Trained model artifacts
│   └── models/
│       ├── exp1/                     # LR=2e-4, rank=8
│       ├── exp2/                     # LR=1e-4, rank=8 (BEST)
│       └── exp3/                     # LR=2e-4, rank=16
│
├── evaluation_results/                # Evaluation outputs
│   ├── evaluation_report.md          # Detailed metrics comparison
│   ├── model_comparison.png          # Performance visualization
│   └── evaluation_results.json       # Structured results
│
├── error_analysis_results/            # Error analysis outputs
│   ├── error_analysis_report.md      # Failure mode analysis
│   ├── error_distribution.png        # Error type visualization
│   └── error_analysis.json           # Structured error data
│
├── evaluation_checkpoints/            # Cached predictions
│   ├── baseline_checkpoint.pkl
│   ├── exp1_checkpoint.pkl
│   ├── exp2_checkpoint.pkl
│   └── exp3_checkpoint.pkl
│
├── checkpoint_model_evaluation.py     # Main evaluation script
├── error_analysis.py                  # Error pattern analysis
├── demo_app.py                        # Interactive inference demo
├── requirements.txt                   # Python dependencies
├── config.py                          # Configuration settings
│
├── ENVIRONMENT_SETUP.md              # Environment configuration guide
├── REPRODUCTION_INSTRUCTIONS.md      # Step-by-step reproduction
├── CODE_DOCUMENTATION.md             # Detailed code reference
└── TECHNICAL_REPORT.md               # Complete methodology (5-7 pages)
```

---

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.10 or 3.11
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 20GB free space
- **OS**: macOS, Linux, or Windows with WSL2

### Installation (5 minutes)

```bash
# 1. Navigate to project directory
cd ReviewSense-AI

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# 5. Verify installation
python -c "import torch, transformers, peft; print('✅ All packages installed')"
```

**Detailed setup instructions**: See [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)

---

## 📊 Results Summary

### Performance Metrics

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | Improvement |
|-------|---------|---------|---------|------|-------------|
| Baseline | 0.3281 | 0.1632 | 0.2599 | 0.1025 | - |
| exp1 | 0.4767 | 0.4342 | 0.4635 | 0.2682 | +45.3% |
| **exp2** ⭐ | **0.5079** | **0.4724** | **0.4942** | **0.2940** | **+54.8%** |
| exp3 | 0.4715 | 0.4177 | 0.4517 | 0.2607 | +43.7% |

**Winner**: exp2 with Learning Rate=1e-4, LoRA rank=8

### Training Configuration

| Parameter | exp1 | exp2 (Best) | exp3 |
|-----------|------|-------------|------|
| Learning Rate | 2e-4 | **1e-4** | 2e-4 |
| LoRA Rank | 8 | **8** | 16 |
| Final Loss | 0.681 | **0.687** | 0.706 |
| Training Time | 2h | 2h | 2h |

---

## 🔬 Methodology

### 1. Dataset Preparation

**Sources:**
- Amazon Polarity: 30,000 product reviews
- Yelp Review Full: 20,000 service reviews  
- Synthetic: 10,000 aspect-annotated reviews

**Processing:**
```
60,000 raw reviews
    ↓ Text cleaning (97% retained)
58,209 valid reviews
    ↓ Quality filtering (98.2% retained)
58,914 quality reviews
    ↓ Deduplication & balancing
33,418 final reviews
    ↓ Split: 70/15/15
14,000 train | 3,000 val | 3,000 test
```

**Scripts:**
- `scripts/01_collect_data.py`: Multi-source collection
- `scripts/02_preprocess_data.py`: Cleaning and filtering
- `scripts/03_create_training_data.py`: Instruction formatting
- `scripts/04_tokenize_data.py`: Tokenization

### 2. Model Architecture

**Base Model**: TinyLlama-1.1B-Chat-v1.0
- Parameters: 1.1 billion
- Pre-training: 3 trillion tokens
- Context length: 2048 tokens

**Fine-Tuning Method**: LoRA (Low-Rank Adaptation)
- Trainable parameters: ~0.5% (5.5M / 1.1B)
- Target modules: q_proj, k_proj, v_proj, o_proj
- Quantization: 4-bit (bitsandbytes)

**Training Platform**: Google Colab (Tesla T4, 16GB VRAM)

### 3. Hyperparameter Optimization

Systematic testing of 3 configurations:

```python
# exp1: Baseline configuration
learning_rate = 2e-4
lora_rank = 8

# exp2: Lower learning rate (BEST)
learning_rate = 1e-4  
lora_rank = 8

# exp3: Higher capacity
learning_rate = 2e-4
lora_rank = 16
```

**Key Finding**: Conservative hyperparameters (lower LR, moderate rank) achieve best results for compact models.

### 4. Evaluation

**Test Set**: 200 reviews (stratified by rating)

**Metrics**:
- ROUGE-1/2/L: Text overlap and fluency
- BLEU: N-gram precision

**Script**: `checkpoint_model_evaluation.py`

### 5. Error Analysis

**Identified Issues**:
- Format leakage: 100% (systematic)
- Truncation: 80%
- Verbatim copying: 70%
- Repetition: 29.5%

**Script**: `error_analysis.py`

---

## 💻 Usage

### Run Evaluation

```bash
# Evaluate all models (takes ~70 minutes)
python checkpoint_model_evaluation.py

# Output: evaluation_results/
```

### Run Error Analysis

```bash
# Analyze best model predictions (takes ~30 seconds)
python error_analysis.py

# Output: error_analysis_results/
```

### Run Demo

```bash
# Interactive mode
python demo_app.py

# Batch mode (for video/screenshots)
python demo_app.py --batch
```

**Demo Features:**
- Pre-loaded positive/negative/neutral reviews
- Custom review input
- Real-time summary generation
- Post-processing cleanup

---

## 🔄 Reproducing Results

### Complete Reproduction (1.5 hours)

Follow these steps in order:

```bash
# 1. Setup environment (5 min)
source venv/bin/activate
pip install -r requirements.txt

# 2. Verify models exist (1 min)
ls -lh all_training_results/models/exp*/

# 3. Run evaluation (70 min)
python checkpoint_model_evaluation.py

# 4. Run error analysis (1 min)
python error_analysis.py

# 5. Test demo (5 min)
python demo_app.py --batch
```

**Detailed instructions**: See [REPRODUCTION_INSTRUCTIONS.md](REPRODUCTION_INSTRUCTIONS.md)

### Success Criteria

Your reproduction is successful if:
- ✅ exp2 achieves ROUGE-1 between 0.49-0.52 (±2% variance is normal)
- ✅ exp2 outperforms exp1 and exp3
- ✅ Error analysis identifies 100% format issues
- ✅ Demo runs without errors

---

## 📚 Documentation

### Core Documents

1. **[ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)**
   - Platform-specific installation
   - Dependency management
   - Troubleshooting

2. **[REPRODUCTION_INSTRUCTIONS.md](REPRODUCTION_INSTRUCTIONS.md)**
   - Step-by-step reproduction guide
   - Expected outputs
   - Verification checklist

3. **[CODE_DOCUMENTATION.md](CODE_DOCUMENTATION.md)**
   - Function-by-function reference
   - Code modification guide
   - Testing and debugging

4. **[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)**
   - Complete methodology (5-7 pages)
   - Detailed analysis
   - Limitations and future work

### Generated Reports

- **evaluation_results/evaluation_report.md**: Metrics comparison
- **error_analysis_results/error_analysis_report.md**: Failure analysis

---

## 🛠️ Technical Stack

### Core Libraries

```
torch==2.1.0                 # Deep learning framework
transformers==4.35.0         # HuggingFace models
peft==0.7.0                  # LoRA implementation
bitsandbytes==0.41.0         # 4-bit quantization
datasets==2.15.0             # Data loading
accelerate==0.25.0           # Distributed training
```

### Evaluation

```
evaluate==0.4.1              # Metrics framework
rouge-score==0.1.2           # ROUGE metric
sacrebleu==2.3.1             # BLEU metric
nltk==3.8.1                  # NLP utilities
```

### Data Processing

```
pandas==2.1.3                # DataFrame operations
numpy==1.24.3                # Numerical computing
tqdm==4.66.1                 # Progress bars
```

### Visualization

```
matplotlib==3.8.2            # Plotting
seaborn==0.13.0              # Statistical graphics
```

**Complete list**: See [requirements.txt](requirements.txt)

---

## 🔍 Key Findings

### What Worked Well

1. **LoRA Efficiency**: 0.5% trainable parameters achieved strong results
2. **Lower Learning Rate**: 1e-4 outperformed 2e-4 by 6.5% ROUGE-1
3. **Moderate Rank**: rank=8 optimal for 1B-scale models
4. **Multi-Source Data**: Combining Amazon + Yelp improved generalization

### Identified Challenges

1. **Format Leakage** (100%): Training data contained template markers
2. **Truncation** (80%): max_tokens=150 insufficient for some reviews
3. **Verbatim Copying** (70%): Model defaults to extractive approach

### Recommendations

**Immediate Fixes:**
- Increase max_tokens to 200
- Enhance post-processing pipeline
- Lower temperature to 0.5

**Training Improvements:**
- Clean template markers from data
- Add abstractive summarization examples
- Extend to 4-5 epochs

**Advanced Techniques:**
- Reinforcement learning from human feedback (RLHF)
- Constraint decoding for sentence completion
- Multi-task training with completion

---

## 📈 Performance Benchmarks

### Inference Speed

- **Per Review**: 2-5 seconds (Mac M1)
- **Batch Processing**: ~720 reviews/hour
- **Memory Usage**: 2.2GB (float16)

### Training Time

- **Per Experiment**: ~2 hours (Tesla T4)
- **Total Training**: ~6 GPU-hours for 3 experiments

---

## 🎓 Academic Context

**Assignment**: Fine-Tuning Large Language Models  
**Course**: INFO 7375  
**Institution**: Northeastern University  
**Semester**: Fall 2025

**Functional Requirements Met:**
- ✅ Dataset Preparation
- ✅ Model Selection & Justification
- ✅ Fine-Tuning Setup
- ✅ Hyperparameter Optimization (3 configs)
- ✅ Model Evaluation (multiple metrics)
- ✅ Error Analysis (patterns + improvements)
- ✅ Inference Pipeline (working demo)
- ✅ Documentation & Reproducibility

---

## 🤝 Acknowledgments

- **HuggingFace** for Transformers, PEFT, and Datasets libraries
- **Google Colab** for free GPU access
- **TinyLlama Team** for the base model
- **Course Instructor** for guidance throughout the project

---

## 📄 License

This project is for academic purposes as part of INFO 7375 coursework.

Base model (TinyLlama-1.1B-Chat-v1.0) is licensed under Apache 2.0.

---

## 📞 Contact

**Sravan Kumar Kurapati**  
Northeastern University  
Email: kurapati.sr@northeastern.edu

---

## 🔗 References

1. Hu et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models"
2. Dettmers et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs"
3. Zhang et al. (2023). "TinyLlama: An Open-Source Small Language Model"
4. HuggingFace Transformers: https://huggingface.co/docs/transformers/
5. PEFT Library: https://github.com/huggingface/peft

---

## ✅ Verification Checklist

Before submission, verify:

- [ ] All code files present and documented
- [ ] requirements.txt complete and tested
- [ ] All 4 documentation files included
- [ ] Evaluation results in evaluation_results/
- [ ] Error analysis in error_analysis_results/
- [ ] Demo script runs without errors
- [ ] README provides clear setup instructions
- [ ] Technical report is comprehensive (5-7 pages)
- [ ] Video walkthrough recorded and uploaded

---

**Last Updated**: October 23, 2025  
**Version**: 1.0  
