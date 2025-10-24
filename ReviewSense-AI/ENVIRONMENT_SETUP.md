# Environment Setup Guide

**ReviewSense AI - Complete Environment Configuration**

This document provides clear, step-by-step instructions to set up your development environment for the ReviewSense AI project.

---

## Prerequisites

Before starting, ensure you have:
- **Operating System**: macOS 10.15+, Ubuntu 20.04+, or Windows 10+ with WSL2
- **Python**: Version 3.10 or 3.11 (avoid 3.13 due to compatibility issues)
- **Disk Space**: At least 20GB free
- **RAM**: Minimum 8GB (16GB recommended)
- **Internet Connection**: Required for downloading models and libraries

---

## Step 1: Create Project Directory

```bash
# Navigate to your preferred location
cd ~/Documents/INFO_7375/Assigment-FineTuningLLM/

# Create project folder
mkdir ReviewSense-AI
cd ReviewSense-AI

# Create subdirectories
mkdir -p data/raw data/processed data/splits
mkdir -p models scripts evaluation_results error_analysis_results
mkdir -p evaluation_checkpoints all_training_results/models
```

**Expected Structure:**
```
ReviewSense-AI/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── models/
├── scripts/
├── evaluation_results/
├── error_analysis_results/
├── evaluation_checkpoints/
└── all_training_results/
    └── models/
```

---

## Step 2: Create Virtual Environment

### For macOS/Linux:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify activation (should show path with 'venv')
which python
```

### For Windows:

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate

# Verify activation
where python
```

**Success Indicator:** Your terminal prompt should now show `(venv)` at the beginning.

---

## Step 3: Install Core Dependencies

### 3.1 Upgrade pip

```bash
pip install --upgrade pip
```

### 3.2 Install PyTorch

**For Mac (M1/M2/M3):**
```bash
pip install torch==2.1.0 torchvision torchaudio
```

**For Linux/Windows with NVIDIA GPU:**
```bash
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CPU-only (any platform):**
```bash
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3.3 Install Transformers and Fine-Tuning Libraries

```bash
pip install transformers==4.35.0
pip install datasets==2.15.0
pip install peft==0.7.0
pip install accelerate==0.25.0
pip install bitsandbytes==0.41.0
```

### 3.4 Install Evaluation Libraries

```bash
pip install evaluate==0.4.1
pip install rouge-score==0.1.2
pip install nltk==3.8.1
pip install sacrebleu==2.3.1
```

### 3.5 Install Data Processing Libraries

```bash
pip install pandas==2.1.3
pip install numpy==1.24.3
pip install tqdm==4.66.1
```

### 3.6 Install Visualization Libraries

```bash
pip install matplotlib==3.8.2
pip install seaborn==0.13.0
```

---

## Step 4: Download NLTK Data

### Standard Method:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### If SSL Certificate Error (Common on Mac):

```bash
python3 << 'EOF'
import ssl
import nltk

# Bypass SSL verification
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
print("✅ NLTK data downloaded successfully!")
EOF
```

---

## Step 5: Verify Installation

Run this verification script:

```bash
python << 'EOF'
import sys
print("="*60)
print("ENVIRONMENT VERIFICATION")
print("="*60)

# Check Python version
print(f"\n✓ Python Version: {sys.version.split()[0]}")

# Check core libraries
try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  - CUDA Available: {torch.cuda.is_available()}")
    print(f"  - MPS Available: {torch.backends.mps.is_available()}")
except ImportError as e:
    print(f"✗ PyTorch: Not installed ({e})")

try:
    import transformers
    print(f"✓ Transformers: {transformers.__version__}")
except ImportError:
    print("✗ Transformers: Not installed")

try:
    import peft
    print(f"✓ PEFT: {peft.__version__}")
except ImportError:
    print("✗ PEFT: Not installed")

try:
    import datasets
    print(f"✓ Datasets: {datasets.__version__}")
except ImportError:
    print("✗ Datasets: Not installed")

try:
    import evaluate
    print(f"✓ Evaluate: {evaluate.__version__}")
except ImportError:
    print("✗ Evaluate: Not installed")

try:
    import pandas
    print(f"✓ Pandas: {pandas.__version__}")
except ImportError:
    print("✗ Pandas: Not installed")

try:
    import matplotlib
    print(f"✓ Matplotlib: {matplotlib.__version__}")
except ImportError:
    print("✗ Matplotlib: Not installed")

print("\n" + "="*60)
print("VERIFICATION COMPLETE")
print("="*60)
EOF
```

**Expected Output:**
```
============================================================
ENVIRONMENT VERIFICATION
============================================================

✓ Python Version: 3.10.x
✓ PyTorch: 2.1.0
  - CUDA Available: False
  - MPS Available: True
✓ Transformers: 4.35.0
✓ PEFT: 0.7.0
✓ Datasets: 2.15.0
✓ Evaluate: 0.4.1
✓ Pandas: 2.1.3
✓ Matplotlib: 3.8.2

============================================================
VERIFICATION COMPLETE
============================================================
```

---

## Step 6: Create Requirements File

Generate a requirements file for future reference:

```bash
pip freeze > requirements.txt
```

This creates a file listing all installed packages and their versions.

---

## Troubleshooting

### Issue: "command not found: python3"

**Solution:**
- On Mac: Install Python from https://www.python.org/downloads/
- On Linux: `sudo apt-get install python3 python3-venv`
- On Windows: Download from python.org and ensure "Add to PATH" is checked

### Issue: "pip: command not found"

**Solution:**
```bash
# Mac/Linux
python3 -m ensurepip --upgrade

# Windows
python -m ensurepip --upgrade
```

### Issue: Virtual environment not activating

**Solution:**
```bash
# Ensure you're in the project directory
cd ~/Documents/INFO_7375/Assigment-FineTuningLLM/ReviewSense-AI/

# Try activating again
source venv/bin/activate  # Mac/Linux
.\venv\Scripts\activate   # Windows
```

### Issue: "bitsandbytes" installation fails

**Solution:**
```bash
# bitsandbytes requires a CUDA-capable GPU or can be skipped for CPU-only
# For Mac/CPU-only systems, you can skip this:
# Comment out bitsandbytes in requirements if not using GPU training
```

### Issue: Slow package installation

**Solution:**
```bash
# Use a faster mirror
pip install --upgrade pip
pip config set global.index-url https://pypi.org/simple
```

---

## Platform-Specific Notes

### macOS (M1/M2/M3)
- Use native Apple Silicon PyTorch (no CUDA needed)
- MPS (Metal Performance Shaders) acceleration available
- Evaluation runs efficiently on CPU/MPS

### Linux with NVIDIA GPU
- Install CUDA 11.8 or 12.1
- Use PyTorch with CUDA support
- Can train models locally (not required for this project)

### Windows
- WSL2 recommended for better compatibility
- Native Windows works but may have path issues
- Use PowerShell for commands

---

## Next Steps

Once environment setup is complete:

1. ✅ **Environment configured**
2. 📥 **Next**: Download trained models (see REPRODUCTION_INSTRUCTIONS.md)
3. 🔄 **Then**: Run evaluation pipeline
4. 📊 **Finally**: Generate results

---

## Hardware Requirements Summary

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | Dual-core 2.0 GHz | Quad-core 3.0 GHz |
| **RAM** | 8GB | 16GB |
| **Storage** | 20GB free | 50GB SSD |
| **GPU** | Not required | Optional for faster inference |
| **Internet** | Required | Required |

---

## Environment Setup Checklist

Before proceeding, verify:

- [ ] Python 3.10 or 3.11 installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed successfully
- [ ] NLTK data downloaded
- [ ] Verification script runs without errors
- [ ] requirements.txt file created
- [ ] Project directory structure exists

---

**Setup Time**: Approximately 15-20 minutes

**Document Version**: 1.0  
**Last Updated**: October 2025  
**Maintained By**: Sravan Kumar Kurapati
