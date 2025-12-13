# 🛠️ ContextWeaver Setup Guide

Complete installation and configuration guide for ContextWeaver.

---

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Platform-Specific Instructions](#platform-specific-instructions)

---

## System Requirements

### Minimum Requirements
- **OS:** macOS, Linux, or Windows 10+
- **Python:** 3.9 or higher
- **RAM:** 4GB minimum, 8GB recommended
- **Disk Space:** 2GB free space
- **Internet:** Required for API calls and package installation

### Recommended Requirements
- **Python:** 3.10+
- **RAM:** 8GB+
- **CPU:** Multi-core processor
- **GPU:** Not required (API-based models)

### Dependencies
- OpenAI API account with credits
- Git installed
- pip package manager

---

## Installation Methods

### Method 1: Standard Installation (Recommended)

#### Step 1: Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/contextweaver.git
cd contextweaver
```

#### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Verify activation (should show venv path)
which python  # macOS/Linux
where python  # Windows
```

#### Step 3: Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Verify installation
pip list | grep langchain
pip list | grep chromadb
```

**Expected Output:**
```
langchain                     0.1.0
langchain-community           0.0.20
langchain-openai              0.0.5
chromadb                      0.4.22
```

#### Step 4: Configure Environment Variables
```bash
# Copy environment template
cp .env.example .env

# Edit with your API key
nano .env  # or use any text editor
```

**Add to `.env`:**
```env
# Required
OPENAI_API_KEY=sk-your-actual-api-key-here

# Optional (uses defaults if not set)
MODEL_NAME=gpt-4-turbo-preview
EMBEDDING_MODEL=text-embedding-3-small
TEMPERATURE=0.1
MAX_TOKENS=2000
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
LOG_LEVEL=INFO
```

#### Step 5: Verify Installation
```bash
# Test configuration
python -c "from src.config import Config; Config.validate(); print('✅ Configuration valid!')"

# Test imports
python -c "from src.contextweaver_pipeline import ContextWeaverPipeline; print('✅ Imports successful!')"

# Quick functionality test
python tests/test_installation.py
```

---

### Method 2: Docker Installation (Optional)

#### Prerequisites
- Docker installed
- Docker Compose installed

#### Build and Run
```bash
# Build Docker image
docker build -t contextweaver:latest .

# Run container
docker run -it --rm \
  -p 8501:8501 \
  -e OPENAI_API_KEY=your_key_here \
  contextweaver:latest

# Or use Docker Compose
docker-compose up
```

---

## Configuration

### Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | ✅ Yes | - | OpenAI API key for GPT-4 and embeddings |
| `MODEL_NAME` | No | `gpt-4-turbo-preview` | LLM model to use |
| `EMBEDDING_MODEL` | No | `text-embedding-3-small` | Embedding model |
| `TEMPERATURE` | No | `0.1` | Model temperature (0-1) |
| `MAX_TOKENS` | No | `2000` | Max response tokens |
| `CHUNK_SIZE` | No | `1000` | Document chunk size |
| `CHUNK_OVERLAP` | No | `200` | Chunk overlap size |
| `CHUNKING_STRATEGY` | No | `hybrid` | Chunking method (fixed/semantic/sentence/hybrid) |
| `TOP_K` | No | `10` | Number of documents to retrieve |
| `LOG_LEVEL` | No | `INFO` | Logging level (DEBUG/INFO/WARNING/ERROR) |

### Advanced Configuration

#### Custom Chunking Strategy
```python
# In .env or programmatically
CHUNKING_STRATEGY=semantic  # Options: fixed, semantic, sentence, hybrid
CHUNK_SIZE=1500
CHUNK_OVERLAP=300
```

#### Custom Model Selection
```python
# Use different OpenAI models
MODEL_NAME=gpt-4-1106-preview  # Newer GPT-4 Turbo
MODEL_NAME=gpt-3.5-turbo       # Faster, cheaper alternative

# Use different embedding models
EMBEDDING_MODEL=text-embedding-3-large  # Higher quality, more expensive
EMBEDDING_MODEL=text-embedding-ada-002  # Legacy model
```

#### Ranking Weights Customization
Edit `src/config.py`:
```python
RANKING_WEIGHTS = {
    'similarity': 0.35,    # Vector similarity
    'credibility': 0.20,   # Source credibility
    'recency': 0.20,       # Document recency
    'quality': 0.15,       # Content quality
    'alignment': 0.10      # Query alignment
}
```

---

## Troubleshooting

### Common Issues

#### Issue 1: "ModuleNotFoundError: No module named 'tiktoken'"
**Solution:**
```bash
# tiktoken requires Rust compiler
# On macOS:
brew install rust

# On Ubuntu/Debian:
sudo apt install cargo

# On Windows:
# Download and install from: https://www.rust-lang.org/tools/install

# Then reinstall tiktoken
pip install --force-reinstall tiktoken
```

#### Issue 2: "chromadb.errors.InvalidDimensionException"
**Cause:** Embedding dimension mismatch
**Solution:**
```bash
# Delete existing ChromaDB
rm -rf data/chroma_db/*

# Restart application (will recreate with correct dimensions)
```

#### Issue 3: "OpenAI API Error: Rate limit exceeded"
**Solution:**
```python
# Add rate limiting in .env
MAX_REQUESTS_PER_MINUTE=50

# Or upgrade OpenAI plan for higher limits
```

#### Issue 4: "Memory Error: Cannot allocate memory"
**Solution:**
```bash
# Reduce batch size in .env
CHUNK_SIZE=500
TOP_K=5

# Or increase system RAM
```

#### Issue 5: Import errors with LangChain
**Cause:** LangChain version incompatibility
**Solution:**
```bash
# Reinstall exact versions
pip uninstall langchain langchain-community langchain-openai -y
pip install langchain==0.1.0 langchain-community==0.0.20 langchain-openai==0.0.5
```

#### Issue 6: Streamlit not found
**Solution:**
```bash
pip install streamlit==1.30.0

# Or reinstall all requirements
pip install -r requirements.txt --force-reinstall
```

### Debugging Tips

#### Enable Debug Logging
```bash
# In .env
LOG_LEVEL=DEBUG
DEBUG=true

# Run with verbose output
python -u app/streamlit_app.py 2>&1 | tee debug.log
```

#### Check API Key
```bash
# Test OpenAI API key
python -c "
import openai
from src.config import Config
openai.api_key = Config.OPENAI_API_KEY
response = openai.models.list()
print('✅ API key valid!')
"
```

#### Verify ChromaDB
```bash
# Check ChromaDB status
python -c "
from src.vector_store import VectorStoreManager
vm = VectorStoreManager('./data/chroma_db', 'test')
stats = vm.get_statistics()
print(f'ChromaDB stats: {stats}')
"
```

---

## Platform-Specific Instructions

### macOS

#### Install Python 3.9+
```bash
# Using Homebrew
brew install python@3.10

# Verify
python3 --version
```

#### Install Rust (for tiktoken)
```bash
brew install rust
```

#### Common macOS Issues
- **M1/M2 Chip:** Some packages may need Rosetta 2
```bash
  softwareupdate --install-rosetta
```

### Linux (Ubuntu/Debian)

#### Install Python 3.9+
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip
```

#### Install Build Tools
```bash
sudo apt install build-essential cargo
```

#### Set Python 3.10 as Default
```bash
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
```

### Windows

#### Install Python 3.9+
1. Download from [python.org](https://www.python.org/downloads/)
2. Check "Add Python to PATH" during installation
3. Verify in Command Prompt:
```cmd
   python --version
```

#### Install Visual C++ Build Tools (for some packages)
Download from: [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

#### Windows-Specific Commands
```cmd
:: Activate virtual environment
venv\Scripts\activate

:: Run Streamlit
python -m streamlit run app/streamlit_app.py
```

---

## Quick Start Commands

### Development Mode
```bash
# Activate environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Run Streamlit app
streamlit run app/streamlit_app.py

# Run tests
pytest tests/ -v

# Run specific test
python tests/test_rag.py
```

### Production Mode
```bash
# Set production config
export LOG_LEVEL=WARNING
export DEBUG=false

# Run with optimizations
streamlit run app/streamlit_app.py --server.port 8501 --server.headless true
```

---

## Verification Checklist

After installation, verify everything works:

- [ ] Virtual environment activated
- [ ] All packages installed (`pip list`)
- [ ] `.env` file created with API key
- [ ] Config validation passes
- [ ] Sample documents exist in `data/sample_docs/`
- [ ] Streamlit app launches without errors
- [ ] Can process a test query
- [ ] ChromaDB database created in `data/chroma_db/`
- [ ] No import errors

**Run Full Verification:**
```bash
python tests/test_installation.py
```

---

## Getting Help

- **GitHub Issues:** [Report bugs or ask questions](https://github.com/YOUR_USERNAME/contextweaver/issues)
- **Documentation:** Check other docs in `docs/` folder
- **Video Tutorial:** [Setup walkthrough](VIDEO_LINK)

---

## Next Steps

After successful installation:
1. Read [USAGE.md](USAGE.md) for API reference
2. Explore [EXAMPLES.md](EXAMPLES.md) for sample code
3. Check [ARCHITECTURE.md](ARCHITECTURE.md) to understand the system
4. Try the [Streamlit demo](http://localhost:8501) with example queries

---

**Need help?** Contact: kurapati.s@northeastern.edu