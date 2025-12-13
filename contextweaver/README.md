# 🧠 ContextWeaver - Advanced Multi-Document Reasoning Engine

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 12/12 Passing](https://img.shields.io/badge/tests-12%2F12%20passing-brightgreen)](./docs/TESTING.md)

> **Next-generation AI system that goes beyond simple RAG to perform true multi-document reasoning with uncertainty quantification, automated fact-checking, and hybrid retrieval.**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Performance](#performance)
- [Documentation](#documentation)
- [Project Structure](#project-structure)
- [License](#license)

---

## 🎯 Overview

ContextWeaver is a sophisticated generative AI system designed for the INFO 7375 (Generative AI) course at Northeastern University. It implements **3 core components** and **4 major innovations** to solve complex multi-document reasoning challenges.

### What Makes It Different?

Unlike traditional RAG systems that simply retrieve and concatenate documents, ContextWeaver:

- 🔗 **Reasons across multiple documents** using multi-hop inference chains
- 🕸️ **Builds knowledge graphs** to understand document relationships
- 🎲 **Quantifies uncertainty** using Bayesian confidence estimation
- ✅ **Automatically fact-checks** claims against source documents
- 🌐 **Falls back intelligently** to web search when local knowledge is insufficient
- 📊 **Detects contradictions** and explains why they exist

### 🎓 Academic Context

- **Course:** INFO 7375 - Generative AI
- **Institution:** Northeastern University
- **Semester:** Fall 2024
- **Student:** Sravan Kumar Kurapati
- **Project Type:** Individual Final Project

---

## ⭐ Key Features

### 🏆 Core Components (3/2 Required = 150%)

| Component | Implementation | Status |
|-----------|---------------|--------|
| **🗄️ RAG System** | Knowledge base, vector storage (ChromaDB), 4 chunking strategies, multi-factor ranking | ✅ Complete |
| **✍️ Prompt Engineering** | 8 systematic templates, few-shot learning, context management, edge handling | ✅ Complete |
| **🧬 Synthetic Data Generation** | Q&A pairs, data augmentation, quality metrics (94.4%), diversity (81.9%) | ✅ Complete |

### 🌟 Advanced Innovations (4 Features)

#### 1. 🕸️ Knowledge Graph (`document_graph.py`)
- **PageRank importance scoring** - Identifies most influential documents
- **Relationship detection** - Finds citations, contradictions, temporal links
- **Graph-based retrieval** - Navigates document networks
- **Interactive visualization** - Plotly-powered graph display

**Test Result:** 3 nodes, 1 edge, PageRank working ✅

#### 2. 🎲 Uncertainty Quantification (`uncertainty_quantification.py`)
- **Bayesian confidence estimation** - Probabilistic reasoning
- **Sensitivity analysis** - "What-if" scenarios
- **Evidence gap detection** - Identifies missing information
- **Confidence calibration** - Ensures trustworthy scores

**Test Result:** 53.4% confidence (MODERATE), well-calibrated ✅

#### 3. ✅ Automated Fact-Checking (`fact_checker.py`)
- **Claim extraction** - Identifies factual statements
- **Multi-source verification** - Cross-checks against documents
- **Red flag detection** - Spots misinformation patterns
- **Risk scoring** - Quantifies misinformation risk

**Test Result:** 100% verification (HIGHLY VERIFIED) ✅

#### 4. 🌐 Hybrid Retrieval (`web_search_fallback.py`)
- **3-tier fallback system** - Local → Web → LLM
- **Intelligent routing** - Automatic source selection
- **Handles ANY query** - No knowledge gaps
- **Confidence scoring** - Per-source confidence levels

**Test Result:** Local 90%, Web 75% confidence ✅

### ⚡ Advanced Reasoning

- **Multi-hop reasoning** across documents (up to 3 hops)
- **Contradiction detection** with severity classification (HIGH/MEDIUM/LOW)
- **Citation tracking** with provenance chains
- **Temporal analysis** of knowledge evolution

**Test Result:** 2-hop reasoning, 85% confidence ✅

---

## 🚀 Quick Start

### Option 1: Web Interface (Recommended)
```bash
# 1. Clone and setup
git clone https://github.com/YOUR_USERNAME/contextweaver.git
cd contextweaver
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# 2. Configure API key
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=your-key-here

# 3. Launch Streamlit app
streamlit run app/streamlit_app.py

# 4. Open browser to http://localhost:8501
```

### Option 2: Python API
```python
from src.contextweaver_pipeline import ContextWeaverPipeline

# Initialize
pipeline = ContextWeaverPipeline(use_existing_db=False)

# Load documents
pipeline.ingest_documents(['data/sample_docs/study.txt'])

# Query with all features
result = pipeline.query(
    "Is moderate coffee consumption safe?",
    enable_multi_hop=True,
    enable_contradiction_detection=True,
    enable_uncertainty=True,
    enable_fact_checking=True
)

# Results
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['uncertainty']['confidence_score']:.2%}")
print(f"Source: {result['retrieval']['retrieval_source']}")
```

---

## 🏗️ Architecture

### System Architecture Diagram
```
┌─────────────────────────────────────────────────────────────┐
│                    USER QUERY                                │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────▼──────────────┐
         │  Query Validation        │
         │  & Classification        │
         └───────────┬──────────────┘
                     │
         ┌───────────▼──────────────┐
         │  HYBRID RETRIEVAL        │
         │  ┌─────────────────────┐ │
         │  │ Tier 1: Local KB    │ │ ← 90% confidence
         │  │ Tier 2: Web Search  │ │ ← 75% confidence
         │  │ Tier 3: LLM Direct  │ │ ← 50% confidence
         │  └─────────────────────┘ │
         └───────────┬──────────────┘
                     │
         ┌───────────▼──────────────┐
         │  Multi-Factor Ranking    │
         │  (Sim + Cred + Rec + Q)  │
         └───────────┬──────────────┘
                     │
         ┌───────────▼──────────────┐
         │  Graph Expansion         │
         │  (NetworkX traversal)    │
         └───────────┬──────────────┘
                     │
         ┌───────────▼──────────────┐
         │  Multi-Hop Reasoning     │
         │  (up to 3 hops)          │
         └───────────┬──────────────┘
                     │
         ┌───────────▼──────────────┐
         │  Contradiction Detection │
         │  (HIGH/MEDIUM/LOW)       │
         └───────────┬──────────────┘
                     │
         ┌───────────▼──────────────┐
         │  Uncertainty             │
         │  Quantification          │
         │  (Bayesian estimation)   │
         └───────────┬──────────────┘
                     │
         ┌───────────▼──────────────┐
         │  Fact-Checking           │
         │  & Red Flag Detection    │
         └───────────┬──────────────┘
                     │
         ┌───────────▼──────────────┐
         │  Response Synthesis      │
         │  with Citations          │
         └───────────┬──────────────┘
                     │
         ┌───────────▼──────────────┐
         │  FINAL ANSWER            │
         │  + Metrics + Confidence  │
         └──────────────────────────┘
```

### Technology Stack

- **LLM:** OpenAI GPT-4 Turbo Preview
- **Embeddings:** OpenAI text-embedding-3-small (1536 dimensions)
- **Vector DB:** ChromaDB 0.4.22
- **Framework:** LangChain 0.1.0
- **Graph Library:** NetworkX 3.2.1
- **UI:** Streamlit 1.30.0
- **Visualization:** Plotly 5.18.0

---

## 💻 Installation

See [docs/SETUP.md](docs/SETUP.md) for detailed installation instructions.

### Quick Install
```bash
# Clone
git clone https://github.com/YOUR_USERNAME/contextweaver.git
cd contextweaver

# Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Add your OPENAI_API_KEY to .env

# Verify
python -c "from src.config import Config; Config.validate(); print('✅ Ready!')"
```

---

## 📖 Usage

See [docs/USAGE.md](docs/USAGE.md) for complete API reference.

### Example 1: Medical Research Query
```python
query = "Is moderate coffee consumption safe for heart health?"

result = pipeline.query(
    query,
    enable_multi_hop=True,
    enable_contradiction_detection=True
)

# Output includes:
# - Multi-hop reasoning chain (2 hops)
# - Detected contradictions with explanations
# - Uncertainty quantification (69.9% confidence)
# - Fact-checked claims with sources
```

### Example 2: Out-of-Domain Query (Web Fallback)
```python
query = "Is chicken meat healthy?"

result = pipeline.query(query)

# Automatic fallback to web search
# Returns: Web-sourced answer with 75% confidence
```

---

## 📊 Performance

### Verified Test Results (December 12, 2024)

**Test Suite:** 12/12 tests passed (100% success rate)

| Metric | Value | Grade |
|--------|-------|-------|
| Document Processing | 0.01s | ⚡ A+ |
| Vector Embeddings | 5.70s | ✅ A |
| Multi-Hop Reasoning | 25.2s | ✅ B+ |
| Contradiction Detection | 9.0s | ✅ A- |
| Fact-Checking | 1.0s | ⚡ A+ |
| Full Pipeline Query | 27.7s | ✅ B+ |
| Synthetic Data Quality | 94.4% | ⭐ A |
| Synthetic Data Diversity | 81.9% | ⭐ A- |

**Detailed metrics:** See [docs/TESTING.md](docs/TESTING.md)

---

## 📚 Documentation

- **[SETUP.md](docs/SETUP.md)** - Installation and configuration guide
- **[USAGE.md](docs/USAGE.md)** - API reference and examples
- **[TESTING.md](docs/TESTING.md)** - Test results and benchmarks (100% pass rate)
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System design and components
- **[EXAMPLES.md](docs/EXAMPLES.md)** - Example queries and outputs

---

## 📁 Project Structure
```
contextweaver/
├── src/                           # Source code (11 modules)
│   ├── config.py                 # Configuration management
│   ├── document_processor.py     # Document loading & chunking
│   ├── vector_store.py          # ChromaDB vector storage
│   ├── prompt_engineering.py    # 8 prompt templates
│   ├── reasoning_engine.py      # Multi-hop reasoning
│   ├── document_graph.py        # Knowledge graph (NetworkX)
│   ├── uncertainty_quantification.py  # Bayesian confidence
│   ├── fact_checker.py          # Fact verification
│   ├── web_search_fallback.py   # Hybrid retrieval
│   ├── synthetic_data_generator.py  # Synthetic data
│   └── contextweaver_pipeline.py    # Main integration
│
├── app/
│   └── streamlit_app.py         # Web interface
│
├── data/
│   ├── sample_docs/             # Knowledge base (3 demo docs)
│   │   ├── coffee_study_2018.txt
│   │   ├── coffee_study_2023.txt
│   │   └── meta_analysis_2022.txt
│   ├── chroma_db/              # Vector database (gitignored)
│   └── synthetic_data/         # Generated datasets
│
├── tests/
│   └── test_all_components.py  # Comprehensive test suite
│
├── docs/
│   ├── SETUP.md                # Setup instructions
│   ├── USAGE.md                # API reference
│   ├── TESTING.md              # Test results
│   ├── ARCHITECTURE.md         # System design
│   └── EXAMPLES.md             # Example outputs
│
├── examples/                    # Example outputs
│   ├── example_query_coffee.json
│   ├── example_query_chicken.json
│   └── sample_outputs.md
│
├── .env.example                # Environment template
├── .gitignore                  # Git ignore rules
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md                   # This file
```

---

## 🎮 Demo

### Streamlit Web Interface

![ContextWeaver Demo](docs/images/demo_screenshot.png)

**Features:**
- 🎨 Beautiful, interactive UI with animations
- 📊 Live pipeline visualization
- 📈 Real-time metrics dashboard
- 💾 Export results (JSON, TXT)
- 🔄 Component toggle controls

**Try it:** `streamlit run app/streamlit_app.py`

---

## 🤝 Contributing

This is an academic project for INFO 7375. For questions or suggestions:

- 📧 Email: kurapati.s@northeastern.edu
- 🐙 GitHub: [Create an issue](https://github.com/YOUR_USERNAME/contextweaver/issues)

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Course Instructor:** Professor [Name], Northeastern University
- **Frameworks:** LangChain, ChromaDB, OpenAI API, Streamlit
- **Inspiration:** Research on multi-document reasoning and epistemic uncertainty

---

## 📞 Contact

**Sravan Kumar Kurapati**
- 📧 kurapati.s@northeastern.edu
- 💼 LinkedIn: [Your LinkedIn]
- 🌐 Portfolio: [Your Website]

---

**Built with ❤️ for advancing AI reasoning capabilities**

Last Updated: December 12, 2024