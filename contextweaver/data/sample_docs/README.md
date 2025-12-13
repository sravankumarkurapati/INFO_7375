# ContextWeaver Knowledge Base

## Overview

This directory contains the sample knowledge base documents used for demonstrating ContextWeaver's capabilities.

## Contents

### 1. coffee_study_2018.txt
- **Year:** 2018
- **Type:** Research Study
- **Topic:** Coffee and cardiovascular risk
- **Key Finding:** High coffee consumption (>5 cups/day) associated with 23% increased risk
- **Limitation:** Did not control for confounding variables

### 2. coffee_study_2023.txt
- **Year:** 2023
- **Type:** Research Study
- **Topic:** Protective effects of moderate coffee
- **Key Finding:** Moderate consumption (2-3 cups/day) shows 15% risk reduction
- **Strength:** Rigorous controls for smoking, diet, exercise

### 3. meta_analysis_2022.txt
- **Year:** 2022
- **Type:** Meta-Analysis
- **Topic:** Review of 50 coffee studies (2010-2022)
- **Key Finding:** Explains contradictions via methodological differences
- **Insight:** Confounding variables explain early negative findings

## Why These Documents?

These 3 documents create an ideal test case for ContextWeaver because they:

✅ **Temporal Range:** 2018-2023 (6 years)  
✅ **Apparent Contradictions:** High vs moderate consumption  
✅ **Methodological Evolution:** Uncontrolled → controlled studies  
✅ **Multiple Evidence Types:** Individual studies + meta-analysis  
✅ **Clear Narrative:** Shows knowledge evolution  

## Test Results

When processed by ContextWeaver:
- **Chunks Created:** 3
- **Processing Time:** 0.01s
- **Vector Embeddings:** 5.70s
- **Contradictions Detected:** 2 (HIGH severity)
- **Knowledge Graph:** 3 nodes, 4 edges

## Adding Your Own Documents

To add documents to the knowledge base:
```bash
# 1. Copy files to this directory
cp your_document.pdf data/sample_docs/

# 2. Supported formats
# - PDF (.pdf)
# - Text (.txt)
# - Word (.docx)

# 3. Re-run ingestion
python -c "
from src.contextweaver_pipeline import ContextWeaverPipeline
pipeline = ContextWeaverPipeline()
pipeline.ingest_documents(['data/sample_docs/your_document.pdf'])
"
```

## Document Statistics
```
Total Documents: 3
Total Size: ~15 KB
Domains: research (100%)
Years: 2018, 2022, 2023
Credibility: Medium (60%)
Coverage Score: 23.3%
```

## RAG Component Implementation

These documents are used to demonstrate:

1. ✅ **Build knowledge base for domain** - Hierarchical organization
2. ✅ **Implement vector storage** - ChromaDB with OpenAI embeddings
3. ✅ **Design chunking strategies** - 4 strategies (hybrid used)
4. ✅ **Create ranking mechanisms** - Multi-factor ranking

**All RAG requirements met** ✅

---

**Knowledge Base Maintained By:** Sravan Kumar Kurapati  
**Last Updated:** December 12, 2025