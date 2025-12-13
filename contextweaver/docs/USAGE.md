# 📖 ContextWeaver Usage Guide

Complete API reference and usage examples for ContextWeaver.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Core API](#core-api)
- [Component Usage](#component-usage)
- [Advanced Features](#advanced-features)
- [Configuration](#configuration)
- [Best Practices](#best-practices)

---

## Quick Start

### Streamlit Web Interface (Easiest)
```bash
# Start the web app
streamlit run app/streamlit_app.py

# Navigate to http://localhost:8501
# Use the interactive UI to:
# 1. Initialize the system
# 2. Enter queries
# 3. View results with visualizations
```

### Python API (Programmatic)
```python
from src.contextweaver_pipeline import ContextWeaverPipeline

# Initialize
pipeline = ContextWeaverPipeline(use_existing_db=False)

# Ingest documents
report = pipeline.ingest_documents([
    'data/sample_docs/coffee_study_2018.txt',
    'data/sample_docs/coffee_study_2023.txt',
    'data/sample_docs/meta_analysis_2022.txt'
])

# Query with all features
result = pipeline.query(
    "Is moderate coffee consumption safe for heart health?",
    enable_multi_hop=True,
    enable_contradiction_detection=True,
    enable_uncertainty=True,
    enable_fact_checking=True
)

# Access results
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['uncertainty']['confidence_score']:.2%}")
print(f"Sources: {result['retrieval']['num_documents']}")
```

---

## Core API

### ContextWeaverPipeline

Main pipeline class that orchestrates all components.

#### Initialization
```python
pipeline = ContextWeaverPipeline(use_existing_db=False)
```

**Parameters:**
- `use_existing_db` (bool): Load existing ChromaDB if True, create new if False

#### Methods

##### `ingest_documents(file_paths: List[str]) -> Dict`

Ingests documents into the knowledge base.
```python
report = pipeline.ingest_documents([
    'path/to/doc1.pdf',
    'path/to/doc2.txt',
    'path/to/doc3.docx'
])

# Returns:
{
    'files_processed': 3,
    'chunks_created': 45,
    'knowledge_base_stats': {...},
    'vector_store_stats': {...},
    'graph_stats': {...},
    'contradictions_found': 2
}
```

**Supported Formats:**
- PDF (`.pdf`)
- Text (`.txt`)
- Word (`.docx`)

##### `query(query: str, **options) -> Dict`

Main query method with full pipeline.
```python
result = pipeline.query(
    query="Your question here",
    enable_multi_hop=True,              # Multi-hop reasoning
    enable_contradiction_detection=True, # Find contradictions
    enable_uncertainty=True,             # Uncertainty quantification
    enable_fact_checking=True,           # Fact-check answer
    top_k=10                            # Number of documents to retrieve
)
```

**Parameters:**
- `query` (str): User question
- `enable_multi_hop` (bool): Enable multi-hop reasoning (default: True)
- `enable_contradiction_detection` (bool): Detect contradictions (default: True)
- `enable_uncertainty` (bool): Quantify uncertainty (default: True)
- `enable_fact_checking` (bool): Fact-check answer (default: True)
- `top_k` (int): Number of documents to retrieve (default: 10)

**Returns:** Dictionary with:
```python
{
    'query': str,                    # Original query
    'answer': str,                   # Generated answer
    'raw_answer': str,               # Answer without formatting
    'status': str,                   # 'SUCCESS', 'ERROR', 'NO_DOCUMENTS'
    'retrieval': {
        'num_documents': int,
        'top_sources': List[str],
        'retrieval_source': str,     # 'local', 'web', 'llm_direct'
        'retrieval_confidence': float,
        'web_results_used': int,
        'local_results_used': int
    },
    'reasoning': {                   # If multi-hop enabled
        'reasoning_chain': List[Dict],
        'hops_used': int,
        'confidence': float,
        'citations': List[Dict]
    },
    'contradictions': {              # If contradiction detection enabled
        'contradictions': List[Dict],
        'num_contradictions': int,
        'overall_severity': str
    },
    'uncertainty': {                 # If uncertainty enabled
        'confidence_score': float,
        'confidence_level': str,
        'uncertainty_sources': List[str],
        'evidence_gaps': List[str],
        'sensitivity_analysis': Dict
    },
    'fact_check': {                  # If fact-checking enabled
        'overall_score': float,
        'verification_level': str,
        'num_verified': int,
        'num_contradicted': int,
        'flagged_claims': List[Dict]
    },
    'red_flags': List[Dict],
    'citations': List[Dict],
    'components_used': Dict[str, bool]
}
```

---

## Component Usage

### 1. Document Processor
```python
from src.document_processor import DocumentProcessor

processor = DocumentProcessor()

# Load documents
documents = processor.load_documents([
    'file1.pdf',
    'file2.txt'
])

# Access knowledge base statistics
processor.knowledge_base.print_statistics()

# Filter documents
filtered = processor.knowledge_base.get_documents_by_filter(
    domain='medical',
    year_min=2020,
    min_credibility=0.8
)
```

**Chunking Strategies:**
```python
from src.document_processor import AdvancedChunker

# Fixed-size chunking
chunker = AdvancedChunker(
    chunk_size=1000,
    chunk_overlap=200,
    strategy='fixed'
)

# Semantic chunking (respects meaning)
chunker = AdvancedChunker(strategy='semantic')

# Sentence-based chunking
chunker = AdvancedChunker(strategy='sentence')

# Hybrid (best results)
chunker = AdvancedChunker(strategy='hybrid')
```

### 2. Vector Store
```python
from src.vector_store import VectorStoreManager, AdvancedRetriever

# Initialize
vector_store = VectorStoreManager(
    persist_directory='./data/chroma_db',
    collection_name='my_docs'
)

# Create vector store
vectorstore = vector_store.create_vectorstore(documents)

# Similarity search
results = vector_store.similarity_search(
    query="coffee health",
    k=5
)

# Similarity search with scores
results_with_scores = vector_store.similarity_search_with_score(
    query="coffee health",
    k=5
)

# Advanced retrieval with multi-factor ranking
retriever = AdvancedRetriever(vector_store)
ranked_docs = retriever.rerank_documents(query, documents)
```

### 3. Multi-Hop Reasoning
```python
from src.reasoning_engine import MultiHopReasoningEngine

engine = MultiHopReasoningEngine()

# Reason across documents
result = engine.reason_across_documents(
    query="Why do studies contradict each other?",
    documents=documents,
    max_hops=3
)

print(f"Hops used: {result['hops_used']}")
print(f"Confidence: {result['confidence']:.2%}")

# Access reasoning chain
for step in result['reasoning_chain']:
    print(f"Hop {step['hop_number']}: {step['extracted_info']}")
```

### 4. Contradiction Detection
```python
from src.reasoning_engine import ContradictionDetector

detector = ContradictionDetector()

# Detect contradictions
result = detector.detect_contradictions(documents)

print(f"Found {result['num_contradictions']} contradictions")

for contradiction in result['contradictions']:
    print(f"Severity: {contradiction['severity']}")
    print(f"Claim A: {contradiction['claim_A']}")
    print(f"Claim B: {contradiction['claim_B']}")
```

### 5. Uncertainty Quantification
```python
from src.uncertainty_quantification import UncertaintyQuantifier

quantifier = UncertaintyQuantifier()

# Quantify uncertainty
uncertainty = quantifier.quantify_uncertainty(
    query="Is coffee safe?",
    answer="Yes, moderate coffee...",
    supporting_documents=supporting_docs,
    contradicting_documents=contradicting_docs
)

print(f"Confidence: {uncertainty['confidence_score']:.2%}")
print(f"Level: {uncertainty['confidence_level']}")

# Uncertainty sources
for source in uncertainty['uncertainty_sources']:
    print(f"⚠️ {source}")

# Evidence gaps
for gap in uncertainty['evidence_gaps']:
    print(f"📊 {gap}")

# Sensitivity analysis
for scenario, confidence in uncertainty['sensitivity_analysis'].items():
    print(f"{scenario}: {confidence:.2%}")
```

### 6. Automated Fact-Checking
```python
from src.fact_checker import AutomatedFactChecker

fact_checker = AutomatedFactChecker()

# Fact-check answer
fact_check = fact_checker.fact_check_answer(
    answer="Moderate coffee is safe...",
    source_documents=documents
)

print(f"Verification: {fact_check['overall_score']:.2%}")
print(f"Level: {fact_check['verification_level']}")

# View claim verifications
for verification in fact_check['verifications']:
    print(f"{verification['status']}: {verification['claim']}")
```

### 7. Knowledge Graph
```python
from src.document_graph import DocumentGraph

graph = DocumentGraph()

# Build graph
graph.build_graph(
    documents=documents,
    contradictions=contradictions,
    temporal_order=True
)

# Get statistics
stats = graph.get_graph_statistics()
print(f"Nodes: {stats['num_nodes']}")
print(f"Edges: {stats['num_edges']}")

# Calculate document importance (PageRank)
importance = graph.calculate_document_importance()

# Graph-based retrieval
retrieved = graph.graph_based_retrieval(
    seed_documents=['doc_0', 'doc_1'],
    relationship_types=['cites', 'similar_topic'],
    max_hops=2
)

# Visualize
graph.visualize_graph_plotly('graph.html')
```

### 8. Hybrid Retrieval
```python
from src.web_search_fallback import HybridRetriever, WebSearchFallback

web_search = WebSearchFallback()
hybrid = HybridRetriever(vector_store, web_search)

# Automatic 3-tier fallback
result = hybrid.retrieve(
    query="Is chicken healthy?",
    k=10
)

print(f"Source: {result['source']}")  # 'local', 'web', or 'llm_direct'
print(f"Confidence: {result['confidence']:.2%}")
print(f"Documents: {len(result['documents'])}")
```

---

## Advanced Features

### Custom Prompt Templates
```python
from src.prompt_engineering import PromptEngineeringSystem

prompt_system = PromptEngineeringSystem()

# Create custom prompt
prompt = prompt_system.create_prompt(
    template_name='multi_hop_reasoning',
    variables={
        'query': 'Your question',
        'documents': formatted_docs
    },
    use_few_shot=True,
    num_examples=2,
    use_chain_of_thought=True
)
```

### Synthetic Data Generation
```python
from src.synthetic_data_generator import SyntheticDataGenerator

generator = SyntheticDataGenerator()

# Generate Q&A pairs
qa_pairs = generator.generate_qa_pairs(
    source_documents=documents,
    num_pairs=50,
    difficulty_levels=['easy', 'medium', 'hard']
)

# Generate contradictory pairs
contradictions = generator.generate_contradictory_pairs(
    base_documents=documents,
    num_pairs=10
)

# Generate evaluation benchmark
benchmark = generator.generate_evaluation_benchmark(
    documents=documents,
    num_samples=100
)

# Save benchmark
generator.save_benchmark(benchmark, 'benchmark.json')
```

### Data Augmentation
```python
from src.synthetic_data_generator import DataAugmentation

augmenter = DataAugmentation()

# Augment queries
augmented = augmenter.augment_queries(
    queries=['Is coffee safe?', 'Coffee health effects?'],
    num_variations=3
)

# Augment documents
augmented_docs = augmenter.augment_documents(
    documents=documents,
    augmentation_types=['paraphrase', 'simplify']
)
```

---

## Configuration

### Runtime Configuration
```python
from src.config import Config

# Override defaults programmatically
Config.CHUNK_SIZE = 1500
Config.TOP_K_DOCUMENTS = 15
Config.TEMPERATURE = 0.2

# Validate configuration
Config.validate()

# Print configuration summary
Config.print_summary()
```

### Custom Ranking Weights
```python
# Adjust ranking factors
Config.RANKING_WEIGHTS = {
    'similarity': 0.40,      # Increase similarity weight
    'credibility': 0.30,     # Increase credibility weight
    'recency': 0.15,
    'quality': 0.10,
    'alignment': 0.05
}
```

---

## Best Practices

### 1. Document Ingestion
```python
# ✅ Good: Process documents in batches
batch_size = 10
for i in range(0, len(all_files), batch_size):
    batch = all_files[i:i+batch_size]
    pipeline.ingest_documents(batch)

# ❌ Bad: Processing too many at once
pipeline.ingest_documents(all_10000_files)  # Memory issues!
```

### 2. Query Optimization
```python
# ✅ Good: Disable unused components for speed
result = pipeline.query(
    query="Simple factual question",
    enable_multi_hop=False,        # Not needed for simple queries
    enable_fact_checking=False,
    top_k=5                        # Fewer documents = faster
)

# ✅ Good: Enable everything for complex queries
result = pipeline.query(
    query="Complex multi-document reasoning question",
    enable_multi_hop=True,
    enable_contradiction_detection=True,
    enable_uncertainty=True,
    enable_fact_checking=True,
    top_k=10
)
```

### 3. Error Handling
```python
from src.prompt_engineering import EdgeCaseHandler

edge_handler = EdgeCaseHandler(prompt_system)

# Validate before querying
is_valid, error_response = edge_handler.validate_and_handle(
    query, documents
)

if not is_valid:
    print(f"Error: {error_response['message']}")
else:
    result = pipeline.query(query)
```

### 4. Memory Management
```python
# Clear cache periodically
if len(processed_files) > 1000:
    vector_store.delete_collection()
    vector_store = VectorStoreManager(...)
```

### 5. API Cost Optimization
```python
# Use cheaper models for non-critical tasks
Config.MODEL_NAME = 'gpt-3.5-turbo'  # Faster + cheaper

# Use smaller embedding model
Config.EMBEDDING_MODEL = 'text-embedding-ada-002'

# Reduce token usage
Config.MAX_TOKENS = 1000
Config.CHUNK_SIZE = 500
```

---

## Example Use Cases

### Use Case 1: Research Paper Analysis
```python
# Ingest research papers
papers = [
    'papers/transformer_2017.pdf',
    'papers/bert_2018.pdf',
    'papers/gpt3_2020.pdf'
]
pipeline.ingest_documents(papers)

# Analyze evolution
result = pipeline.query(
    "How has the transformer architecture evolved from 2017 to 2020?",
    enable_multi_hop=True
)
```

### Use Case 2: Medical Literature Review
```python
# Ingest medical studies
studies = glob.glob('medical_studies/*.pdf')
pipeline.ingest_documents(studies)

# Query with fact-checking
result = pipeline.query(
    "What is the current consensus on coffee and heart health?",
    enable_contradiction_detection=True,
    enable_uncertainty=True,
    enable_fact_checking=True
)

# Check confidence
if result['uncertainty']['confidence_score'] < 0.7:
    print("⚠️ Low confidence - review uncertainty sources")
```

### Use Case 3: Legal Document Analysis
```python
# Ingest contracts and amendments
legal_docs = [
    'main_contract.pdf',
    'amendment_2020.pdf',
    'amendment_2022.pdf'
]
pipeline.ingest_documents(legal_docs)

# Check for contradictions
result = pipeline.query(
    "What are the current CEO authority limits?",
    enable_contradiction_detection=True,
    enable_multi_hop=True
)

# Review contradictions
if result['contradictions']['num_contradictions'] > 0:
    for c in result['contradictions']['contradictions']:
        print(f"⚠️ {c['severity']}: {c['explanation']}")
```

---

## Common Patterns

### Pattern 1: Batch Processing
```python
queries = [
    "Query 1",
    "Query 2",
    "Query 3"
]

results = []
for query in queries:
    result = pipeline.query(query)
    results.append(result)
    
# Save results
import json
with open('batch_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
```

### Pattern 2: Confidence-Based Decision Making
```python
result = pipeline.query(query, enable_uncertainty=True)

confidence = result['uncertainty']['confidence_score']

if confidence >= 0.8:
    print("✅ High confidence answer")
elif confidence >= 0.5:
    print("⚠️ Medium confidence - review sources")
else:
    print("❌ Low confidence - more evidence needed")
    print(f"Evidence gaps: {result['uncertainty']['evidence_gaps']}")
```

### Pattern 3: Progressive Enhancement
```python
# Start simple
result = pipeline.query(query, 
    enable_multi_hop=False,
    enable_fact_checking=False
)

# If answer quality is low, enable more features
if result['uncertainty']['confidence_score'] < 0.7:
    result = pipeline.query(query,
        enable_multi_hop=True,
        enable_contradiction_detection=True,
        enable_uncertainty=True,
        enable_fact_checking=True
    )
```

---

## API Reference Summary

### Quick Reference Table

| Class | Method | Purpose |
|-------|--------|---------|
| `ContextWeaverPipeline` | `ingest_documents()` | Load documents |
| | `query()` | Main query interface |
| `DocumentProcessor` | `load_documents()` | Process files |
| `VectorStoreManager` | `similarity_search()` | Vector search |
| `MultiHopReasoningEngine` | `reason_across_documents()` | Multi-hop reasoning |
| `ContradictionDetector` | `detect_contradictions()` | Find conflicts |
| `UncertaintyQuantifier` | `quantify_uncertainty()` | Uncertainty analysis |
| `AutomatedFactChecker` | `fact_check_answer()` | Verify claims |
| `DocumentGraph` | `build_graph()` | Create knowledge graph |
| `HybridRetriever` | `retrieve()` | 3-tier retrieval |

---

## Next Steps

- **Examples:** See [EXAMPLES.md](EXAMPLES.md) for complete code examples
- **Architecture:** Read [ARCHITECTURE.md](ARCHITECTURE.md) to understand internals
- **Testing:** Check [TESTING.md](TESTING.md) for testing guide
- **Troubleshooting:** See [SETUP.md](SETUP.md) for common issues

---

**Questions?** Contact: kurapati.s@northeastern.edu