# 📚 ContextWeaver Example Outputs

Real examples from test runs showing system capabilities.

---

## Table of Contents

- [Example 1: Medical Research Query](#example-1-medical-research-query)
- [Example 2: Out-of-Domain Query (Web Fallback)](#example-2-out-of-domain-query)
- [Example 3: Contradiction Detection](#example-3-contradiction-detection)
- [Example 4: Multi-Hop Reasoning](#example-4-multi-hop-reasoning)
- [Example 5: Synthetic Data Generation](#example-5-synthetic-data-generation)

---

## Example 1: Medical Research Query

### Query
```
"Is moderate coffee consumption safe for heart health?"
```

### System Processing

**Step 1: Hybrid Retrieval**
```
Source: LOCAL (90% confidence)
Documents Retrieved: 4
  - coffee_study_2018.txt
  - coffee_study_2023.txt
  - meta_analysis_2022.txt
  - (1 additional via graph expansion)
```

**Step 2: Multi-Hop Reasoning (2 hops)**

**Hop 1:**
```
Extracted Info:
"2018 study found high coffee consumption increases cardiovascular risk. 
2022 meta-analysis noted reliability concerns due to confounding variables."

Connection:
The 2018 study's findings are questioned by the meta-analysis due to 
uncontrolled confounders like smoking and sugar intake.

Intermediate Conclusion:
High consumption may increase risk, but evidence quality is questionable.
```

**Hop 2:**
```
Extracted Info:
"2023 research shows moderate coffee consumption beneficial for heart health."

Connection:
Recent study with proper controls shows different results for MODERATE 
vs HIGH consumption levels.

Final Conclusion:
Moderate coffee consumption (2-3 cups/day) is safe and potentially beneficial.
```

**Step 3: Contradiction Detection**
```
Contradictions Found: 2

Contradiction #1:
  Claim A (2018): "High coffee increases risk by 23%"
  Claim B (2023): "Moderate coffee reduces risk by 15%"
  Severity: HIGH
  Explanation: Different consumption levels + improved methodology
  Confidence: 95%
```

**Step 4: Uncertainty Quantification**
```
Confidence Score: 69.9%
Confidence Level: MODERATE

Component Breakdown:
  - Evidence Sufficiency: 65%
  - Source Agreement: 95%
  - Source Quality: 92.5%
  - Contradiction Penalty: 30.1%

Uncertainty Sources:
  • Contradictory evidence (2 contradictions found)
```

**Step 5: Fact-Checking**
```
Overall Score: 33%
Verification Level: PARTIALLY VERIFIED
Claims Verified: 2/6

Red Flags: 2
  1. LOW_VERIFICATION: Only 33% of claims verified
  2. CONTRADICTED_CLAIMS: 2 contradictions found
```

### Final Answer
```
Yes, moderate coffee consumption is considered safe for heart health, 
as recent research indicates that consuming 2-3 cups of coffee per day 
may have protective cardiovascular effects, including a 15% reduction 
in cardiovascular disease risk [study_2023.txt, 2023].

Earlier concerns from 2018 studies were likely due to uncontrolled 
confounding variables [meta_analysis_2022.txt, 2022]. The key distinction 
is between MODERATE consumption (beneficial) and HIGH consumption (potentially risky).
```

### Complete Output JSON
```json
{
  "query": "Is moderate coffee consumption safe for heart health?",
  "answer": "Yes, moderate coffee consumption is considered safe...",
  "retrieval": {
    "retrieval_source": "local",
    "num_documents": 4,
    "retrieval_confidence": 0.9
  },
  "reasoning": {
    "hops_used": 2,
    "confidence": 0.85
  },
  "uncertainty": {
    "confidence_score": 0.699,
    "confidence_level": "MODERATE"
  },
  "fact_check": {
    "overall_score": 0.33,
    "verification_level": "PARTIALLY VERIFIED"
  }
}
```

---

## Example 2: Out-of-Domain Query (Web Fallback)

### Query
```
"Is chicken healthy?"
```

### System Processing

**Step 1: Hybrid Retrieval - Tier 1 (Local)**
```
Searching local knowledge base...
Max similarity: 49% (below 60% threshold)
Coverage: Low (query terms not in documents)
Decision: ❌ Local insufficient → Fallback to Tier 2
```

**Step 2: Hybrid Retrieval - Tier 2 (Web)**
```
Triggering web search...
Query: "Is chicken healthy?"
Results: 5 web documents generated

Web Results:
1. Harvard Health - "Chicken, a lean protein source..."
2. WebMD - "Chicken is celebrated for lean protein..."
3. Environmental Nutrition - "Broader implications..."
4. Mayo Clinic - "Poultry benefits..."
5. American Heart Association - "Heart-healthy proteins..."
```

**Step 3: Simple Processing (Web Source)**
```
Source: WEB
Confidence: 75%
Processing: Simplified (skips multi-hop for web sources)
```

### Final Answer
```
🌐 Source: Web Search Results (Medium Confidence)
⚠️ Information retrieved from web search. Verify independently.

🟡 Retrieval Confidence: Medium (75%)

Yes, chicken can be part of a healthy diet, particularly when choosing 
lean cuts and healthy preparation methods. Chicken is a good source of 
lean protein, which is important for heart health. Skinless chicken breast 
is particularly low in saturated fat. However, preparation method matters - 
grilled, baked, or poached chicken is healthier than fried.
```

### Output Comparison

| Metric | Coffee (Local) | Chicken (Web) |
|--------|---------------|---------------|
| Source | LOCAL KB | WEB SEARCH |
| Confidence | 90% | 75% |
| Processing | Full Pipeline | Simplified |
| Components | All 6 enabled | Basic only |
| Response Time | 27.7s | ~10s |

**Key Insight:** System intelligently adapts processing based on source reliability.

---

## Example 3: Contradiction Detection

### Input Documents

**Document A (2018):**
```
"High coffee consumption increases cardiovascular risk by 23%."
```

**Document B (2023):**
```
"Moderate coffee consumption shows 15% reduction in cardiovascular risk."
```

### Detection Output
```json
{
  "contradictions": [
    {
      "claim_A": "High coffee consumption increases cardiovascular risk by 23%",
      "source_A": "study_2018.txt",
      "claim_B": "Moderate coffee consumption shows 15% reduction in cardiovascular risk",
      "source_B": "study_2023.txt",
      "explanation": "The contradiction arises from different consumption levels (high vs moderate) and improved methodology in later studies that controlled for confounding variables",
      "severity": "HIGH",
      "confidence": 0.95
    }
  ],
  "resolution": "The apparent contradiction is resolved by noting: (1) different consumption levels studied (high: >5 cups vs moderate: 2-3 cups), and (2) improved methodology in 2023 study with proper controls for smoking, diet, and exercise."
}
```

### Visual Report
```
======================================================================
CONTRADICTION ANALYSIS
======================================================================

🔍 CLAIM A:
   "High coffee consumption increases cardiovascular risk by 23%"
   Source: study_2018.txt

🔍 CLAIM B:
   "Moderate coffee consumption shows 15% reduction in cardiovascular risk"
   Source: study_2023.txt

⚠️ SEVERITY: HIGH
📊 CONFIDENCE: 95%

📝 EXPLANATION:
The contradiction arises from different consumption levels studied 
(high vs moderate) and improved methodology in later studies that 
controlled for confounding variables.

🔧 RESOLUTION:
Both findings are valid for their respective contexts:
- High consumption (>5 cups/day): May increase risk
- Moderate consumption (2-3 cups/day): Protective effects

The key is the DOSAGE and METHODOLOGY improvements.
======================================================================
```

---

## Example 4: Multi-Hop Reasoning

### Query
```
"Why do studies contradict each other about coffee?"
```

### Reasoning Process

**Hop 1: Extract Information**
```
Documents Used: study_2018.txt, meta_2022.txt

Extracted Information:
"2018 study found increased risk but didn't control for confounders. 
2022 meta-analysis identified this methodological gap."

Connection to Context:
Studies may contradict due to methodological differences.

Intermediate Conclusion:
Early studies had design flaws that affected their conclusions.

Sufficient Info: NO → Continue to Hop 2
```

**Hop 2: Build on Previous**
```
Documents Used: study_2023.txt

Extracted Information:
"2023 research with rigorous controls found protective effects."

Connection to Previous:
When confounders are properly controlled (2023), results differ 
from uncontrolled studies (2018).

Final Conclusion:
Contradictions explained by: (1) methodological improvements, 
(2) different consumption levels studied.

Sufficient Info: YES → Stop reasoning
```

### Final Synthesis
```json
{
  "reasoning_chain": [
    {
      "hop_number": 1,
      "extracted_info": "2018 study lacked confounder controls...",
      "intermediate_conclusion": "Methodological flaws affect results",
      "documents_used": ["study_2018.txt", "meta_2022.txt"]
    },
    {
      "hop_number": 2,
      "extracted_info": "2023 study with proper controls...",
      "final_conclusion": "Contradictions due to methodology + dosage",
      "documents_used": ["study_2023.txt"],
      "sufficient_info": true
    }
  ],
  "answer": "Studies contradict due to: (1) Methodological improvements - earlier studies didn't control for confounders like smoking and diet, while recent studies do. (2) Different consumption levels - high consumption (>5 cups) vs moderate (2-3 cups) have different effects. [meta_2022.txt, study_2023.txt]",
  "confidence": 0.85,
  "hops_used": 2
}
```

---

## Example 5: Synthetic Data Generation

### Generated Q&A Pairs

**Easy Difficulty:**
```json
{
  "question": "What did the 2023 study find about moderate coffee consumption?",
  "answer": "The 2023 study found that moderate coffee consumption (2-3 cups per day) is associated with a 15% reduction in cardiovascular disease risk when confounding factors are properly controlled. [study_2023.txt, 2023]",
  "difficulty": "easy",
  "requires_docs": 1,
  "source_docs": ["study_2023.txt"]
}
```

**Medium Difficulty:**
```json
{
  "question": "How does the recent research on the protective cardiovascular effects of coffee consumption compare to earlier findings?",
  "answer": "Recent 2023 research contrasts with earlier 2018 findings. The 2018 study found increased risk, but lacked controls for confounders. The 2023 study, with rigorous methodology, found protective effects for moderate consumption. The difference is explained by improved methodology and focus on moderate vs high consumption. [study_2018.txt, study_2023.txt, meta_2022.txt]",
  "difficulty": "medium",
  "requires_docs": 2,
  "reasoning_steps": [
    "Compare 2018 vs 2023 findings",
    "Identify methodological differences",
    "Note consumption level distinction"
  ]
}
```

**Hard Difficulty:**
```json
{
  "question": "Based on the evolution of coffee research from 2018-2023, what can we conclude about the role of confounding variables in cardiovascular studies?",
  "answer": "The evolution from 2018 to 2023 demonstrates the critical importance of controlling confounding variables. Early studies showing negative effects failed to account for factors like smoking, sugar intake, and exercise. The 2022 meta-analysis identified this as the primary source of contradictions. When the 2023 study implemented rigorous controls, it found opposite results - protective effects instead of harm. This illustrates how uncontrolled confounders can completely reverse study conclusions in cardiovascular research.",
  "difficulty": "hard",
  "requires_docs": 3,
  "reasoning_steps": [
    "Analyze 2018 methodology",
    "Review 2022 meta-analysis critique",
    "Compare with 2023 improvements",
    "Synthesize methodological lesson"
  ]
}
```

### Quality Assessment
```
Generated: 3 Q&A pairs
Processing Time: 27.65s

Quality Metrics:
  Overall Quality: 94.4% ⭐
  High-Quality Ratio: 100%
  
  Breakdown:
    - Completeness: 100%
    - Length Appropriateness: 100%
    - Reasoning Steps: 100% (for hard)
    - Coherence: 88%

Diversity Metrics:
  Overall Diversity: 81.9% ⭐
  Lexical Diversity: 45.7%
  Difficulty Distribution: 100%
  Length Variance: High
```

---

## Example Outputs for Documentation

### 1. Vector Search Output
```
Query: "coffee heart health"
Top 5 Results:

1. [Score: 0.776] study_2023.txt
   "Recent research suggests that moderate coffee intake may have 
    protective effects on heart health..."

2. [Score: 0.742] meta_2022.txt
   "A 2022 meta-analysis found no significant correlation between 
    coffee consumption and heart disease..."

3. [Score: 0.698] study_2018.txt
   "Coffee consumption has been linked to increased cardiovascular 
    risk in some studies from 2018..."
```

### 2. Multi-Factor Ranking Output
```
Query: "Coffee and cardiovascular health"
Ranking Weights: {similarity: 35%, credibility: 20%, recency: 20%, quality: 15%, alignment: 10%}

Ranked Results:

#1 - Final Score: 0.643
Source: study_2018.txt
  Similarity: 0.816 (81.6%)
  Credibility: 0.600 (60.0%)
  Recency: 0.497 (49.7%) [7 years old]
  Quality: 0.917 (91.7%)

#2 - Final Score: 0.712
Source: study_2023.txt
  Similarity: 0.776 (77.6%)
  Credibility: 0.600 (60.0%)
  Recency: 0.990 (99.0%) [recent]
  Quality: 0.917 (91.7%)

#3 - Final Score: 0.684
Source: meta_2022.txt
  Similarity: 0.742 (74.2%)
  Credibility: 0.600 (60.0%)
  Recency: 0.861 (86.1%) [3 years old]
  Quality: 0.950 (95.0%)
```

### 3. Knowledge Graph PageRank
```
Document Importance (PageRank):

#1. study_2023.txt: 0.481 (48.1%) ⭐ Most Important
    Why: Most recent, cited by others, high quality
    
#2. study_2018.txt: 0.260 (26.0%)
    Why: Historical significance, establishes baseline
    
#3. meta_2022.txt: 0.260 (26.0%)
    Why: Synthesizes multiple sources, high credibility
```

### 4. Uncertainty Report
```
======================================================================
UNCERTAINTY QUANTIFICATION REPORT
======================================================================

Query: Is moderate coffee consumption safe for heart health?

Overall Confidence: 69.9%
Confidence Level: MODERATE

──────────────────────────────────────────────────────────────
COMPONENT SCORES
──────────────────────────────────────────────────────────────

- Evidence Sufficiency: 65.0%
- Source Agreement: 95.0%
- Source Quality: 92.5%
- Contradiction Penalty: 30.1%

──────────────────────────────────────────────────────────────
UNCERTAINTY SOURCES
──────────────────────────────────────────────────────────────

1. Contradictory evidence (2 contradictions found)

──────────────────────────────────────────────────────────────
EVIDENCE GAPS
──────────────────────────────────────────────────────────────

1. Limited number of sources (3 found, recommend 5+)
2. Gap in temporal coverage (6 year span)

──────────────────────────────────────────────────────────────
SENSITIVITY ANALYSIS
──────────────────────────────────────────────────────────────

What confidence would change under different scenarios:

  Add High Quality Source: 73% (↑ 3%)
  Remove Contradictions: 85% (↑ 15%)
  Add Contradiction: 62% (↓ 8%)
  Double Evidence: 80% (↑ 10%)

======================================================================
```

### 5. Fact-Check Report
```
======================================================================
FACT-CHECK REPORT
======================================================================

Answer: Yes, moderate coffee consumption is considered safe...

──────────────────────────────────────────────────────────────
VERIFICATION SUMMARY
──────────────────────────────────────────────────────────────

Overall Verification: 33%
Verification Level: PARTIALLY VERIFIED

✅ Verified Claims: 2/6
⚠️ Unsupported Claims: 4
❌ Contradicted Claims: 0

──────────────────────────────────────────────────────────────
DETAILED CLAIM VERIFICATION
──────────────────────────────────────────────────────────────

1. ✅ Moderate coffee consumption is considered safe
   Status: VERIFIED
   Confidence: 85%
   Sources: study_2023.txt, meta_2022.txt

2. ✅ May have protective cardiovascular effects
   Status: VERIFIED
   Confidence: 80%
   Sources: study_2023.txt

3. ⚠️ Consuming 2-3 cups per day
   Status: UNSUPPORTED
   Confidence: 40%

... (additional claims)

──────────────────────────────────────────────────────────────
⚠️ FLAGGED CLAIMS (Needs Review)
──────────────────────────────────────────────────────────────

- Specific dosage recommendation (2-3 cups)
  Reason: UNSUPPORTED - Mentioned in documents but not strongly supported

======================================================================
```

---

## Performance Examples

### Query Response Times
```
Simple Query (local, no reasoning):
  "What year was the 2023 study published?"
  Response Time: 2.3s
  
Medium Query (local, basic reasoning):
  "Is coffee safe?"
  Response Time: 12.8s
  
Complex Query (local, full pipeline):
  "Why do studies contradict each other?"
  Response Time: 27.7s

Out-of-Domain (web fallback):
  "Is chicken healthy?"
  Response Time: 10.5s
```

### Token Usage Examples
```
Simple Query:
  Input: ~500 tokens
  Output: ~200 tokens
  Total: ~700 tokens
  Cost: ~$0.01

Complex Query:
  Input: ~2,000 tokens (multiple documents)
  Output: ~500 tokens
  Total: ~2,500 tokens
  Cost: ~$0.04

Full Pipeline (all features):
  Input: ~3,000 tokens
  Output: ~800 tokens
  Total: ~3,800 tokens
  Cost: ~$0.06
```

---

## Code Examples

### Example 1: Basic Usage
```python
from src.contextweaver_pipeline import ContextWeaverPipeline

# Initialize
pipeline = ContextWeaverPipeline()

# Ingest
pipeline.ingest_documents(['doc1.pdf', 'doc2.txt'])

# Query
result = pipeline.query("Your question here")

# Access answer
print(result['answer'])
```

**Output:**
```
Answer generated with citations
Confidence: 85%
Source: Local Knowledge Base
```

### Example 2: Advanced Usage with All Features
```python
# Query with full capabilities
result = pipeline.query(
    "Complex multi-document question",
    enable_multi_hop=True,
    enable_contradiction_detection=True,
    enable_uncertainty=True,
    enable_fact_checking=True,
    top_k=10
)

# Access detailed results
print(f"Hops: {result['reasoning']['hops_used']}")
print(f"Contradictions: {result['contradictions']['num_contradictions']}")
print(f"Confidence: {result['uncertainty']['confidence_score']:.2%}")

---

## Test Coverage Examples

### Unit Tests
```python
# Test document processor
python src/document_processor.py
# ✅ PASS - Chunking strategies working

# Test vector store
python src/vector_store.py
# ✅ PASS - Embeddings and search working

# Test reasoning
python src/reasoning_engine.py
# ✅ PASS - Multi-hop reasoning working
```

### Integration Tests
```python
# Test complete pipeline
python tests/test_all_components.py
# ✅ 12/12 tests passed (100%)
```

---

## Real-World Use Case Examples

### Use Case 1: Legal Document Analysis
```python
# Ingest legal documents
pipeline.ingest_documents([
    'main_contract.pdf',
    'amendment_2020.pdf',
    'amendment_2022.pdf',
    'bylaws.pdf'
])

# Query
result = pipeline.query(
    "Can the CEO approve this $3M acquisition?",
    enable_multi_hop=True,
    enable_contradiction_detection=True
)

# System will:
# 1. Find CEO authority clauses
# 2. Detect amendments that modify original contract
# 3. Check for contradictions
# 4. Provide citation chain showing precedence
```

### Use Case 2: Research Literature Review
```python
# Ingest papers
pipeline.ingest_documents([
    'transformer_2017.pdf',
    'bert_2018.pdf',
    'gpt3_2020.pdf'
])

# Query
result = pipeline.query(
    "How has the transformer architecture evolved?",
    enable_multi_hop=True
)

# System will:
# 1. Build temporal sequence
# 2. Track architectural innovations
# 3. Identify breakthrough moments
# 4. Synthesize evolution narrative
```

---

## Appendix: Complete Test Output

See `test_outputs/comprehensive_test_results.json` for complete machine-readable test results.

---

**Examples Documented By:** Sravan Kumar Kurapati  
**Test Date:** December 12, 2025  
**All examples verified with actual system output** ✅