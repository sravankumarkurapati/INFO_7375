# Sample ContextWeaver Outputs

Quick reference of typical system outputs.

## Query 1: Coffee Safety (Local KB)

**Input:** "Is moderate coffee consumption safe for heart health?"

**Output:**
```
🟢 Retrieval Confidence: High (90%)

Yes, moderate coffee consumption (2-3 cups/day) is considered safe for 
heart health, as recent research indicates protective cardiovascular effects.

Confidence: 69.9% (MODERATE)
Sources: 3 documents analyzed
Contradictions: 2 found (explained)
Hops: 1 reasoning step
```

## Query 2: Chicken (Web Fallback)

**Input:** "Is chicken healthy?"

**Output:**
```
🟡 Retrieval Confidence: Medium (75%)
⚠️ Information retrieved from web search. Verify independently.

Yes, chicken is a healthy protein source when prepared properly. Choose 
lean cuts and avoid fried preparation for best health benefits.

Confidence: 64% (MEDIUM - Web Source)
Sources: 5 web results
```

## Query 3: Evolution Analysis

**Input:** "How has coffee research evolved from 2018 to 2023?"

**Output:**
```
📊 Temporal Analysis (3 time points)

2018: Early studies showed increased risk (uncontrolled confounders)
2022: Meta-analysis identified methodological gaps
2023: Improved studies show protective effects

Evolution: Understanding shifted from "harmful" to "beneficial for moderate consumption"
Turning Point: 2022 meta-analysis explaining confounders

Confidence: 78% (HIGH)
```
```

---

## ✅ **REQUIREMENT 3: Setup Instructions**

**Status:** ✅ COMPLETE

- ✅ `docs/SETUP.md` (created earlier - comprehensive setup guide)

---

## ✅ **REQUIREMENT 4: Testing Scripts**

**Status:** ✅ COMPLETE
```
tests/
└── test_all_components.py  ✅ (12 comprehensive tests)

Test Results:
✅ 12/12 tests passed (100%)
⏱️ 136.68s total time
📄 Results saved to test_outputs/comprehensive_test_results.json
```

---

## ✅ **REQUIREMENT 5: Example Outputs**

**Status:** ✅ COMPLETE (just created!)
```
examples/
├── example_query_coffee.json      ✅ (Local KB example)
├── example_query_chicken.json     ✅ (Web fallback example)
└── sample_outputs.md              ✅ (Quick reference)

docs/
└── EXAMPLES.md                     ✅ (Comprehensive examples)