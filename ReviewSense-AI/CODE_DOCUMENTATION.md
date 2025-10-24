# Code Documentation

**ReviewSense AI - Complete Code Reference**

This document explains the structure, functionality, and key components of all code files in the ReviewSense AI project.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Core Scripts](#core-scripts)
3. [Configuration Files](#configuration-files)
4. [Key Functions](#key-functions)
5. [Code Examples](#code-examples)

---

## Project Structure

```
ReviewSense-AI/
├── checkpoint_model_evaluation.py    # Main evaluation script
├── error_analysis.py                 # Error pattern analysis
├── demo_app.py                       # Interactive inference demo
├── requirements.txt                  # Python dependencies
├── ENVIRONMENT_SETUP.md             # Setup instructions
├── REPRODUCTION_INSTRUCTIONS.md     # Reproduction guide
├── TECHNICAL_REPORT.md              # Complete methodology
└── CODE_DOCUMENTATION.md            # This file
```

---

## Core Scripts

### Data Preparation Scripts

**Location**: `scripts/` directory

These scripts form the complete data pipeline from raw collection to tokenized training data.

---

#### 1. scripts/01_collect_data.py

**Purpose**: Multi-source data collection from Amazon, Yelp, and synthetic generation

**Process Flow:**
```
1. Load Amazon Polarity dataset (30K reviews)
2. Load Yelp Review Full dataset (20K reviews)
3. Generate synthetic aspect-annotated reviews (10K)
4. Combine all sources
5. Save to data/raw/
```

**Key Functions:**

```python
def collect_amazon_reviews(num_samples=30000):
    """
    Collects product reviews from Amazon Polarity dataset
    
    Args:
        num_samples (int): Number of reviews to collect
    
    Returns:
        pd.DataFrame: Reviews with columns:
            - review_id, text, rating, title, category, source
    
    Process:
        1. Load from HuggingFace datasets
        2. Sample uniformly across ratings
        3. Add metadata (source, review_id)
        4. Convert to DataFrame
    """

def collect_yelp_reviews(num_samples=20000):
    """
    Collects service reviews from Yelp dataset
    
    Args:
        num_samples (int): Number of reviews to collect
    
    Returns:
        pd.DataFrame: Reviews with 5-star ratings
    
    Note:
        Yelp provides richer rating distribution (1-5)
        compared to Amazon's binary sentiment
    """

def generate_synthetic_reviews(num_samples=10000):
    """
    Creates synthetic reviews with aspect annotations
    
    Args:
        num_samples (int): Number to generate
    
    Returns:
        pd.DataFrame: Reviews with aspect tags
    
    Why Synthetic Data:
        - Fills gaps in aspect coverage
        - Ensures balanced examples
        - Provides edge cases for training
    """
```

**Output Files:**
- `data/raw/amazon_reviews.csv` (30K reviews)
- `data/raw/yelp_reviews.csv` (20K reviews)
- `data/raw/synthetic_reviews.csv` (10K reviews)
- `data/raw/all_reviews_combined.csv` (60K reviews)

**Console Output:**
```
✅ Collected 30,000 Amazon reviews
✅ Collected 20,000 Yelp reviews
✅ Generated 10,000 synthetic reviews
📊 Total: 60,000 reviews
```

**Execution Time**: ~30 minutes

---

#### 2. scripts/02_preprocess_data.py

**Purpose**: Text cleaning, quality filtering, and dataset balancing

**Process Flow:**
```
60,000 raw reviews
    ↓ Text cleaning
58,209 valid reviews (97% retained)
    ↓ Quality filtering
58,914 quality reviews (98.2% retained)
    ↓ Duplicate removal
58,209 unique reviews
    ↓ Rating balancing
33,418 balanced reviews
```

**Key Functions:**

```python
def clean_text(text):
    """
    Cleans raw review text
    
    Args:
        text (str): Raw review text
    
    Returns:
        str: Cleaned text
    
    Operations:
        1. Remove HTML tags
        2. Strip special characters
        3. Normalize whitespace
        4. Fix common OCR errors
        5. Standardize punctuation
    
    Example:
        Input:  "Great product!!!   <b>Highly</b> recommend."
        Output: "Great product! Highly recommend."
    """

def apply_quality_filters(df):
    """
    Filters low-quality reviews
    
    Args:
        df (pd.DataFrame): Reviews dataframe
    
    Returns:
        pd.DataFrame: Filtered reviews
    
    Filter Criteria:
        - Minimum length: 20 characters
        - Minimum words: 5
        - Maximum length: 5000 characters
        - Must contain letter characters
        - Remove spam patterns
    
    Why These Filters:
        Too short: Not enough context
        Too long: Truncation issues
        Spam: Affects model quality
    """

def balance_ratings(df):
    """
    Balances rating distribution to prevent bias
    
    Args:
        df (pd.DataFrame): Reviews with imbalanced ratings
    
    Returns:
        pd.DataFrame: Balanced dataset
    
    Strategy:
        - Cap extreme ratings (1★, 5★) at 8,000 each
        - Retain all middle ratings (2-4★)
        - Use stratified undersampling
    
    Why Balance:
        Prevents model from learning rating bias
        Ensures all sentiments represented
    """
```

**Output Files:**
- `data/processed/reviews_cleaned.csv` (33,418 reviews)
- `data/processed/preprocessing_stats.json` (statistics)

**Statistics Generated:**
```json
{
  "initial_reviews": 60000,
  "after_cleaning": 58209,
  "after_filtering": 58914,
  "final_count": 33418,
  "avg_text_length": 465,
  "avg_word_count": 84
}
```

**Execution Time**: ~10 minutes

---

#### 3. scripts/03_create_training_data.py

**Purpose**: Format data for instruction tuning with multi-task examples

**Process Flow:**
```
33,418 cleaned reviews
    ↓ Create aspect analysis task (10K)
    ↓ Create summarization task (5K)
    ↓ Create sentiment task (5K)
20,000 training examples
    ↓ Split 70/15/15
14K train | 3K val | 3K test
```

**Key Functions:**

```python
def create_aspect_analysis_examples(reviews, num_samples=10000):
    """
    Creates aspect-based sentiment analysis examples
    
    Args:
        reviews (pd.DataFrame): Cleaned reviews
        num_samples (int): Number of examples to create
    
    Returns:
        list: Training examples in instruction format
    
    Format:
        {
            "instruction": "Analyze this review for aspect sentiments...",
            "input": "[review text]",
            "response": "Aspects identified:\n- Service: positive\n..."
        }
    
    Why This Task:
        Teaches model to identify specific features
        mentioned in reviews (food, service, price, etc.)
    """

def create_summarization_examples(reviews, num_samples=5000):
    """
    Creates review summarization examples
    
    Args:
        reviews (pd.DataFrame): Cleaned reviews
        num_samples (int): Number of examples
    
    Returns:
        list: Summarization training examples
    
    Format:
        {
            "instruction": "Summarize this review concisely...",
            "input": "[full review]",
            "response": "[generated summary]"
        }
    
    Summary Generation:
        Uses extractive approach (first/last sentences)
        plus key phrase extraction for training labels
    """

def create_sentiment_examples(reviews, num_samples=5000):
    """
    Creates sentiment classification examples
    
    Args:
        reviews (pd.DataFrame): Reviews with ratings
        num_samples (int): Number to create
    
    Returns:
        list: Sentiment classification examples
    
    Format:
        {
            "instruction": "What is the sentiment?",
            "input": "[review]",
            "response": "Positive/Negative/Neutral"
        }
    
    Mapping:
        1-2 stars: Negative
        3 stars: Neutral
        4-5 stars: Positive
    """
```

**Output Files:**
- `data/splits/train.jsonl` (14,000 examples, ~15MB)
- `data/splits/val.jsonl` (3,000 examples, ~3.2MB)
- `data/splits/test.jsonl` (3,000 examples, ~3.2MB)
- `data/splits/split_stats.json` (split statistics)

**Task Distribution:**
```
Aspect Analysis:  10,000 examples (50%)
Summarization:     5,000 examples (25%)
Sentiment:         5,000 examples (25%)
```

**Execution Time**: ~15 minutes

---

#### 4. scripts/04_tokenize_data.py

**Purpose**: Tokenize data for TinyLlama model training

**Process Flow:**
```
JSONL files (train/val/test)
    ↓ Format with chat template
    ↓ Tokenize with TinyLlama tokenizer
    ↓ Create attention masks
    ↓ Add labels for loss computation
Tokenized datasets ready for training
```

**Key Functions:**

```python
def format_with_chat_template(example):
    """
    Formats examples using TinyLlama's chat template
    
    Args:
        example (dict): Raw example with instruction/input/response
    
    Returns:
        dict: Formatted with proper chat markers
    
    Template:
        <|system|>
        You are a helpful assistant...
        <|user|>
        {instruction}
        {input}
        <|assistant|>
        {response}
    
    Why This Format:
        TinyLlama-Chat is pre-trained with this format
        Maintains consistency with base model training
    """

def tokenize_function(examples, tokenizer, max_length=512):
    """
    Tokenizes formatted examples
    
    Args:
        examples (dict): Batch of formatted examples
        tokenizer: HuggingFace tokenizer
        max_length (int): Maximum sequence length
    
    Returns:
        dict: Tokenized with input_ids, attention_mask, labels
    
    Process:
        1. Tokenize text with truncation
        2. Create attention masks (1 for real, 0 for padding)
        3. Copy input_ids to labels for next-token prediction
        4. Set padding token labels to -100 (ignored in loss)
    
    Why max_length=512:
        Balance between context and memory usage
        Most reviews fit within this limit
    """
```

**Output Files:**
- `data/processed/tokenized_datasets/train/` (PyArrow dataset)
- `data/processed/tokenized_datasets/validation/` (PyArrow dataset)
- `data/processed/tokenized_datasets/test/` (PyArrow dataset)

**Token Statistics:**
```
Average tokens per example: ~380
Max tokens: 512 (truncated)
Vocabulary size: 32,000 (TinyLlama)
```

**Execution Time**: ~5 minutes

---

### Evaluation and Analysis Scripts

#### 1. checkpoint_model_evaluation.py

**Purpose**: Comprehensive model evaluation with checkpoint support

**Key Features:**
- Evaluates 4 models (baseline + 3 fine-tuned)
- Calculates ROUGE and BLEU metrics
- Saves checkpoints to resume interrupted runs
- Generates comparison visualizations

**Main Components:**

```python
# Configuration Class
class EvaluationConfig:
    """
    Stores all configuration parameters for evaluation
    
    Attributes:
        BASE_DIR: Project root directory
        MODELS_DIR: Location of trained models
        RESULTS_DIR: Output directory for results
        CHECKPOINT_DIR: Directory for cached predictions
        BASE_MODEL_NAME: HuggingFace model identifier
        EXPERIMENTS: Dictionary of model configurations
        MAX_EVAL_SAMPLES: Number of test samples (200)
        DEVICE: Compute device (mps/cuda/cpu)
    """
```

**Key Functions:**

```python
def save_checkpoint(model_name, predictions, metrics):
    """
    Saves evaluation results to avoid recomputation
    
    Args:
        model_name (str): Name of model (baseline/exp1/exp2/exp3)
        predictions (list): Generated summaries for test samples
        metrics (dict): ROUGE and BLEU scores
    
    Returns:
        None. Saves pickle file to CHECKPOINT_DIR
    """

def load_checkpoint(model_name):
    """
    Loads previously saved evaluation results
    
    Args:
        model_name (str): Name of model to load
    
    Returns:
        dict: Contains predictions and metrics, or None if not found
    """

def load_model_and_tokenizer(experiment_name, use_baseline=False):
    """
    Loads TinyLlama base model with optional LoRA adapter
    
    Args:
        experiment_name (str): Model name (exp1/exp2/exp3)
        use_baseline (bool): If True, skip LoRA loading
    
    Returns:
        tuple: (model, tokenizer)
    
    Process:
        1. Load base TinyLlama-1.1B-Chat
        2. Configure tokenizer padding
        3. Load LoRA adapter if not baseline
        4. Merge adapter into base model
        5. Set to evaluation mode
    """

def generate_predictions(model, tokenizer, eval_data, experiment_name):
    """
    Generates summaries for all test samples
    
    Args:
        model: Loaded PyTorch model
        tokenizer: HuggingFace tokenizer
        eval_data: Dataset object with test reviews
        experiment_name (str): Model identifier for logging
    
    Returns:
        list: Dictionaries containing:
            - review_id: Sample index
            - original_text: Full review text
            - rating: Star rating (1-5)
            - prediction: Generated summary
            - reference: First 200 chars (for ROUGE)
    
    Generation Parameters:
        - max_new_tokens: 200
        - temperature: 0.7 (sampling randomness)
        - top_p: 0.9 (nucleus sampling)
        - do_sample: True (enable sampling)
    """

def calculate_metrics(predictions):
    """
    Computes evaluation metrics for predictions
    
    Args:
        predictions (list): Model outputs from generate_predictions
    
    Returns:
        dict: Metrics dictionary with keys:
            - rouge1: Unigram overlap
            - rouge2: Bigram overlap
            - rougeL: Longest common subsequence
            - bleu: N-gram precision
    
    Metrics Explanation:
        ROUGE-1: Word-level similarity (0-1, higher better)
        ROUGE-2: Phrase-level coherence (0-1, higher better)
        ROUGE-L: Sentence fluency (0-1, higher better)
        BLEU: Overall generation quality (0-1, higher better)
    """

def create_comparison_plots(results):
    """
    Creates bar charts comparing all models
    
    Args:
        results (dict): Evaluation results for all models
    
    Output:
        Saves model_comparison.png with 4 subplots:
            - ROUGE-1 comparison
            - ROUGE-2 comparison
            - ROUGE-L comparison
            - BLEU comparison
    """

def generate_evaluation_report(results, eval_data):
    """
    Generates markdown report with tables and analysis
    
    Args:
        results (dict): All model evaluation results
        eval_data: Test dataset
    
    Output:
        Saves evaluation_report.md containing:
            - Model configurations table
            - Performance metrics table
            - Best model identification
            - Sample predictions
    """
```

**Usage Example:**

```python
# Run complete evaluation
python checkpoint_model_evaluation.py

# What happens:
# 1. Creates output directories
# 2. Loads test dataset (200 samples)
# 3. For each model (baseline, exp1, exp2, exp3):
#    a. Check if checkpoint exists
#    b. If yes: load cached results
#    c. If no: load model, generate predictions, calculate metrics
#    d. Save checkpoint
# 4. Create comparison visualizations
# 5. Generate markdown report
```

**Output Files:**
- `evaluation_results/evaluation_report.md`
- `evaluation_results/model_comparison.png`
- `evaluation_results/evaluation_results.json`
- `evaluation_checkpoints/[model]_checkpoint.pkl`

---

### 2. error_analysis.py

**Purpose**: Analyzes failure modes and error patterns in model predictions

**Key Features:**
- Categorizes errors into 7 types
- Identifies worst-performing cases
- Compares with baseline model
- Generates actionable recommendations

**Main Components:**

```python
BEST_MODEL = "exp2"  # Best performing model from evaluation

def load_predictions(model_name):
    """
    Loads predictions from evaluation checkpoint
    
    Args:
        model_name (str): Model identifier (exp2)
    
    Returns:
        list: Predictions with metrics
    
    Checks:
        - Verifies checkpoint file exists
        - Loads pickled data
        - Returns predictions and performance metrics
    """

def analyze_error_patterns(predictions):
    """
    Categorizes errors in predictions
    
    Args:
        predictions (list): Model outputs
    
    Returns:
        tuple: (error_categories dict, well_performing list)
    
    Error Categories:
        - truncated: Cuts off mid-sentence
        - verbatim_copy: Copies review instead of summarizing
        - too_short: Less than 10 words
        - repetitive: Excessive word repetition
        - format_issues: Contains prompt markers
    
    Detection Logic:
        Truncated: prediction doesn't end with '.!?'
        Verbatim: >25 of first 30 words from original
        Too Short: <10 words
        Repetitive: Any word appears >5 times
        Format Issues: Contains 'Review:', 'Task:', etc.
    """

def identify_failure_cases(predictions, num_examples=15):
    """
    Selects worst predictions for detailed analysis
    
    Args:
        predictions (list): All model predictions
        num_examples (int): Number of cases to analyze
    
    Returns:
        list: Worst-performing predictions with quality scores
    
    Quality Scoring:
        - Start with score 1.0
        - Penalize truncation: score * 0.3
        - Penalize too short: score * 0.4
        - Penalize format issues: score * 0.2
        - Penalize low unique word ratio
    
    Selection:
        - Sort by quality score (ascending)
        - Take bottom N examples
    """

def generate_error_report(predictions, error_categories, failure_cases, well_performing):
    """
    Creates comprehensive error analysis markdown report
    
    Args:
        predictions (list): All predictions
        error_categories (dict): Categorized errors
        failure_cases (list): Worst examples
        well_performing (list): Best examples
    
    Output:
        Saves error_analysis_report.md with:
            - Performance comparison vs baseline
            - Error distribution table
            - Detailed analysis of each error type
            - 10 failure case examples
            - 5 success case examples
            - Immediate and long-term recommendations
    """

def create_visualizations(error_categories, well_performing, predictions):
    """
    Creates error distribution charts
    
    Args:
        error_categories (dict): Error counts by type
        well_performing (list): Successful predictions
        predictions (list): All predictions
    
    Output:
        Saves error_distribution.png with 2 subplots:
            1. Bar chart: Error type frequencies
            2. Pie chart: Well-performing vs issues
    """
```

**Usage Example:**

```python
# Run error analysis
python error_analysis.py

# Process:
# 1. Load exp2 predictions from checkpoint
# 2. Analyze error patterns
# 3. Identify 15 worst cases
# 4. Create visualizations
# 5. Generate comprehensive report
```

**Output Files:**
- `error_analysis_results/error_analysis_report.md`
- `error_analysis_results/error_distribution.png`
- `error_analysis_results/error_analysis.json`

---

### 3. demo_app.py

**Purpose**: Interactive inference pipeline for review summarization

**Key Features:**
- Interactive menu-driven interface
- Pre-loaded sample reviews
- Custom review input
- Post-processing cleanup
- Batch demo mode

**Main Components:**

```python
BEST_MODEL = "exp2"  # Best performing model
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

SAMPLE_REVIEWS = [
    {
        "id": 1,
        "text": "Positive review text...",
        "rating": 5,
        "type": "Positive"
    },
    # ... 2 more samples
]

def clean_summary(text):
    """
    Post-processes model output to remove formatting issues
    
    Args:
        text (str): Raw model generation
    
    Returns:
        str: Cleaned summary
    
    Cleaning Steps:
        1. Remove format markers ('Review:', 'Summary:', etc.)
        2. Strip bullet points and dashes
        3. Remove excessive newlines
        4. Remove quotes at start/end
        5. Truncate at last complete sentence if needed
        6. Remove multiple spaces
        7. Remove remaining format markers at end
    
    Why Needed:
        Model trained on structured prompts, sometimes
        includes template markers in output. This function
        cleans output for production use.
    """

def load_model(model_name=BEST_MODEL):
    """
    Loads fine-tuned model for inference
    
    Args:
        model_name (str): Model identifier (default: exp2)
    
    Returns:
        tuple: (model, tokenizer)
    
    Process:
        1. Load TinyLlama tokenizer
        2. Configure padding token
        3. Load base model (float16 for efficiency)
        4. Load LoRA adapter from models directory
        5. Merge adapter into base model
        6. Set to evaluation mode
    
    Memory Usage:
        - Base model: ~2.2GB in float16
        - With adapter: ~2.3GB
        - Inference: ~3GB peak
    """

def generate_summary(model, tokenizer, review_text, temperature=0.7, max_tokens=200):
    """
    Generates summary for a single review
    
    Args:
        model: Loaded PyTorch model
        tokenizer: HuggingFace tokenizer
        review_text (str): Review to summarize
        temperature (float): Sampling randomness (0.0-1.0)
        max_tokens (int): Maximum new tokens to generate
    
    Returns:
        str: Cleaned summary
    
    Prompt Template:
        <|system|>
        You are a helpful assistant that analyzes customer reviews.
        <|user|>
        Review: {review_text}
        Task: Generate a concise summary.
        <|assistant|>
        [Model generates here]
    
    Generation Parameters:
        - max_new_tokens: 200 (limit output length)
        - temperature: 0.7 (control randomness)
        - top_p: 0.9 (nucleus sampling)
        - do_sample: True (enable sampling)
    
    Post-Processing:
        - Extracts text after <|assistant|> marker
        - Applies clean_summary() function
        - Returns final cleaned output
    """

def display_menu():
    """
    Shows interactive menu options
    
    Options:
        1. Positive 5-star review sample
        2. Negative 2-star review sample
        3. Neutral 3-star review sample
        4. Custom review input
        5. Exit
    """

def run_demo():
    """
    Main interactive demo loop
    
    Process:
        1. Load model once (reuse for all predictions)
        2. Display menu
        3. Get user choice
        4. Load/accept review text
        5. Generate summary
        6. Display result
        7. Ask to continue or exit
    
    User Flow:
        Select option → See review → Wait for generation →
        See summary → Try another? → Exit
    """

def run_batch_demo():
    """
    Automatic demo of all 3 sample reviews
    
    Purpose:
        - Quick demonstration mode
        - Perfect for video recording
        - No user interaction needed
    
    Process:
        1. Load model
        2. For each sample review:
           - Display review text
           - Generate summary
           - Display summary
        3. Show completion message
    """
```

**Usage Examples:**

```python
# Interactive mode
python demo_app.py
# Then select options 1, 3, 4, 5

# Batch mode (for video/screenshots)
python demo_app.py --batch
# Runs all 3 examples automatically

# Programmatic usage in code
from demo_app import load_model, generate_summary

model, tokenizer = load_model("exp2")
review = "Great product, fast shipping!"
summary = generate_summary(model, tokenizer, review)
print(summary)
```

---

## Configuration Files

### requirements.txt

**Purpose**: Lists all Python package dependencies with versions

**Key Packages:**

```
torch==2.1.0                 # Deep learning framework
transformers==4.35.0         # HuggingFace models
peft==0.7.0                  # LoRA implementation
datasets==2.15.0             # Data loading
evaluate==0.4.1              # Metrics computation
rouge-score==0.1.2           # ROUGE metric
nltk==3.8.1                  # NLP utilities
sacrebleu==2.3.1             # BLEU metric
pandas==2.1.3                # Data manipulation
numpy==1.24.3                # Numerical computing
matplotlib==3.8.2            # Visualization
seaborn==0.13.0              # Statistical plots
tqdm==4.66.1                 # Progress bars
```

**Installation:**
```bash
pip install -r requirements.txt
```

---

## Key Functions Reference

### Model Loading

```python
def load_model_and_tokenizer(model_name, use_baseline=False):
    """
    Standard model loading pattern used across all scripts
    
    Returns base TinyLlama + optional LoRA adapter
    Handles device placement (CPU/MPS/CUDA)
    Configures tokenizer padding
    """
```

### Prediction Generation

```python
def generate_predictions(model, tokenizer, data):
    """
    Batch prediction with progress tracking
    
    Uses torch.no_grad() for efficiency
    Applies consistent generation parameters
    Extracts assistant responses from chat format
    """
```

### Metrics Calculation

```python
def calculate_metrics(predictions, references):
    """
    Computes ROUGE-1, ROUGE-2, ROUGE-L, BLEU
    
    Uses HuggingFace evaluate library
    Returns dictionary of scores
    """
```

### Checkpoint Management

```python
def save_checkpoint(name, data):
    """Saves results to pickle file"""

def load_checkpoint(name):
    """Loads results from pickle file"""
```

---

## Code Style Guidelines

**Followed Throughout Project:**

1. **Docstrings**: All functions have clear docstrings
2. **Type Hints**: Used where helpful for clarity
3. **Comments**: Explain WHY, not WHAT
4. **Naming**: Descriptive variable/function names
5. **Constants**: ALL_CAPS for configuration values
6. **Error Handling**: Try-except where failures expected

**Example:**

```python
def process_review(review_text: str, max_length: int = 512) -> str:
    """
    Preprocesses review text for model input
    
    Args:
        review_text: Raw review string
        max_length: Maximum characters to retain
    
    Returns:
        Cleaned and truncated review text
    
    Note:
        Preserves sentence boundaries when truncating
    """
    # Remove HTML tags (user-generated content may contain markup)
    clean_text = remove_html_tags(review_text)
    
    # Truncate at sentence boundary if needed
    if len(clean_text) > max_length:
        clean_text = truncate_at_sentence(clean_text, max_length)
    
    return clean_text
```

---

## Common Code Patterns

### Pattern 1: Model Loading with Error Handling

```python
def safe_load_model(model_path):
    """Loads model with comprehensive error handling"""
    try:
        model = load_model(model_path)
        print(f"✅ Model loaded from {model_path}")
        return model
    except FileNotFoundError:
        print(f"❌ Model not found at {model_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
```

### Pattern 2: Progress Tracking

```python
from tqdm import tqdm

for item in tqdm(dataset, desc="Processing"):
    result = process_item(item)
    results.append(result)
```

### Pattern 3: Result Saving

```python
import json
from pathlib import Path

def save_results(results, output_dir):
    """Saves results in multiple formats"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON for programmatic access
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Markdown for human reading
    with open(output_dir / "results.md", 'w') as f:
        f.write(format_as_markdown(results))
```

---

## Performance Considerations

### Memory Management

```python
# Use float16 for efficiency
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Halves memory usage
    device_map="auto"
)

# Clear cache after each model
del model
torch.mps.empty_cache()  # or torch.cuda.empty_cache()
```

### Inference Optimization

```python
# Disable gradient computation
with torch.no_grad():
    outputs = model.generate(...)

# Use efficient attention
model.config.use_cache = True
```

---

## Testing Your Code

### Unit Test Example

```python
def test_clean_summary():
    """Test post-processing function"""
    
    # Test case 1: Remove format markers
    input_text = "Review: Great product! **Outlook**: Positive"
    expected = "Great product!"
    assert clean_summary(input_text) == expected
    
    # Test case 2: Handle truncation
    input_text = "This is a sentence without ending"
    output = clean_summary(input_text)
    assert output.endswith('.')  # Should add period
    
    print("✅ All tests passed")

test_clean_summary()
```

---

## Debugging Tips

### Enable Verbose Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug(f"Processing {len(samples)} samples")
logger.info(f"Model loaded: {model_name}")
```

### Check Intermediate Outputs

```python
# Print shapes
print(f"Input shape: {inputs.shape}")
print(f"Output shape: {outputs.shape}")

# Inspect tokenization
print(f"Tokens: {tokenizer.convert_ids_to_tokens(input_ids)}")

# Sample predictions
print(f"Sample output: {predictions[0][:100]}...")
```

---

## Code Modification Guide

### To Change Evaluation Sample Size

**File**: `checkpoint_model_evaluation.py`

```python
# Line ~37
MAX_EVAL_SAMPLES = 200  # Change to desired number
```

### To Use Different Model

**File**: `demo_app.py`

```python
# Line ~26
BEST_MODEL = "exp2"  # Change to "exp1" or "exp3"
```

### To Adjust Generation Parameters

**File**: `demo_app.py`, function `generate_summary()`

```python
# Line ~180
outputs = model.generate(
    **inputs,
    max_new_tokens=200,      # Increase for longer summaries
    temperature=0.7,         # Lower for less randomness
    top_p=0.9,              # Adjust nucleus sampling
    # ... other parameters
)
```

---

## Additional Resources

- **HuggingFace Transformers Docs**: https://huggingface.co/docs/transformers/
- **PEFT Documentation**: https://github.com/huggingface/peft
- **PyTorch Docs**: https://pytorch.org/docs/
- **ROUGE Metric**: https://github.com/google-research/google-research/tree/master/rouge

---

**Document Version**: 1.0  
**Last Updated**: October 2025  
**Maintained By**: Sravan Kumar Kurapati
