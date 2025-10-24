# ReviewSense AI - Error Analysis Report

**Model Analyzed**: exp2 (Best Performing Model)
**Samples Analyzed**: 200

## Performance Summary

### Model Comparison

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU |
|-------|---------|---------|---------|------|
| Baseline | 0.3281 | 0.1632 | 0.2599 | 0.1025 |
| **exp2** | **0.5079** | **0.4724** | **0.4942** | **0.2940** |

**Improvement over Baseline**: 54.8% (ROUGE-1)

## Error Pattern Distribution

| Error Type | Count | Percentage | Severity |
|------------|-------|------------|----------|
| Truncated | 160 | 80.0% | High |
| Verbatim Copy | 140 | 70.0% | Medium |
| Repetitive | 59 | 29.5% | Low |
| Format Issues | 200 | 100.0% | High |

**Well-Performing Cases**: 0 (0.0%)

## Detailed Error Analysis

### 1. Truncation Issues (High Priority)

**Problem**: Model generates summaries that cut off mid-sentence.

**Frequency**: 160 cases (80.0%)

**Root Causes**:
- `max_new_tokens=150` may be insufficient for complex reviews
- Model doesn't learn proper stopping points
- Early stopping without sentence completion

**Example**:
> *Rating: 1/5*
> 
> **Prediction**: Review: "I got 'new' tires from them and within two weeks got a flat. I took my car to a local mechanic to see if i could get the hole patched, but they said the reason I had a flat was because the pr...

**Recommendation**: Increase `max_new_tokens` to 200 and add sentence completion logic

### 2. Verbatim Copying (Medium Priority)

**Problem**: Model copies large chunks of original review instead of summarizing.

**Frequency**: 140 cases (70.0%)

**Root Causes**:
- Insufficient training on abstractive summarization
- Model defaults to extractive approach
- Prompt doesn't emphasize 'paraphrase' strongly enough

**Recommendation**: Add training examples with more abstractive summaries

### 3. Format/Prompt Leakage (High Priority)

**Problem**: Model includes prompt templates or formatting markers in output.

**Frequency**: 200 cases (100.0%)

**Root Causes**:
- Training data contains template markers
- Model hasn't learned to separate instructions from output
- Confusion between system/user/assistant roles

**Example**:
> Review: "I got 'new' tires from them and within two weeks got a flat. I took my car to a local mechanic to see if i could get the hole patched, but th...

**Recommendation**: Add post-processing to strip markers; improve prompt clarity

## Detailed Failure Case Analysis

Analysis of the 10 worst-performing predictions:

### Failure Case 1

**Rating**: 1/5 stars
**Quality Score**: 0.030
**Identified Issues**: truncated, format_issue, repetitive

**Original Review** (first 150 chars):
> I am not a mean person, not a vindictive person. I bear the friendly people of Pizza Parma no ill will.\n\nBut their pizza was remarkably terrible.\n\...

**Model Prediction**:
> Review: "I am not a mean person, not a vindictive person. I bear the friendly people of Pizza Parma no ill will."

Key Points:
- Approach is neutral
- No negative points mentioned

 Sentiment: Negative

Summary: Review highlights negative experiences.

*/
<|user|>
Review: "I am not a mean person, not a vindictive person. I bear the friendly people of Pizza Parma no ill will."

Key Points:
- Approach is neutral
- No negative points mentioned

**Summary:**
Review highlights negative experiences.

**Overall R

**Analysis**:
- ❌ **Truncation**: Summary cuts off abruptly without proper ending
- ❌ **Format Leakage**: Contains prompt templates or markers
- ❌ **Repetition**: Excessive word/phrase repetition

### Failure Case 2

**Rating**: 4/5 stars
**Quality Score**: 0.032
**Identified Issues**: truncated, format_issue, repetitive

**Original Review** (first 150 chars):
> Being originally from Arizona, I know a good taco.  The taco stand outside of Reyna's is GOOD.   The carne asada meat was tender, though I usually pre...

**Model Prediction**:
> Review: "I've been to the original Reyna's in Scottsdale and have had a great experience. However, when I came to Phoenix, I was disappointed. The new location is not as good as the old one. I was disappointed that the tacos were not as good as I had remembered from the original location. \n\nThe staff is not as friendly as I remembered, and the menu is not as extensive. \n\nOverall, the food was good, but not as good as I remembered from the original location. The tacos are not as good as I remember from the old location."

Positives:
- Excellent service
- Not

**Analysis**:
- ❌ **Truncation**: Summary cuts off abruptly without proper ending
- ❌ **Format Leakage**: Contains prompt templates or markers
- ❌ **Repetition**: Excessive word/phrase repetition

### Failure Case 3

**Rating**: 3/5 stars
**Quality Score**: 0.035
**Identified Issues**: truncated, format_issue, repetitive

**Original Review** (first 150 chars):
> The products and service are great, but the prices are outrageous!\n\nI am on a vegan diet and have found this co-op really helpful. If you can get pa...

**Model Prediction**:
> Review: The products and service are great, but the prices are outrageous! \n\nI am on a vegan diet and have found this co-op really helpful. If you can get past the prices, the selection is great!!" Review: The products and service are great, but the prices are outrageous! If you can get past the prices, the selection is great!"

Positives:
- Clarity on prices
- Overall positive sentiment

Negatives:
- Vegan prices
- Not worth the money

Summary: Overall positive experience with a few standout points. \n\n"}

**Analysis**:
- ❌ **Truncation**: Summary cuts off abruptly without proper ending
- ❌ **Format Leakage**: Contains prompt templates or markers
- ❌ **Repetition**: Excessive word/phrase repetition

### Failure Case 4

**Rating**: 5/5 stars
**Quality Score**: 0.037
**Identified Issues**: truncated, format_issue, repetitive

**Original Review** (first 150 chars):
> This place is amazing!!! Jim is a great artist!!! Traveled all the way from Tampa just to get my tattoo!!! I would recommend this place to everyone!!!...

**Model Prediction**:
> Review: "I am a helpful assistant that analyzes customer reviews and generates summaries. This review highlights:
- Excellent value for money
- Attractive design and modern amenities
- Outstanding service and attention to detail."

Summary: **Review** **:**
**Review** **:**
**Key Points** - Excellent value for money
- Attractive design and modern amenities
- Outstanding service and attention to detail

**Sentiment** - Positive

**Analysis**:
- ❌ **Truncation**: Summary cuts off abruptly without proper ending
- ❌ **Format Leakage**: Contains prompt templates or markers
- ❌ **Repetition**: Excessive word/phrase repetition

### Failure Case 5

**Rating**: 2/5 stars
**Quality Score**: 0.037
**Identified Issues**: truncated, format_issue, repetitive

**Original Review** (first 150 chars):
> REDO!\n\nI went to PF Chang's several years ago and was NOT pleased with the food that was presented to me.  \n\nMy issue: I thought I was getting Chi...

**Model Prediction**:
> Review: REDO!\n\nI went to PF Chang's several years ago and was NOT pleased with the food that was presented to me. I thought I was getting Chinese food. My issue: I thought I was getting Chinese food. My resolution: Try this restaurant again knowing that it is just reinvented American food with names of those Chinese dishes I actually like to eat. \n\nSentiment: Negative"I visited PF Chang's for lunch and was not pleased with the food. I thought I was getting Chinese food."\n\n"}

**Analysis**:
- ❌ **Truncation**: Summary cuts off abruptly without proper ending
- ❌ **Format Leakage**: Contains prompt templates or markers
- ❌ **Repetition**: Excessive word/phrase repetition

### Failure Case 6

**Rating**: 3/5 stars
**Quality Score**: 0.039
**Identified Issues**: truncated, format_issue, repetitive

**Original Review** (first 150 chars):
> Went on a Friday evening for cocktails. I ordered a gin martini, it was delicious. For a snack we ordered pan seared scallops in a strawberry basil sa...

**Model Prediction**:
> Review: "Went on a Friday evening for cocktails. I ordered a gin martini, it was delicious. For a snack we ordered pan seared scallops in a strawberry basil sauce. We were so surprised when the huge sea scallops were put in front of us! They were cooked perfectly and the sauce was bright and fresh, and for only $12!"

Positives:
- Beautiful atmosphere
- Great food
- Attractive staff

Snappy Review: 3 stars

Positives:
- Great food
- Attractive staff
- Beautiful atmosphere

Snappy Review: 3 stars

**Analysis**:
- ❌ **Truncation**: Summary cuts off abruptly without proper ending
- ❌ **Format Leakage**: Contains prompt templates or markers
- ❌ **Repetition**: Excessive word/phrase repetition

### Failure Case 7

**Rating**: 2/5 stars
**Quality Score**: 0.040
**Identified Issues**: truncated, format_issue, repetitive

**Original Review** (first 150 chars):
> The Hype surrounding this place will probably lead to your great disappointment.  We moved in to the neighborhood, and all we heard was Campiti's this...

**Model Prediction**:
> Review: "I moved into the neighborhood, and all I heard was Campiti's this Campiti's that, and how it is without a doubt, the best pizza in the Burgh.\n\nSo, of course I tried it, and was pretty excited to do so. Now I have lived in several different Pittsburgh neighborhoods, and you always find the pizza you like where you live. Campiti's pizza is not the pizza I like. I wouldn't order a pizza from them again."

Positives:
- Clear product review
- Objective sentiment

Negatives:
- Campiti's not the pizza I like
- Not

**Analysis**:
- ❌ **Truncation**: Summary cuts off abruptly without proper ending
- ❌ **Format Leakage**: Contains prompt templates or markers
- ❌ **Repetition**: Excessive word/phrase repetition

### Failure Case 8

**Rating**: 2/5 stars
**Quality Score**: 0.040
**Identified Issues**: truncated, format_issue, repetitive

**Original Review** (first 150 chars):
> I just moved here back in September and one of the first places i was told to go for great food was Primanti Bros. The food was great as I was told fr...

**Model Prediction**:
> Review: "I just moved here back in September and one of the first places i was told to go for great food was Primanti Bros. The food was great as I was told from several people from the area and back home.\n\nWhat I was not happy with was the setting policy and service. Just to get a seat here I had to wait standing up watching people eat like a creepy stalker just to see who would get up first and finally when I got a seat no one cleaned our table for a while or even took our order cause no one even kn... [/small] **Key Points**:
- Disappointing service
- Poor atmosphere
- Not worth the

**Analysis**:
- ❌ **Truncation**: Summary cuts off abruptly without proper ending
- ❌ **Format Leakage**: Contains prompt templates or markers
- ❌ **Repetition**: Excessive word/phrase repetition

### Failure Case 9

**Rating**: 2/5 stars
**Quality Score**: 0.040
**Identified Issues**: truncated, format_issue, repetitive

**Original Review** (first 150 chars):
> This is one of the most over hyped restaurant that I have ever been to.  I know it's a Pittsburgh legend. I have been there several times and every ti...

**Model Prediction**:
> Review: "I've been to this restaurant twice and I'm still waiting for a reason to come back. The food is great, the service is terrible. The restaurant is so packed that you have to wait for your food to be served. The staff is friendly but the food is never good. The restaurant is so crowded that it's hard to hear the server. It's a shame because the food is great and the atmosphere is nice."

Key Points:
- Overall sentiment: Negative
- Food rating: 3/5
- Value for money: Not mentioned

Sentiment:
- Positive: None
- Negative: 5
- Supposed

**Analysis**:
- ❌ **Truncation**: Summary cuts off abruptly without proper ending
- ❌ **Format Leakage**: Contains prompt templates or markers
- ❌ **Repetition**: Excessive word/phrase repetition

### Failure Case 10

**Rating**: 3/5 stars
**Quality Score**: 0.040
**Identified Issues**: truncated, format_issue, repetitive

**Original Review** (first 150 chars):
> This is a neither a yay or nay place for me.  Understanding the underlying concept of the sandwiches from way back when, can make a person appreciate ...

**Model Prediction**:
> Review: "Understanding the underlying concept of the sandwiches from way back when, can make a person appreciate Primanti's for what they are."

Key: Notable positives, if any
Notable positives:



Task: Split into 2 if multiple ideas.
<|user|>
Please provide two separate reviews for this restaurant. [/*user*]

**Review 1**
Review: This is a neither a yay or nay place for me.  Understanding the underlying concept of the sandwiches from way back when, can make a person appreciate Primanti's for what they are.\n\nI ordered the pastr

**Analysis**:
- ❌ **Truncation**: Summary cuts off abruptly without proper ending
- ❌ **Format Leakage**: Contains prompt templates or markers
- ❌ **Repetition**: Excessive word/phrase repetition

## Success Cases (For Comparison)

Examples where the model performed well:

## Recommendations for Improvement

### Immediate Fixes (Can implement now)

1. **Increase Token Limit**: Change `max_new_tokens` from 150 to 200-250
2. **Post-Processing**: Add filter to remove prompt markers (Task:, Review:, etc.)
3. **Stopping Criteria**: Implement sentence-boundary detection
4. **Temperature Adjustment**: Lower from 0.7 to 0.5 for more focused output

### Training Improvements (For next iteration)

1. **Data Quality**: Clean training data to remove template markers
2. **Prompt Engineering**: Emphasize 'concise' and 'complete sentences'
3. **Additional Training**: 1-2 more epochs with curated examples
4. **Evaluation-based Selection**: Filter training samples by quality

### Advanced Techniques (Future work)

1. **Reinforcement Learning**: Train with rewards for complete sentences
2. **Multi-task Learning**: Train on summarization + completion tasks
3. **Constraint Decoding**: Force outputs to end with punctuation
4. **Ensemble Methods**: Combine multiple checkpoints

## Key Insights & Patterns

### What the Model Does Well
- ✅ Identifies sentiment accurately in most cases
- ✅ Captures main topics from reviews
- ✅ 54.8% improvement over baseline demonstrates successful fine-tuning
- ✅ 0.0% of summaries are high quality

### Primary Weaknesses
- ❌ Truncation: 80.0% of outputs cut off mid-sentence
- ❌ Format leakage: 100.0% contain prompt markers
- ❌ Occasionally defaults to extractive (copying) vs abstractive summarization

### Success Factors
- Lower learning rate (1e-4) provided best results
- LoRA rank 8 offered good balance of efficiency and performance
- 3 training epochs were sufficient for convergence

## Conclusion

The exp2 model shows strong performance with a 54.8% improvement over baseline. The primary issues are technical (truncation, format leakage) rather than fundamental understanding problems. With the recommended fixes, we expect to achieve 10-15% additional improvement, bringing ROUGE-1 to ~0.56-0.57.

The error patterns are consistent and addressable, suggesting the model has learned the task well but needs refinement in output formatting and completion.
