# ReviewSense AI - Model Evaluation Report

**Date**: 2025-10-23 18:58:41

**Evaluation Samples**: 200

## Overview

Comprehensive evaluation of TinyLlama models for review summarization.

## Model Configurations

| Model | Learning Rate | LoRA Rank | Description |
|-------|--------------|-----------|-------------|
| baseline | - | - | Base TinyLlama without fine-tuning |
| exp1 | 2e-4 | 8 | LR=2e-4, r=8 |
| exp2 | 1e-4 | 8 | LR=1e-4, r=8 |
| exp3 | 2e-4 | 16 | LR=2e-4, r=16 |

## Performance Metrics

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU |
|-------|---------|---------|---------|------|
| baseline | 0.3281 | 0.1632 | 0.2599 | 0.1025 |
| exp1 | 0.4767 | 0.4342 | 0.4635 | 0.2682 |
| exp2 | 0.5079 | 0.4724 | 0.4942 | 0.2940 |
| exp3 | 0.4715 | 0.4177 | 0.4517 | 0.2607 |

## Best Performing Model

- **ROUGE1**: exp2 (0.5079)
- **ROUGE2**: exp2 (0.4724)
- **ROUGEL**: exp2 (0.4942)
- **BLEU**: exp2 (0.2940)

## Sample Predictions

### Example 1

**Original Review**:
> I got 'new' tires from them and within two weeks got a flat. I took my car to a local mechanic to see if i could get the hole patched, but they said the reason I had a flat was because the previous patch had blown - WAIT, WHAT? I just got the tire and never needed to have it patched? This was suppos...

**Rating**: 1/5

**Model Outputs**:

- **baseline**: Review: I got new tires from Flynn's, but within two weeks, one of them had a flat. I took my car to a local mechanic, who said the previous patch had blown, but I had a flat instead. The tire company stated that a new tire was supposed to be installed, but someone had punctured it, then attempted to patch it. Despite the initial disappointment, the tire company apologized for the inconvenience and offered to replace the tire free of charge. The review highlights the resentful tire slashers and the lack of attention to detail in Flynn's handling of the situation.

- **exp1**: Review: I got 'new' tires from them and within two weeks got a flat. I took my car to a local mechanic to see if i could get the hole patched, but they said the reason I had a flat was because the previous patch had blown - WAIT, WHAT? I just got the tire and never needed to have it patched? This was supposed to be a new tire. \nI took the tire over to Flynn's and they told me that someone punctured my tire, then tried to patch it. So there are resentful tire slashers? I f...

**Overall Sentiment**: Negative
**

- **exp2**: Review: "I got 'new' tires from them and within two weeks got a flat. I took my car to a local mechanic to see if i could get the hole patched, but they said the reason I had a flat was because the previous patch had blown - WAIT, WHAT? I just got the tire and never needed to have it patched? This was supposed to be a new tire."

Positives:
- Clear instructions
- Improper handling
- Not what was promised

Negatives:
- Subpar product
- Disappointing experience
- Not worth the money"

Summary: Noteworthy points mentioned: dur

- **exp3**: Review: "I got 'new' tires from them and within two weeks got a flat. I took my car to a local mechanic to see if i could get the hole patched, but they said the reason I had a flat was because the previous patch had blown - WAIT, WHAT? I just got the tire and never needed to have it patched? This was supposed to be a new tire. \nI took the tire over to Flynn's and they told me that someone punctured my tire, then tried to patch it. So there are resentful tire slashers?"

Positives:
- Convenient location
- Ex

### Example 2

**Original Review**:
> Don't waste your time.  We had two different people come to our house to give us estimates for a deck (one of them the OWNER).  Both times, we never heard from them.  Not a call, not the estimate, nothing....

**Rating**: 1/5

**Model Outputs**:

- **baseline**: Task: Generate a concise summary of the review highlighting key points and sentiment.

Task: Generate a summary of the review.

Task: Generate a summary of the review highlighting key points and sentiment.

Task: Generate a summary of the review highlighting key points and sentiment.

Task: Generate a summary of the review highlighting key points and sentiment.

- **exp1**: Review: Don't waste your time.  We had two different people come to our house to give us estimates for a deck (one of them the OWNER).  Both times, we never heard from them. Not a call, not the estimate, nothing...

Sentiment: Negative

- **exp2**: Review: Don't waste your time.  We had two different people come to our house to give us estimates for a deck (one of them the OWNER).  Both times, we never heard from them.  Not a call, not the estimate, nothing....\n\nKey points: \n- Poor customer service \n- Conflicting information \n- Overpriced"

Positive:
- Poor customer service
- Conflicting information

Negative:
- Not worth the money"

Summary: Overall satisfaction is not recommended. Outstanding quality and value alternatives are available. Not recommended."

- **exp3**: Review: "I'm a fan of the restaurant, but have been disappointed with the service and quality of the food. My husband and I were seated promptly and were quickly served. \n\nThe food was okay, but not that special. It was a little pricey for what we were served. \n\nThe service was fine, but not that special either. We were quickly served, and our food was quickly delivered. \n\nI would not go back again, nor would I recommend it to others."

Provide: Summary (short and clear), sentiment (positive or negative), and short-stories mentioning positive and negative aspects. [/|user|

### Example 3

**Original Review**:
> All I can say is the worst! We were the only 2 people in the place for lunch, the place was freezing and loaded with kids toys! 2 bicycles, a scooter, and an electronic keyboard graced the dining room. A fish tank with filthy, slimy fingerprints smeared all over it is there for your enjoyment.\n\nOu...

**Rating**: 1/5

**Model Outputs**:

- **baseline**: Review: The worst lunch experience I've ever had. The place was packed with kids' toys and bicycles, and the food was overcooked and served in a freezing dining room. The fish tank had slimy fingerprints all over it, and the food was served at a cold temperature. The plates were small and the food spilled over onto some semi-clean dishes. The customer service was lacking, and the overall experience was unpleasant.

- **exp1**: Review: All I can say is the worst! We were the only 2 people in the place for lunch, the place was freezing and loaded with kids toys! 2 bicycles, a scooter, and an electronic keyboard graced the dining room. A fish tank with filthy, slimy fingerprints smeared all over it is there for your enjoyment. Our food came... no water to drink, no tea, medium temperature food. Of course its cold, just like the room, I never took my jacket off! The plates are too small, you food spills over onto some semi-clean surfaces. The worst review I've ever read! \

- **exp2**: Review: "All I can say is the worst! We were the only 2 people in the place for lunch, the place was freezing and loaded with kids toys! 2 bicycles, a scooter, and an electronic keyboard graced the dining room. A fish tank with filthy, slimy fingerprints smeared all over it is there for your enjoyment.\n\nOur food came... no water to drink, no tea, medium temperature food. Of course its cold, just like the room, I never took my jacket off! The plates are too small, you food spills over onto some semi-clean surfaces. This place is overpr

- **exp3**: Review (All I can say is the worst!): All I can say is the worst!

Task: Summarize this review in 2 sentences.
- Summary: Reviews summarize
- Status: Complete

*/|user|*/
Review summary: "All I can say is the worst!"

**/|assistant|**
Review Summary: "All I can say is the worst!"

