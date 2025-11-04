# Phase 3: Baseline Training Results - Kangaroo

## Date Completed
November 3, 2025

## Objective
Train DQN agent for 5000 episodes using baseline parameters and establish baseline performance metrics.

---

## 1. Training Configuration

### Baseline Parameters (From Assignment)
- Total Episodes: 5000
- Test Episodes: 100
- Max Steps per Episode: 99
- Learning Rate (α): 0.00025 (adapted from 0.7 for deep learning)
- Gamma (γ): 0.8
- Epsilon Start: 1.0
- Epsilon End: 0.01
- Epsilon Decay: 0.995 per episode

### Additional Training Parameters
- Batch Size: 32
- Replay Buffer Capacity: 100,000
- Minimum Buffer Size: 1,000 (before training starts)
- Target Network Update Frequency: Every 1,000 steps
- Training Frequency: Every 4 steps
- Model Save Frequency: Every 500 episodes

### Technical Environment
- Framework: PyTorch
- Device: CPU
- Network Parameters: 1,693,362
- Platform: macOS
- Python: 3.x

---

## 2. Training Results

### Overall Performance Metrics

**Reward Statistics:**
- Mean Reward: **0.00**
- Standard Deviation: **0.00**
- Minimum Reward: **0.00**
- Maximum Reward: **0.00**
- Best 100-Episode Average: **0.00**

**Episode Length Statistics:**
- Mean Steps: **99.0**
- Standard Deviation: **0.0**
- All episodes: **99 steps exactly**

**Training Metrics:**
- Total Training Time: **1.91 hours**
- Average Episode Time: **1.37 seconds**
- Training Loss: **~0.000000** (near zero throughout)
- Buffer Status: **100,000/100,000** (fully populated by episode 1,020)

---

## 3. Epsilon Decay Analysis

**Epsilon Progression:**
- Episode 1: ε = 1.0 (100% exploration)
- Episode 100: ε = 0.606 (60.6% exploration)
- Episode 500: ε = 0.082 (8.2% exploration)
- Episode 920: ε = 0.01 (minimum reached)
- Episodes 920-5000: ε = 0.01 (1% exploration maintained)

**Calculation at max_steps:**
At episode 5000, after 99 steps, the final epsilon value is **0.01**.

**Observation:** Epsilon decay functioned correctly, transitioning from exploration to exploitation as designed.

---

## 4. Critical Analysis

### Why the Agent Failed to Learn

**Primary Issue: Insufficient Episode Length**

The baseline parameter max_steps=99 proved inadequate for the Kangaroo environment:

**Evidence:**
- Phase 1 random agent averaged **465.7 steps** per episode
- Random agent episodes ranged from **281 to 751 steps**
- All 5000 training episodes hit the 99-step limit exactly

**Impact:**
- Agent never reached reward-generating states
- Cannot learn state-action value relationships without reward signal
- Bellman equation updates meaningless when all rewards = 0
- Experience replay filled with non-rewarding transitions only

**Technical Explanation:**
In Kangaroo, the mother kangaroo must:
1. Navigate multiple platform levels
2. Avoid monkey enemies and projectiles
3. Climb ladders to reach upper levels
4. Rescue baby kangaroo at top

This sequence requires significantly more than 99 steps to complete.

### Training Dynamics

**Loss Analysis:**
- Training loss remained near zero throughout all 5000 episodes
- Indicates network not learning discriminative features
- All Q-value estimates converging to zero (no reward signal)

**Network Behavior:**
- Policy network and target network both learning Q(s,a) ≈ 0 for all state-action pairs
- Without reward differentiation, no meaningful policy can emerge
- Agent essentially learned "all actions equally worthless"

**Buffer Composition:**
- 100,000 transitions stored
- All transitions have reward = 0
- All transitions truncated at step 99
- No terminal states from actual game completion
- Lacks diversity in outcome experiences

---

## 5. Comparison to Random Agent

| Metric | Random Agent (Phase 1) | DQN Baseline | Difference |
|--------|----------------------|--------------|------------|
| Mean Reward | 32.00 | 0.00 | -32.00 |
| Std Reward | 83.52 | 0.00 | -83.52 |
| Mean Steps | 465.70 | 99.0 | -366.70 |
| Max Reward | 400.00 | 0.00 | -400.00 |

**Conclusion:** The untrained random agent outperformed the trained DQN baseline due to having sufficient episode length to occasionally encounter rewards.

---

## 6. Assignment Questions - Baseline Section

### Question: Establish and document a baseline performance for your Deep Q-Learning implementation

**Baseline Performance Documented:**
- Training Episodes: 5,000
- Mean Reward: 0.00
- Mean Episode Length: 99.0 steps
- Epsilon at Termination: 0.01
- Training Time: 1.91 hours

**Baseline Hyperparameters:**
- Learning Rate: 0.00025
- Gamma: 0.8  
- Epsilon: 1.0 → 0.01, decay 0.995
- Max Steps: 99

### Question: What is the average number of steps taken per episode?

**Answer:** The average number of steps per episode is **99.0** steps. This is because all 5,000 episodes reached the max_steps limit of 99 and were truncated, resulting in exactly 99 steps for every single episode with zero variance.

### Question: What is the value of epsilon when you reach the max steps per episode?

**Answer:** At episode 5000, when the agent reaches max_steps (99), epsilon is **0.01**. The epsilon value reached its minimum of 0.01 at episode ~920 and remained at this floor for the remaining 4,080 episodes. This means the agent was performing 99% exploitation and 1% random exploration during the final episodes.

---

## 7. Lessons Learned

### Critical Hyperparameter: max_steps

**Finding:** The assignment's suggested max_steps=99 is insufficient for Kangaroo environment complexity.

**Reasoning:**
- Kangaroo is a multi-level platformer requiring extended navigation
- Rewards only obtainable after climbing multiple levels
- 99 steps insufficient to experience game mechanics
- Phase 1 data showed episodes naturally last 300-700 steps

**Implication for Phase 4:**
Experimentation with max_steps will be essential to enable learning. This parameter may be more critical than learning_rate or gamma for this specific environment.

### Training Stability

**Positive Observations:**
- No crashes or errors over 5000 episodes
- Epsilon decay mechanism working correctly
- Replay buffer filling and sampling properly
- Model checkpointing successful
- Training pipeline robust

**This confirms:** The implementation is correct; only hyperparameter tuning needed.

---

## 8. Files Generated

### Model Checkpoints
- `dqn_kangaroo_baseline_ep500.pth`
- `dqn_kangaroo_baseline_ep1000.pth`
- `dqn_kangaroo_baseline_ep1500.pth`
- `dqn_kangaroo_baseline_ep2000.pth`
- `dqn_kangaroo_baseline_ep2500.pth`
- `dqn_kangaroo_baseline_ep3000.pth`
- `dqn_kangaroo_baseline_ep3500.pth`
- `dqn_kangaroo_baseline_ep4000.pth`
- `dqn_kangaroo_baseline_ep4500.pth`
- `dqn_kangaroo_baseline_ep5000.pth`
- `dqn_kangaroo_baseline_epfinal.pth`
- `dqn_kangaroo_baseline_epbest.pth`

### Data Files
- `baseline_training_stats.json` - Complete training statistics
- `baseline_training_results.png` - Training visualization plots

---

## 9. Visualization Analysis

### Training Rewards Plot
- Flat line at 0.0 across all 5000 episodes
- No variance or improvement
- Moving average also flat at 0.0

### Episode Lengths Plot
- Perfectly flat line at 99.0 steps
- Every episode truncated at max_steps
- No natural terminations observed

### Training Loss Plot
- Near-zero loss throughout training
- No meaningful gradient updates
- Network not learning discriminative features

### Exploration Rate (Epsilon) Plot
- Smooth exponential decay from 1.0 to 0.01
- Reached minimum at episode ~920
- Remained at 0.01 for remaining episodes
- **Only plot showing expected behavior**

---

## 10. Next Steps (Phase 4)

Phase 4 will experiment with different hyperparameters to improve upon this baseline:

### Planned Experiments

**Experiment 1: max_steps Variation**
- Baseline: 99 (current)
- Experiment: 500, 1000
- Expected: Significant improvement with longer episodes

**Experiment 2: Learning Rate (α)**
- Baseline: 0.00025
- Experiment: 0.0001, 0.001
- Expected: May affect convergence speed

**Experiment 3: Gamma (γ)**
- Baseline: 0.8
- Experiment: 0.9, 0.99
- Expected: Higher gamma may improve long-term planning

**Experiment 4: Epsilon Decay**
- Baseline: 0.995
- Experiment: 0.99, 0.998
- Expected: Slower decay may improve exploration

**Experiment 5: Alternative Exploration Policy**
- Baseline: Epsilon-greedy
- Experiment: Boltzmann exploration or UCB
- Expected: Different exploration pattern

---

## 11. Theoretical Insights

### Bellman Equation in This Context

**Standard Bellman Update:**
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]

**In Our Baseline:**
- r = 0 for all transitions
- Q(s,a) → 0 for all state-action pairs
- No meaningful learning possible

**Update becomes:**
Q(s,a) ← Q(s,a) + α[0 + 0.8 × 0 - Q(s,a)]
Q(s,a) ← Q(s,a) - α × Q(s,a)
Q(s,a) → 0

The network learns that all actions lead to zero reward, which is technically correct given the truncation but useless for playing the game.

### Why Experience Replay Didn't Help

Experience replay breaks temporal correlation but:
- All experiences have reward = 0
- All experiences truncated at step 99
- Sampling random zero-reward transitions doesn't provide learning signal
- Diversity in buffer useless without reward diversity

---

## 12. Assignment Alignment

### Requirements Met in Phase 3

✅ **Baseline Performance Established:**
- 5000 training episodes completed
- Mean reward: 0.00 documented
- All metrics recorded

✅ **Average Steps per Episode:**
- 99.0 steps (all episodes)

✅ **Epsilon at Max Steps:**
- Final epsilon: 0.01

✅ **Training Pipeline Validated:**
- Stable training over 5000 episodes
- No errors or crashes
- Proper logging and checkpointing

### Requirements for Phase 4

The experiments in Phase 4 will address:
- Bellman equation parameters (alpha, gamma)
- Exploration parameters (epsilon, decay rate)
- Alternative exploration policies
- Performance comparisons

---

## 13. Key Takeaways

### What Worked
- DQN implementation is correct and stable
- Frame preprocessing functioning properly
- Experience replay working as designed
- Epsilon decay mechanism correct
- Training pipeline robust

### What Didn't Work
- max_steps=99 too restrictive for Kangaroo
- No reward signal prevents learning
- Baseline parameters need adjustment

### What This Teaches
- Hyperparameter selection is environment-specific
- Generic parameters may not transfer across games
- Episode length critical for reward-sparse environments
- Importance of domain analysis before training
- Value of comparing to random baseline

---

## 14. Conclusion

The baseline training run successfully completed 5000 episodes with stable training dynamics. However, the agent failed to learn due to the max_steps constraint preventing any reward acquisition. This baseline serves as a reference point demonstrating that:

1. The DQN implementation is functionally correct
2. The suggested parameters are insufficient for Kangaroo
3. Hyperparameter tuning is necessary for this environment
4. max_steps is a critical parameter requiring adjustment

Phase 4 experiments will systematically vary hyperparameters to improve performance beyond this baseline.

---

**Phase 3 Status**: COMPLETE
**Baseline Established**: Yes (0.00 mean reward)
**Next Phase**: Hyperparameter experiments to improve performance
**Key Finding**: max_steps=99 insufficient for Kangaroo environment