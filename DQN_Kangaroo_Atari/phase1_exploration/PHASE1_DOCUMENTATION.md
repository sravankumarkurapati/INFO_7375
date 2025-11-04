# Phase 1: Environment Setup & Exploration - Kangaroo

## Date Completed
November 3, 2025

## Objective
Understand the Kangaroo Atari environment's state space, action space, and establish why Deep Q-Learning (function approximation) is necessary instead of tabular Q-learning.

---

## 1. Environment Details

### State Space
- **Observation Shape**: (210, 160, 3) - RGB image
- **Total Pixel Values**: 100,800 continuous values per frame
- **Data Type**: uint8 (range 0-255)
- **Observed Value Range**: [0, 223]

### Action Space
- **Type**: Discrete
- **Number of Actions**: 18
- **Available Actions**:
  - **Basic Movements**: NOOP, FIRE, UP, RIGHT, LEFT, DOWN
  - **Diagonal Movements**: UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT
  - **Movement + Fire**: UPFIRE, RIGHTFIRE, LEFTFIRE, DOWNFIRE, UPRIGHTFIRE, UPLEFTFIRE, DOWNRIGHTFIRE, DOWNLEFTFIRE

---

## 2. Q-Table Size Analysis (Assignment Question)

### Question: What are the states, the actions, and the size of the Q-table?

**States:**
- Each state is a 210×160×3 RGB image = 100,800 pixel values
- Theoretical possible states: 256^100,800 ≈ 10^241,925 unique states

**Actions:**
- 18 discrete actions

**Q-Table Size:**
- Traditional Q-Table would need: States × Actions entries
- Size = 256^100,800 × 18 ≈ 10^241,925 state-action pairs
- **This is IMPOSSIBLY LARGE** (universe has only ~10^80 atoms)

**Why We Need Deep Q-Learning:**
- Cannot store Q-table with 10^241,925 entries
- DQN uses neural network as **function approximator**
- Network learns Q(s,a) mapping with ~1-2 million parameters
- Generalizes across similar visual states without explicit storage

---

## 3. Exploratory Random Agent Baseline

### Purpose
Understand environment behavior and establish a reference point for non-learning agent performance.

### Setup
- Episodes: 100
- Max Steps: 1000 per episode
- Policy: Uniform random action selection

### Results

**Reward Statistics:**
- Mean Reward: **32.00**
- Standard Deviation: **83.52**
- Min Reward: 0.00
- Max Reward: 400.00

**Episode Length Statistics:**
- Mean Steps: **465.70**
- Standard Deviation: **99.17**
- Min Steps: 281
- Max Steps: 751

### Key Insights
- Random agent performs poorly (32.00 avg reward)
- High variance (σ > μ) indicates inconsistent performance
- Episodes terminate around 450-500 steps on average
- Demonstrates clear need for intelligent learning algorithm

---

## 4. Game Mechanics

**Objective:** Mother kangaroo climbs platforms to rescue baby kangaroo

**Elements:**
- Lives: 3 per episode
- Enemies: Monkeys throwing projectiles
- Structure: Multi-level platforms with ladders
- Scoring: Points for climbing, avoiding damage, reaching objectives

---

## 5. Files Created

| File | Purpose |
|------|---------|
| `test_environment.py` | Environment verification |
| `explore_environment.py` | State/action space analysis |
| `baseline_random_agent.py` | Random agent performance |
| `kangaroo_sample_frames.png` | Visual game samples |
| `baseline_random_agent.png` | Performance plots |
| `PHASE1_DOCUMENTATION.md` | This documentation |

---

## 6. Technical Environment

- **Platform**: macOS
- **Python**: 3.x
- **Environment**: ALE/Kangaroo-v5
- **Key Libraries**: gymnasium, ale-py, torch, numpy, matplotlib

---

## 7. Visualizations Generated

### Sample Frames
Six sequential frames showing:
- Game layout with multiple platform levels
- Mother kangaroo character (bottom)
- Baby kangaroo (top level)
- Monkey enemies at various levels
- Score display and lives indicator

### Random Agent Performance Plots
- **Rewards over episodes**: Sporadic spikes with mostly low scores
- **Episode lengths**: High variability around 450-500 step average
- **Moving averages**: Show overall poor and inconsistent performance

---

**Phase 1 Status**: ✅ COMPLETE  
**Next Phase**: Implement DQN architecture (CNN, replay buffer, training loop)