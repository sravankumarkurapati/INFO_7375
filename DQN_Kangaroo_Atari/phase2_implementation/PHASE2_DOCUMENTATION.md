# Phase 2: DQN Implementation - Kangaroo

## Date Completed
November 3, 2025

## Objective
Implement the complete Deep Q-Network architecture including frame preprocessing, replay buffer, neural network, and training loop.

---

## 1. Components Implemented

### 1.1 Frame Preprocessing
**File**: `frame_preprocessing.py`
**Purpose**: Convert raw Atari frames to suitable DQN input

**Transformations**:
- RGB to Grayscale: Reduces 3 channels to 1
- Resize: 210×160 to 84×84
- Normalize: [0, 255] to [0, 1]
- Frame Stacking: 4 consecutive frames for motion

**Input**: (210, 160, 3) RGB frame
**Output**: (84, 84, 4) stacked grayscale frames
**Memory**: 110.25 KB per state

### 1.2 Replay Buffer
**File**: `replay_buffer.py`
**Purpose**: Store and sample experiences

**Specifications**:
- Capacity: 100,000 transitions
- Storage: (state, action, reward, next_state, done)
- Sampling: Uniform random, batch size 32
- Memory: ~215 MB for full buffer

**Why Experience Replay?**
Breaks temporal correlation, enables data reuse, stabilizes training

### 1.3 DQN Network
**File**: `dqn_network.py`
**Purpose**: Neural network Q-value approximator

**Architecture**:
- Conv1: 4→32 channels, 8×8 kernel, stride 4, ReLU
- Conv2: 32→64 channels, 4×4 kernel, stride 2, ReLU
- Conv3: 64→64 channels, 3×3 kernel, stride 1, ReLU
- FC1: 3136→512, ReLU
- FC2: 512→18 Q-values

**Parameters**: 1,693,362 trainable

### 1.4 DQN Agent
**File**: `dqn_agent.py`
**Purpose**: Complete DQN algorithm implementation

**Components**:
- Policy Network: Active training network
- Target Network: Stabilization network (updated every 1000 steps)
- Epsilon-Greedy: Exploration strategy
- Optimizer: Adam, learning rate 0.00025
- Loss: Smooth L1 Loss (Huber Loss)

---

## 2. Training Configuration

### Baseline Parameters
- Total Episodes: 5000
- Test Episodes: 100
- Max Steps: 99 per episode
- Learning Rate: 0.00025
- Gamma: 0.8
- Epsilon: 1.0 to 0.01, decay 0.995

### Additional Parameters
- Batch Size: 32
- Buffer Capacity: 100,000
- Target Update: Every 1000 steps
- Train Frequency: Every 4 steps
- Save Frequency: Every 500 episodes

---

## 3. DQN Algorithm

### Bellman Equation
Traditional Q-Learning: Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]

DQN replaces Q-table with neural network Q(s,a;θ)

### Training Loop
1. Select action using epsilon-greedy: a = argmax Q(s,a) or random
2. Execute action, observe reward r and next state s'
3. Store transition (s,a,r,s',done) in replay buffer
4. Sample random batch from buffer
5. Compute target: y = r + γ max Q(s',a';θ⁻)
6. Update network: minimize (y - Q(s,a;θ))²
7. Every N steps: update target network θ⁻ = θ

---

## 4. Training Pipeline Test

### Test Configuration
- Episodes: 10
- Purpose: Verify integration

### Results
- Episodes completed: 10/10
- Average reward: 0.00 (untrained)
- Episode length: 99 steps (max reached)
- Loss: 0.000060 to 0.000000 (decreasing)
- Epsilon: 1.0 to 0.9511 (decaying)
- Buffer: 0 to 990 transitions
- Models saved: Episodes 5, 10, final
- Plots generated: Training results

### Status
All components working correctly, ready for full training

---

## 5. Files Created

| File | Purpose | Lines |
|------|---------|-------|
| frame_preprocessing.py | Frame processing | 150 |
| replay_buffer.py | Experience replay | 100 |
| dqn_network.py | CNN architecture | 120 |
| dqn_agent.py | DQN algorithm | 200 |
| config.py | Configuration | 80 |
| train_dqn.py | Training loop | 300 |
| test_training.py | Quick test | 30 |
| PHASE2_DOCUMENTATION.md | Documentation | - |

**Total Code**: ~980 lines

---

## 6. Key Design Decisions

### Learning Rate 0.00025
Assignment suggested 0.7 for tabular Q-learning. Deep learning requires smaller rates. 0.00025 is standard for DQN.

### Gamma 0.8
As per assignment. Lower than typical 0.99, emphasizes immediate rewards.

### Epsilon Decay 0.995
Gradual exploration reduction. After 1000 episodes: ε ≈ 0.007.

### Frame Stacking 4
Captures motion and velocity. Single frame insufficient for dynamics.

### Target Network
Prevents moving target problem. Updated every 1000 steps for stability.

---

## 7. Technical Environment

- Framework: PyTorch (CPU)
- Environment: Gymnasium ALE/Kangaroo-v5
- Python: 3.x
- Platform: macOS
- Device: CPU
- Total Parameters: 1,693,362

---

## 8. Next Steps (Phase 3)

Phase 3 involves:
1. Full training run: 5000 episodes
2. Baseline performance documentation
3. Testing: 100 test episodes
4. Analysis: Compare to random baseline
5. Visualizations: Comprehensive plots
6. Metrics: Average reward, steps, loss

Estimated training time: 4-6 hours on CPU

---

## 9. Component Testing Results

### Frame Preprocessing Test
- Original: (210, 160, 3), uint8, [0-223]
- Processed: (84, 84, 1), float32, [0.000-0.722]
- Stacked: (84, 84, 4), float32, 110.25 KB
- Status: PASSED

### Replay Buffer Test
- Capacity: 1000 transitions
- Samples: 100 transitions added
- Batch: 32 sampled successfully
- Memory: 6.89 MB per batch, 215.33 MB full buffer
- Status: PASSED

### DQN Network Test
- Architecture: Conv layers + FC layers verified
- Parameters: 1,693,362 total
- Forward pass: (32, 4, 84, 84) → (32, 18)
- Action selection: Greedy and epsilon-greedy working
- Status: PASSED

### DQN Agent Test
- Agent created on CPU
- Action selection working
- Buffer filling correctly
- Training step executing
- Loss computed: 0.125455
- Epsilon decay working: 1.0 → 0.995
- Status: PASSED

### Training Pipeline Test
- 10 episodes completed
- Logs generated every 2 episodes
- Models saved at episodes 5, 10, final
- Training plots generated
- No errors or crashes
- Status: PASSED

---

## 10. Hyperparameter Justification

### Why Learning Rate = 0.00025?
The assignment suggests learning_rate = 0.7, which is appropriate for tabular Q-learning where we directly update Q-values. However, in Deep Q-Learning:
- We're training a neural network with gradient descent
- Large learning rates (like 0.7) would cause divergence
- 0.00025 is the standard from DeepMind's Nature paper
- This rate ensures stable convergence over millions of gradient steps

### Why Gamma = 0.8?
Per assignment specification. This is lower than typical Atari DQN (0.99):
- Emphasizes immediate rewards more
- Suitable for games with shorter episode horizons
- Faster learning of near-term strategy
- May sacrifice long-term planning

### Why These Epsilon Parameters?
- Start at 1.0: Full exploration initially to discover environment
- Decay to 0.01: Maintain 1% exploration to prevent local optima
- Decay rate 0.995: Gradual transition from exploration to exploitation
- After 500 episodes: ε ≈ 8% (mostly exploitation)
- After 1000 episodes: ε ≈ 0.7% (near minimum)

### Why Batch Size = 32?
- Standard mini-batch size for efficient GPU/CPU computation
- Balances gradient estimate quality with computational cost
- Larger batches: more stable but slower
- Smaller batches: faster but noisier gradients

### Why Buffer Capacity = 100,000?
- Stores diverse experiences from multiple episodes
- Enables sampling from wide distribution of states
- 100k transitions ≈ 1000 episodes worth of data
- Balances memory usage (~215 MB) with diversity

### Why Target Update Frequency = 1000?
- Too frequent: target moves too quickly, unstable training
- Too infrequent: slow learning, outdated targets
- 1000 steps: proven effective in DeepMind's research
- Provides stability while allowing progress

---

## 11. Theoretical Foundation

### Q-Learning Algorithm
Q-learning is a value-based reinforcement learning algorithm that learns the optimal action-value function Q*(s,a), representing the expected cumulative reward for taking action a in state s and following the optimal policy thereafter.

**Update Rule**:
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]

**Components**:
- α (alpha): Learning rate, controls update step size
- γ (gamma): Discount factor, balances immediate vs future rewards
- r: Immediate reward
- max_a' Q(s',a'): Maximum Q-value for next state (greedy action)

### Deep Q-Learning
DQN extends Q-learning to high-dimensional state spaces by using neural networks as function approximators:

**Key Insight**: Instead of storing Q-values in a table, approximate them with a neural network Q(s,a;θ) parameterized by weights θ.

**Loss Function**:
L(θ) = E[(r + γ max_a' Q(s',a';θ⁻) - Q(s,a;θ))²]

Where θ⁻ are the target network parameters.

**Innovations**:
1. Experience Replay: Store transitions, sample randomly to break correlations
2. Target Network: Separate network for computing targets, updated periodically
3. Frame Preprocessing: Reduce dimensionality, normalize inputs
4. Epsilon-Greedy: Balance exploration and exploitation

### Value-Based vs Policy-Based
Q-learning (including DQN) is VALUE-BASED:
- Learns value function Q(s,a)
- Derives policy implicitly: π(s) = argmax_a Q(s,a)
- Iteration updates value estimates

Policy-based methods:
- Learn policy π(s,a) directly
- No value function needed
- Examples: REINFORCE, Actor-Critic

DQN is definitively value-based because it learns Q-values and derives actions through maximization.

---

**Phase 2 Status**: COMPLETE
**All Tests**: PASSED
**Ready for Phase 3**: YES
**Estimated Phase 3 Time**: 4-6 hours