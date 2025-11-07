# Deep Q-Learning for Atari Kangaroo

**Course:** INFO 7375 - Fine-Tuning Large Language Models  
**Author:** Sravan Kumar Kurapati  
**Game:** Kangaroo (ALE/Kangaroo-v5)  
**Date:** November 2025

## 🎮 Project Overview

This project implements a Deep Q-Network (DQN) agent to play the Atari Kangaroo game using reinforcement learning. The implementation features experience replay, frame preprocessing, epsilon-greedy exploration, and comprehensive experimentation with various hyperparameters.

## 🏆 Key Results

- **Best Performance:** 188.8 mean reward (Experiment 5: epsilon_decay=0.99)
- **8 Systematic Experiments** testing learning rates, gamma values, and exploration strategies
- **Boltzmann Exploration** experiment showing limitations for this environment
- **Professional visualizations** with training curves and performance analysis

## 📁 Project Structure

```
kangaroo-dqn/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
├── dqn_agent.py                      # Core DQN implementation
├── experiment_configs.py             # Configuration for all experiments
├── run_single_experiment.py          # Experiment orchestration
├── run_boltzmann_experiment.py       # Boltzmann exploration variant
├── docs/                             # Complete documentation
│   ├── 01_baseline_performance.md
│   ├── 02_environment_analysis.md
│   ├── 03_reward_structure.md
│   ├── 04_bellman_parameters.md
│   ├── 05_policy_exploration.md
│   ├── 06_exploration_parameters.md
│   ├── 07_performance_metrics.md
│   ├── 08-14_theoretical_questions.md
│   └── 15-18_code_documentation.md
└── experiment_results/               # Results from all experiments
    ├── exp1_baseline/
    ├── exp2_lr_0001/
    ├── exp3_lr_001/
    └── ... (9 experiments total)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- macOS (tested on M1 MacBook Air) or Linux
- 8GB+ RAM recommended

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd kangaroo-dqn

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Atari ROMs
pip install gymnasium[atari,accept-rom-license]
```

### Running Experiments

```bash
# Run baseline experiment
python run_single_experiment.py --experiment 1

# Run all experiments
for i in {1..8}; do
    python run_single_experiment.py --experiment $i
done

# Run Boltzmann exploration experiment
python run_boltzmann_experiment.py
```

## 🧪 Experimental Results

### Experiment Configuration Summary

| Exp | Learning Rate | Gamma | Epsilon Decay | Mean Reward | Outcome |
|-----|---------------|-------|---------------|-------------|---------|
| 1   | 0.00025       | 0.99  | 0.995         | 178.6       | Baseline |
| 2   | 0.0001        | 0.99  | 0.995         | 176.2       | Slower learning |
| 3   | 0.001         | 0.99  | 0.995         | 165.4       | Unstable |
| 4   | 0.00025       | 0.8   | 0.995         | 162.8       | Shortsighted |
| 5   | 0.00025       | 0.99  | 0.99          | **188.8**   | **Best** |
| 6   | 0.00025       | 0.99  | 0.998         | 170.3       | Too conservative |
| 7   | 0.00025       | 0.9   | 0.995         | 172.1       | Moderate |
| 8   | 0.00025       | 0.95  | 0.995         | 174.5       | Balanced |
| 9   | 0.00025       | 0.99  | Boltzmann     | 44.0        | Failed |

### Key Findings

1. **Slower epsilon decay (0.99) performs best** - Allows more exploration during training
2. **Higher gamma (0.99) is crucial** - Long-term planning essential for Kangaroo gameplay
3. **Boltzmann exploration fails** - Unsuitable for this environment's action space
4. **Learning rate sensitivity** - 0.00025 provides optimal stability vs. speed

## 🎯 Assignment Requirements

All 18 assignment requirements are fully documented:

### Experimental Results (35 points)
- ✅ Section 1: Baseline Performance
- ✅ Section 2: Environment Analysis  
- ✅ Section 3: Reward Structure
- ✅ Section 4: Bellman Parameters (α, γ)
- ✅ Section 5: Policy Exploration
- ✅ Section 6: Exploration Parameters (ε)
- ✅ Section 7: Performance Metrics

### Theoretical Questions (35 points)
- ✅ Section 8: Q-Learning Classification
- ✅ Section 9: Q-Learning vs. LLM Agents
- ✅ Section 10: Bellman Equation Concepts
- ✅ Section 11: RL for LLM Agents
- ✅ Section 12: Planning in RL vs. LLM
- ✅ Section 13: Q-Learning Algorithm
- ✅ Section 14: LLM Agent Integration

### Code Documentation (30 points)
- ✅ Section 15: Code Attribution
- ✅ Section 16: Code Clarity
- ✅ Section 17: Licensing
- ✅ Section 18: Professionalism

See `/docs` directory for complete documentation.

## 🧠 Technical Implementation

### DQN Architecture

```
Input: 84x84x4 grayscale frames (stacked)
   ↓
Conv2D(32, 8x8, stride=4) + ReLU
   ↓
Conv2D(64, 4x4, stride=2) + ReLU
   ↓
Conv2D(64, 3x3, stride=1) + ReLU
   ↓
Flatten → Dense(512) + ReLU
   ↓
Output: Dense(18) [Q-values for each action]
```

### Key Features

- **Experience Replay**: 10,000 transition buffer
- **Frame Preprocessing**: 84x84 grayscale, frame stacking (4 frames)
- **Target Network**: Updated every 1000 steps
- **Epsilon-Greedy Exploration**: Start=1.0, min=0.01
- **Batch Learning**: Size 32
- **Huber Loss**: Robust to outliers

### Hyperparameters

```python
learning_rate = 0.00025      # Adam optimizer
gamma = 0.99                 # Discount factor
epsilon_start = 1.0          # Initial exploration
epsilon_min = 0.01           # Minimum exploration
epsilon_decay = 0.995        # Decay per episode
batch_size = 32              # Experience replay batch
memory_size = 10000          # Replay buffer capacity
target_update_freq = 1000    # Target network updates
max_steps = 500              # Steps per episode
num_episodes = 1000          # Training episodes
```

## 📊 Performance Visualization

Training curves, reward distributions, and comparative analyses are available in `experiment_results/` for each experiment.

## 🔬 Code Attribution

- **Original Code (60%)**: Experiment infrastructure, configurations, orchestration, analysis
- **Adapted Code (40%)**: DQN core architecture based on Mnih et al. (2015) and OpenAI Baselines (MIT License)

Full attribution details in `/docs/15_code_attribution.md`

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

### Third-Party Licenses
- OpenAI Baselines: MIT License
- Gymnasium: MIT License
- ALE (Arcade Learning Environment): GPL-2.0 (runtime only)

## 🎓 Academic Context

This project fulfills the requirements for the INFO 7375 assignment on Deep Q-Learning. All code follows academic integrity guidelines with proper attribution and licensing.

### Assignment Adaptations

The assignment suggested parameters (learning_rate=0.7, max_steps=99) are appropriate for tabular Q-learning but not for Deep Q-Networks. This implementation follows established DQN literature (Mnih et al., 2015) using:
- learning_rate=0.00025 (neural network standard)
- max_steps=500 (sufficient for Kangaroo gameplay)
- 1000 episodes (computational feasibility)

## 📚 References

1. Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." *Nature*, 518(7540), 529-533.
2. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction*. MIT Press.
3. OpenAI Baselines: https://github.com/openai/baselines
4. Gymnasium Documentation: https://gymnasium.farama.org/
5. ALE Kangaroo Environment: https://ale.farama.org/environments/kangaroo/

## 🤝 Contributing

This is an academic project. For questions or suggestions, please contact the author.

## 📧 Contact

**Sravan Kumar Kurapati**  
Course: INFO 7375 - Fine-Tuning Large Language Models  
Northeastern University

---

**Note**: This implementation demonstrates Deep Q-Learning for educational purposes. The agent achieves reasonable performance (188.8 mean reward) though professional-grade implementations may achieve higher scores through additional optimizations.