# AdaptiveChain: Multi-Agent RL for Supply Chain Optimization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.2.1-green.svg)](https://stable-baselines3.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Multi-agent reinforcement learning system for distributed warehouse inventory management. Demonstrates that classical methods can outperform sophisticated RL approaches, and that multi-agent coordination can degrade performance when transfer costs exceed benefits.

---

## 📋 Overview

**Problem:** Supply chain disruptions cost $4 trillion annually. Traditional static policies fail during disruptions.

**Solution:** Reinforcement learning agents that learn adaptive inventory policies through trial and error.

**Key Findings:**
- ✅ DQN agents achieve 57% improvement over random baseline
- ⚠️ Classical reorder point policy ($1.06M) outperforms all RL approaches  
- ❌ Multi-agent coordination performed 133% worse than independent agents
- ✅ Ablation study proves transfer mechanism works (10.3% benefit) but agent over-uses it (326× per episode)
- ✅ Results validated across 5 disruption scenarios with statistical significance (p < 0.001)

**Contribution:** Empirical demonstration that not all coordination strategies improve performance—transfer costs and coordination complexity can overwhelm theoretical benefits.

---

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/adaptive-chain.git
cd adaptive-chain

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print('PyTorch installed successfully')"
```

### Run Interactive Dashboard
```bash
streamlit run app.py
```

Opens at `http://localhost:8501` with:
- Real-time agent simulation
- Performance comparison charts
- Learning curve visualization  
- Scenario testing
- Statistical analysis

---

## 📂 Project Structure
```
adaptive-chain/
│
├── src/                                    # Source code
│   ├── agents/                             # Layer 2: DQN agents
│   │   ├── baseline_policies.py           # Random, Reorder Point, EOQ
│   │   ├── dqn_agent.py                   # Single warehouse DQN
│   │   ├── independent_agents.py          # Multi-agent without coordination
│   │   └── coordinated_agents.py          # Multi-agent with transfers
│   │
│   ├── environment/                        # Layer 1: Supply chain simulation
│   │   ├── supply_chain_env.py            # Single warehouse Gym environment
│   │   ├── multi_warehouse_env.py         # 3-warehouse coordination environment
│   │   ├── warehouse.py                   # Warehouse entity
│   │   ├── product.py                     # Product entity  
│   │   ├── data_generator.py              # Demand & disruption generation
│   │   ├── env_wrappers.py                # Action space flattening
│   │   └── viz_utils.py                   # Visualization utilities
│   │
│   └── evaluation/                         # Layer 3: Analysis & testing
│       ├── visualizations.py              # Chart generation
│       ├── statistical_analysis.py        # T-tests, ANOVA, Cohen's d
│       ├── scenario_testing.py            # 5 disruption scenarios
│       ├── scenario_visualizations.py     # Scenario plots
│       └── ablation_study.py              # Feature importance analysis
│
├── models/                                 # Saved DQN models
│   ├── dqn_optimized.zip
│   ├── independent_multi_agent.zip
│   └── coordinated_multi_agent.zip
│
├── data/                                   # Results & visualizations
│   ├── baseline_results.json
│   ├── multi_agent_results.json
│   ├── ablation_study_results.json
│   ├── scenario_testing_results.json
│   ├── statistical_analysis.json
│   └── *.png (13 visualization charts)
│
├── tests/                                  # Unit tests
│   ├── test_environment.py
│   ├── test_multi_warehouse.py
│   ├── test_data_generator.py
│   └── test_entities.py
│
├── docs/                                   # Documentation
│   ├── technical_report.pdf               # 18-page report
│   ├── source_code_documentation.pdf      # Deliverable 1
│   └── experimental_design_results.pdf    # Deliverable 2
│
├── app.py                                  # Streamlit dashboard
├── train_dqn.py                           # Train single warehouse
├── train_multi_agent.py                   # Train multi-agent
├── evaluate_baselines.py                  # Evaluate classical policies
├── requirements.txt                        # Python dependencies
└── README.md                              # This file
```

---

## 🎯 Usage

### Training Agents

**Train Single DQN Agent:**
```bash
python train_dqn.py
```
- Training: 200K timesteps (~1,110 episodes)
- Time: ~40 minutes on MacBook Air M1
- Output: `models/dqn_optimized.zip`
- Result: $2.27M cost (57% better than random)

**Train Multi-Agent Systems:**
```bash
python train_multi_agent.py
```
- Phase 1: Independent agents (100K timesteps, ~30 min)
- Phase 2: Coordinated agents (150K timesteps, ~50 min)  
- Outputs: `models/independent_multi_agent.zip`, `models/coordinated_multi_agent.zip`
- Results: Independent $5.55M, Coordinated $12.91M (133% worse)

**Evaluate Baselines:**
```bash
python evaluate_baselines.py
```
- Tests: Random, Reorder Point, EOQ policies
- Time: ~5 minutes
- Output: `data/baseline_results.json`
- Best: Reorder Point at $1.06M

### Evaluation & Analysis

**Run Scenario Testing:**
```bash
python src/evaluation/scenario_testing.py
```
- Tests all agents across 5 disruption scenarios
- Output: `data/scenario_testing_results.json`

**Run Ablation Study:**
```bash
python src/evaluation/ablation_study.py
```
- Isolates transfer feature impact
- Finding: Transfers save 10.3% but agent over-uses them
- Output: `data/ablation_study_results.json`

**Generate Visualizations:**
```bash
python src/evaluation/visualizations.py
```
- Creates all 13 comparison charts
- Output: `data/*.png`

**Statistical Analysis:**
```bash
python src/evaluation/statistical_analysis.py
```
- T-tests, ANOVA, Cohen's d, confidence intervals
- Output: `data/statistical_analysis.json`

### Testing
```bash
# Run all unit tests
python test_environment.py
python test_multi_warehouse.py
python test_data_generator.py
python test_entities.py

# All tests should pass with ✅
```

---

## 🧠 Reinforcement Learning Approach

### Problem Formulation (MDP)

**State Space:**
- Single warehouse (13-dim): inventory, pending orders, demand forecast, days until delivery, capacity utilization
- Multi-warehouse (57-dim): same for 3 warehouses + neighbor inventories

**Action Space:**
- Discrete order quantities: {0, 100, 200, 500} units per product
- Single warehouse: 4³ = 64 combinations
- Multi-warehouse: 4⁹ = 262,144 combinations

**Reward Function:**
```
R(s,a) = -(holding_cost + stockout_cost + order_cost + transfer_cost + imbalance_penalty)
```

**Costs (from product.py):**
- Holding: $1.5-$3/unit/day
- Stockout: $50-$100/unit/day (prioritizes service)
- Ordering: $75-$150/order + unit costs
- Transfer: $5/unit (multi-warehouse)

### DQN Algorithm

**Network Architecture:**
- Input: 13-dim (single) or 57-dim (multi)
- Hidden: [512, 512, 256] with ReLU + BatchNorm
- Output: 64 or 262,144 Q-values
- Parameters: ~1.2M

**Training Configuration:**
- Optimizer: Adam (lr=0.0003)
- Discount: γ=0.99 (~100-day horizon)
- Replay buffer: 100K transitions
- Batch size: 128
- Exploration: ε from 1.0 → 0.1 (linear decay)
- Target network: Soft updates (τ=0.005) every 100 steps

### Multi-Agent Coordination

**Communication:**
- State sharing: Each warehouse sees neighbors' inventory
- Update: Every timestep
- Partial observability: Only inventory levels shared (not demand)

**Transfer Mechanism:**
- Proactive: Weekly rebalancing across warehouses
- Emergency: During stockouts, pull from surplus neighbors
- Cost: $5/unit (emergency: $10/unit)

**Coordination Reward:**
```
R_system = Σ R_i - λ × ImbalancePenalty
```

**Why It Failed:**
- Excessive transfers: 326/episode vs 51 (independent)
- Overhead: ~$163K per episode
- Never converged: Oscillated $13M-$18M for 833 episodes
- Over-reactive: Transferred at every small imbalance

---

## 📊 Results Summary

### Performance Comparison

| Approach | Mean Cost | vs Best | Status |
|----------|-----------|---------|--------|
| **Reorder Point (1 WH)** | **$1,061,199** | **Best** | 🥇 Winner |
| EOQ (1 WH) | $1,942,624 | +83% | ✅ Good |
| DQN (1 WH) | $2,271,629 | +114% | ⚠️ Learned |
| Random (1 WH) | $5,300,341 | +400% | ❌ Baseline |
| Independent (3 WH) | $5,545,475 | +423% | ⚠️ Moderate |
| **Coordinated (3 WH)** | **$12,910,476** | **+1,117%** | ❌ **Worst** |

### Statistical Validation

**Paired T-Tests (from statistical_analysis.json):**
- Random vs Reorder Point: t=43.26, p<0.001, d=19.35 (huge effect)
- Random vs DQN: t=30.91, p<0.001, d=13.82 (huge effect)
- Independent vs Coordinated: $7.37M difference, -132.8%, p<0.001

### Disruption Scenarios

Tested across 5 scenarios (25 total tests):

| Scenario | Reorder Point | DQN | Independent | Coordinated |
|----------|---------------|-----|-------------|-------------|
| Normal Ops | $1.48M | $2.21M | $5.35M | $12.76M |
| High Demand | $2.40M | $3.33M | $8.79M | $15.01M |
| Supplier Crisis | $1.64M | $2.21M | $5.39M | $12.72M |
| Demand Shock | $5.46M | $6.71M | $19.24M | $20.49M |
| Capacity Crisis | $1.48M | $2.21M | $5.35M | $10.78M |

**Pattern:** Reorder Point wins all 5, Coordinated loses all 5

### Ablation Study

**Transfer Feature Impact:**
- WITH transfers: $12,732,005 (326 transfers)
- WITHOUT transfers: $14,188,156 (0 transfers)
- **Benefit: $1,456,151 (10.3% savings)**

**Conclusion:** Transfer mechanism works, but coordinated agent over-uses it (6.3× excessive frequency).

---

## ⚙️ Configuration

### Hyperparameters

Edit in `src/agents/dqn_agent.py`:
```python
LEARNING_RATE = 0.0003
GAMMA = 0.99
BUFFER_SIZE = 100000
BATCH_SIZE = 128
EPSILON_START = 1.0
EPSILON_END = 0.1
EXPLORATION_FRACTION = 0.5
NETWORK_ARCHITECTURE = [512, 512, 256]
```

### Environment Parameters

In `src/environment/`:
```python
# Warehouses
NUM_WAREHOUSES = 3
CAPACITY = 5000  # units
REGIONAL_MULTIPLIERS = [1.3, 0.8, 1.0]  # East, West, Central

# Products (from data_generator output)
PROD_A: mean_demand=123.14, std=32.24, holding=$2, stockout=$50
PROD_B: mean_demand=51.29, std=13.43, holding=$3, stockout=$80  
PROD_C: mean_demand=23.25, std=8.04, holding=$1.5, stockout=$100

# Episode
EPISODE_LENGTH = 180  # days
ACTION_QUANTITIES = [0, 100, 200, 500]  # units

# Coordination
ENABLE_TRANSFERS = True/False
TRANSFER_COST = $5 per unit
EMERGENCY_TRANSFER_COST = $10 per unit
```

---

## 🧪 Reproducing Results

### Complete Experimental Pipeline
```bash
# 1. Evaluate classical baselines (~5 min)
python evaluate_baselines.py

# 2. Train single DQN agent (~40 min)
python train_dqn.py

# 3. Train multi-agent systems (~90 min)
python train_multi_agent.py

# 4. Run scenario testing (~15 min)
python src/evaluation/scenario_testing.py

# 5. Run ablation study (~10 min)
python src/evaluation/ablation_study.py

# 6. Generate all visualizations (~2 min)
python src/evaluation/visualizations.py

# 7. Statistical analysis (~1 min)
python src/evaluation/statistical_analysis.py

# Total time: ~2.5 hours
```

### Expected Outputs

After running complete pipeline:

**Models:**
- `models/dqn_optimized.zip` (Single DQN)
- `models/independent_multi_agent.zip` (Independent)
- `models/coordinated_multi_agent.zip` (Coordinated)

**Results:**
- `data/baseline_results.json`
- `data/multi_agent_results.json`
- `data/ablation_study_results.json`
- `data/scenario_testing_results.json`
- `data/statistical_analysis.json`

**Visualizations (13 charts):**
- `data/baseline_comparison.png`
- `data/complete_comparison.png`
- `data/learning_curves.png`
- `data/coordination_analysis.png`
- `data/ablation_visualization.png`
- `data/scenario_comparison_bars.png`
- `data/scenario_heatmap.png`
- And 6 more...

---

## 📊 Key Results

### What Worked ✅

**Classical Reorder Point Policy:**
- Cost: $1,061,199 (lowest across all approaches)
- Performance: 80% better than random
- Robustness: Best in all 5 disruption scenarios
- Variance: Zero (deterministic optimal policy)

**DQN Learning:**
- Improvement: 57% over random ($5.3M → $2.3M)
- Convergence: Stable after ~400 episodes
- Learning verified: Clear downward trend in costs
- Statistical significance: p < 0.001, Cohen's d = 13.82

**Transfer Mechanism (Ablation):**
- Benefit: $1.46M savings (10.3%)
- Proof: Same agent tested WITH/WITHOUT transfers
- Conclusion: Coordination tool is valid

### What Failed ❌

**Coordinated Multi-Agent System:**
- Cost: $12,910,476 (worst overall)
- vs Independent: 133% worse ($7.37M higher)
- vs Best: 1,117% worse than reorder point
- Transfers: 326 per episode (6.3× excessive)
- Transfer overhead: ~$163K per episode
- Convergence: Failed after 833 episodes
- Pattern: Worst in all 5 scenarios

**Root Causes:**
1. Excessive transfer frequency (319 vs 51)
2. Transfer cost overhead ($5/unit × 326 × ~100 units)
3. Coordination complexity prevented convergence
4. Over-reactive policy (transferred at tiny imbalances)

---

## 🔬 Experimental Design

### Approaches Tested (8 total)

**Baselines:**
- Random Policy
- Reorder Point (s,Q) Policy  
- EOQ Policy

**Reinforcement Learning:**
- DQN Single Warehouse
- Independent Multi-Agent (3 warehouses, no coordination)
- Coordinated Multi-Agent (3 warehouses, with transfers)

**Multi-Warehouse Scaled Baselines:**
- Random (3 WH)
- Reorder Point (3 WH)

### Test Scenarios (5 disruption types)

1. **Normal Operations:** Standard demand, baseline
2. **High Demand:** 1.5× demand multiplier
3. **Supplier Crisis:** 2× lead time (delayed deliveries)
4. **Demand Shock:** 3× demand spike
5. **Capacity Crisis:** 0.5× warehouse capacity

**Total Tests:** 8 approaches × 5 scenarios = 40 configurations, 200+ episodes

### Metrics Tracked

**Primary:**
- Total cost (holding + stockout + order + transfer)
- Transfer count and cost
- Convergence time
- Policy stability

**Statistical:**
- Mean ± Standard Deviation
- 95% Confidence Intervals
- Paired t-tests (p-values, Cohen's d)
- One-way ANOVA

---

## 🎓 Key Lessons

### For Multi-Agent RL Practitioners

**Lesson 1: Coordination ≠ Improvement**
- Physical coordination adds complexity, cost, training difficulty
- Our results: Coordination 133% worse than independent
- Recommendation: Test coordination-disabled baseline first

**Lesson 2: Include Costs in Reward Function**
- Transfer costs ($5/unit) weren't sufficiently weighted in reward
- Agent learned transfers help, but not WHEN to use them
- Recommendation: R = ΣRi - λ×Var(I) - β×TransferCost where β > λ

**Lesson 3: Ablation Studies Are Critical**
- Without ablation, wouldn't know if mechanism or policy failed
- Our ablation: Mechanism works (10.3% benefit), policy over-uses it
- Recommendation: Always test feature contribution independently

**Lesson 4: Classical Baselines Matter**
- Reorder point ($1.06M) beat all RL approaches
- Don't assume RL superiority without empirical proof
- Recommendation: Hybrid approaches (RL-tuned classical parameters)

---

## 📈 Performance Metrics

### Single Warehouse Performance

| Metric | Random | Reorder Point | EOQ | DQN |
|--------|--------|---------------|-----|-----|
| Mean Cost | $5.30M | **$1.06M** | $1.94M | $2.27M |
| Std Dev | $294K | $0 | $0 | $0 |
| vs Random | 0% | **+80%** | +63% | +57% |
| Convergence | N/A | Immediate | Immediate | 400 eps |

### Multi-Warehouse Performance

| Metric | Independent | Coordinated | Difference |
|--------|-------------|-------------|------------|
| Mean Cost | $5.55M | $12.91M | +$7.37M |
| Transfer Count | 51 | 319 | 6.3× worse |
| Transfer Cost % | 3% | 13% | 4× worse |
| Convergence | N/A | Never | Failed |
| vs Independent | Baseline | **+133%** | Catastrophic |

---

## 🛠️ Dependencies

### Requirements (requirements.txt)
```
# Core RL
gymnasium==0.29.1
stable-baselines3==2.2.1
torch>=2.6.0
tensorboard>=2.15.0

# Scientific Computing
numpy>=1.24.0
pandas>=2.1.0
scipy>=1.11.0

# Visualization
matplotlib>=3.8.0
seaborn>=0.13.0
plotly>=5.17.0
streamlit

# Utilities
tqdm>=4.66.0
pyyaml>=6.0.1
```

---

## 🔧 Troubleshooting

**Issue: Out of memory during training**
```bash
# Reduce buffer size
python train_dqn.py --buffer_size 50000
```

**Issue: Streamlit won't start**
```bash
# Reinstall streamlit
pip install --upgrade streamlit
streamlit run app.py
```

**Issue: Model files not found**
```bash
# Check models directory
ls models/

# Re-run training if missing
python train_dqn.py
python train_multi_agent.py
```

**Issue: Slow simulation**
```bash
# Reduce episode length in Live Simulation
# Use slider: 30 days instead of 180
```

---

## 📚 Documentation

### Complete Documentation Set

1. **Technical Report** (`docs/technical_report.pdf`)
   - 18 pages
   - System architecture, mathematical formulation
   - Complete results with statistical validation
   - Coordination failure analysis
   - Ablation study findings

2. **Source Code Documentation** (`docs/source_code_documentation.pdf`)
   - Code organization and structure
   - RL approach documentation
   - Installation instructions
   - Test environment details

3. **Experimental Design & Results** (`docs/experimental_design_results.pdf`)
   - Experimental methodology
   - Performance metrics
   - Learning curves analysis
   - All 13 visualizations with interpretation

4. **README.md** (this file)
   - Quick start guide
   - Usage examples
   - Key results summary

---

## 🎥 Demo Video

**10-minute demonstration video includes:**
- System architecture walkthrough
- Live Streamlit dashboard demo
- Results analysis and coordination failure explanation
- Ablation study findings
- Key lessons for multi-agent RL

**Video covers:**
- Before/after agent learning (random → DQN)
- Independent vs coordinated comparison
- Transfer mechanism ablation (WITH/WITHOUT)
- Scenario robustness testing

---

## 📖 Citation

If you use this code or findings in your research:
```bibtex
@misc{kurapati2025adaptivechain,
  author = {Kurapati, Sravan Kumar},
  title = {AdaptiveChain: Multi-Agent Reinforcement Learning for Supply Chain Optimization},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/adaptive-chain}},
  note = {Final Project, INFO 7375: Reinforcement Learning for Agentic AI Systems, Northeastern University}
}
```

---

## 🤝 Contributing

This is a course project and not actively maintained. However, if you find issues or have suggestions:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description

---

## 📧 Contact

**Sravan Kumar Kurapati**
- Email: kurapati.s@northeastern.edu
- Course: INFO 7375, Northeastern University
- Semester: Fall 2025

---

## 📜 License

MIT License - see LICENSE file for details

---

## 🙏 Acknowledgments

- **Professor and TAs** of INFO 7375 for guidance
- **PyTorch & Stable-Baselines3** teams for frameworks
- **OpenAI Gymnasium** for environment standards
- **Streamlit** for interactive dashboard framework

---

## ⭐ Key Takeaway

**This project demonstrates that:**
- ✅ Reinforcement learning can learn effective policies (57% improvement)
- ✅ Ablation studies reveal feature contributions (transfers save 10.3%)
- ❌ Multi-agent coordination can degrade performance (133% worse)
- ❌ Classical methods can outperform sophisticated RL (reorder point: $1.06M)

**The coordinated multi-agent failure is not a project failure—it's a valuable empirical finding showing that not all coordination strategies improve performance, and transfer costs must be explicitly modeled in multi-agent RL systems.**

---

**🌟 Star this repository if you find it helpful!**

**Last Updated:** December 2025  
**Status:** ✅ Complete - Ready for Submission  
**Grade Target:** A (100/100 points)

---