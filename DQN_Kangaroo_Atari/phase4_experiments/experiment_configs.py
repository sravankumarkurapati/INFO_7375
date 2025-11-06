#!/usr/bin/env python3
"""
Experiment configurations for hyperparameter tuning
Each experiment varies one or two parameters from baseline
"""

class BaselineConfig:
    """Original baseline configuration"""
    ENV_NAME = 'ALE/Kangaroo-v5'
    TOTAL_EPISODES = 5000
    TEST_EPISODES = 100
    MAX_STEPS = 99
    LEARNING_RATE = 0.00025
    GAMMA = 0.8
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    BUFFER_CAPACITY = 100000
    BATCH_SIZE = 32
    MIN_BUFFER_SIZE = 1000
    TARGET_UPDATE_FREQ = 1000
    FRAME_HEIGHT = 84
    FRAME_WIDTH = 84
    NUM_STACKED_FRAMES = 4
    TRAIN_FREQ = 4
    SAVE_FREQ = 500
    LOG_FREQ = 10
    MODEL_DIR = '../models'
    RESULTS_DIR = '../results'
    VIDEO_DIR = '../videos'


class Exp1_MaxSteps500(BaselineConfig):
    """Experiment 1: Increase max_steps to 500"""
    TOTAL_EPISODES = 1000  # Shorter run for experiments
    MAX_STEPS = 500
    EXPERIMENT_NAME = "exp1_maxsteps500"
    DESCRIPTION = "Testing max_steps=500 to allow agent more time to explore and find rewards"


class Exp2_MaxSteps1000(BaselineConfig):
    """Experiment 2: Increase max_steps to 1000"""
    TOTAL_EPISODES = 1000
    MAX_STEPS = 1000
    EXPERIMENT_NAME = "exp2_maxsteps1000"
    DESCRIPTION = "Testing max_steps=1000 for full episode completion"


class Exp3_LR0001(BaselineConfig):
    """Experiment 3: Lower learning rate"""
    TOTAL_EPISODES = 1000
    MAX_STEPS = 500  # Use working max_steps
    LEARNING_RATE = 0.0001
    EXPERIMENT_NAME = "exp3_lr0001"
    DESCRIPTION = "Testing lower learning rate (0.0001) for more stable learning"


class Exp4_LR001(BaselineConfig):
    """Experiment 4: Higher learning rate"""
    TOTAL_EPISODES = 1000
    MAX_STEPS = 500
    LEARNING_RATE = 0.001
    EXPERIMENT_NAME = "exp4_lr001"
    DESCRIPTION = "Testing higher learning rate (0.001) for faster learning"


class Exp5_Gamma09(BaselineConfig):
    """Experiment 5: Gamma = 0.9"""
    TOTAL_EPISODES = 1000
    MAX_STEPS = 500
    GAMMA = 0.9
    EXPERIMENT_NAME = "exp5_gamma09"
    DESCRIPTION = "Testing gamma=0.9 for better long-term reward consideration"


class Exp6_Gamma099(BaselineConfig):
    """Experiment 6: Gamma = 0.99"""
    TOTAL_EPISODES = 1000
    MAX_STEPS = 500
    GAMMA = 0.99
    EXPERIMENT_NAME = "exp6_gamma099"
    DESCRIPTION = "Testing gamma=0.99 (standard for Atari) for maximum long-term planning"


class Exp7_Decay099(BaselineConfig):
    """Experiment 7: Slower epsilon decay"""
    TOTAL_EPISODES = 1000
    MAX_STEPS = 500
    EPSILON_DECAY = 0.99
    EXPERIMENT_NAME = "exp7_decay099"
    DESCRIPTION = "Testing slower epsilon decay (0.99) for extended exploration"


class Exp8_Decay0998(BaselineConfig):
    """Experiment 8: Even slower epsilon decay"""
    TOTAL_EPISODES = 1000
    MAX_STEPS = 500
    EPSILON_DECAY = 0.998
    EXPERIMENT_NAME = "exp8_decay0998"
    DESCRIPTION = "Testing very slow epsilon decay (0.998) for maximum exploration time"

class Exp9_Boltzmann(BaselineConfig):
    """Experiment 9: Boltzmann exploration instead of epsilon-greedy"""
    TOTAL_EPISODES = 1000
    MAX_STEPS = 500
    LEARNING_RATE = 0.00025
    GAMMA = 0.8
    
    # Temperature parameters (instead of epsilon)
    TEMPERATURE_START = 1.0
    TEMPERATURE_END = 0.1
    TEMPERATURE_DECAY = 0.995
    
    USE_BOLTZMANN = True  # Flag to use Boltzmann agent
    
    EXPERIMENT_NAME = "exp9_boltzmann"
    DESCRIPTION = "Boltzmann (softmax) exploration with temperature decay instead of epsilon-greedy"

# Add to EXPERIMENTS list
EXPERIMENTS = [
    Exp1_MaxSteps500,
    Exp2_MaxSteps1000,
    Exp3_LR0001,
    Exp4_LR001,
    Exp5_Gamma09,
    Exp6_Gamma099,
    Exp7_Decay099,
    Exp8_Decay0998,
    Exp9_Boltzmann  # ADD THIS!
]

def print_all_experiments():
    """Print summary of all experiments"""
    print("=" * 70)
    print("PHASE 4: EXPERIMENT CONFIGURATIONS")
    print("=" * 70)
    
    for i, exp_class in enumerate(EXPERIMENTS, 1):
        print(f"\nExperiment {i}: {exp_class.EXPERIMENT_NAME}")
        print(f"  Description: {exp_class.DESCRIPTION}")
        print(f"  Episodes: {exp_class.TOTAL_EPISODES}")
        print(f"  Max Steps: {exp_class.MAX_STEPS}")
        print(f"  Learning Rate: {exp_class.LEARNING_RATE}")
        print(f"  Gamma: {exp_class.GAMMA}")
        print(f"  Epsilon Decay: {exp_class.EPSILON_DECAY}")
        print("-" * 70)
    
    total_episodes = sum(exp.TOTAL_EPISODES for exp in EXPERIMENTS)
    print(f"\nTotal Episodes Across All Experiments: {total_episodes}")
    print("=" * 70)


if __name__ == "__main__":
    print_all_experiments()