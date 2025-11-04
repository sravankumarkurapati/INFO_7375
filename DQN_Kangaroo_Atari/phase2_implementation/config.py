#!/usr/bin/env python3
"""
Configuration file for DQN training
All hyperparameters in one place
"""

class Config:
    """DQN Training Configuration"""
    
    # Environment
    ENV_NAME = 'ALE/Kangaroo-v5'
    
    # Training episodes (as per assignment baseline)
    TOTAL_EPISODES = 5000
    TEST_EPISODES = 100
    MAX_STEPS = 99  # Can be adjusted if needed
    
    # DQN Hyperparameters
    LEARNING_RATE = 0.00025  # Adjusted for deep learning (0.7 is for tabular)
    GAMMA = 0.8  # Discount factor (as per assignment)
    
    # Epsilon-greedy parameters
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995  # Decay per episode
    
    # Replay buffer
    BUFFER_CAPACITY = 100000
    BATCH_SIZE = 32
    MIN_BUFFER_SIZE = 1000  # Start training after this many transitions
    
    # Network update
    TARGET_UPDATE_FREQ = 1000  # Update target network every N steps
    
    # Frame preprocessing
    FRAME_HEIGHT = 84
    FRAME_WIDTH = 84
    NUM_STACKED_FRAMES = 4
    
    # Training
    TRAIN_FREQ = 4  # Train every N steps
    SAVE_FREQ = 500  # Save model every N episodes
    
    # Logging
    LOG_FREQ = 10  # Print stats every N episodes
    
    # Paths
    MODEL_DIR = '../models'
    RESULTS_DIR = '../results'
    VIDEO_DIR = '../videos'
    
    @classmethod
    def print_config(cls):
        """Print all configuration parameters"""
        print("=" * 60)
        print("DQN TRAINING CONFIGURATION")
        print("=" * 60)
        print(f"\nEnvironment: {cls.ENV_NAME}")
        print(f"\nTraining Parameters:")
        print(f"  Total Episodes: {cls.TOTAL_EPISODES}")
        print(f"  Test Episodes: {cls.TEST_EPISODES}")
        print(f"  Max Steps per Episode: {cls.MAX_STEPS}")
        print(f"\nDQN Hyperparameters:")
        print(f"  Learning Rate: {cls.LEARNING_RATE}")
        print(f"  Gamma (discount): {cls.GAMMA}")
        print(f"  Epsilon: {cls.EPSILON_START} → {cls.EPSILON_END}")
        print(f"  Epsilon Decay: {cls.EPSILON_DECAY}")
        print(f"\nReplay Buffer:")
        print(f"  Capacity: {cls.BUFFER_CAPACITY}")
        print(f"  Batch Size: {cls.BATCH_SIZE}")
        print(f"  Min Buffer Size: {cls.MIN_BUFFER_SIZE}")
        print(f"\nFrame Processing:")
        print(f"  Frame Size: {cls.FRAME_HEIGHT}x{cls.FRAME_WIDTH}")
        print(f"  Stacked Frames: {cls.NUM_STACKED_FRAMES}")
        print("=" * 60)


if __name__ == "__main__":
    Config.print_config()