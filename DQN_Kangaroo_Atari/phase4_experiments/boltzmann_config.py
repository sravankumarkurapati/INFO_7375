#!/usr/bin/env python3
"""
Configuration for Boltzmann exploration experiment
"""

class Exp9_Boltzmann:
    """Experiment 9: Boltzmann exploration instead of epsilon-greedy"""
    ENV_NAME = 'ALE/Kangaroo-v5'
    TOTAL_EPISODES = 1000
    TEST_EPISODES = 100
    MAX_STEPS = 500
    
    # DQN parameters
    LEARNING_RATE = 0.00025
    GAMMA = 0.8
    
    # Boltzmann (Temperature) parameters instead of epsilon
    TEMPERATURE_START = 1.0
    TEMPERATURE_END = 0.1
    TEMPERATURE_DECAY = 0.995
    
    # Replay buffer
    BUFFER_CAPACITY = 100000
    BATCH_SIZE = 32
    MIN_BUFFER_SIZE = 1000
    
    # Network update
    TARGET_UPDATE_FREQ = 1000
    
    # Frame preprocessing
    FRAME_HEIGHT = 84
    FRAME_WIDTH = 84
    NUM_STACKED_FRAMES = 4
    
    # Training
    TRAIN_FREQ = 4
    SAVE_FREQ = 500
    LOG_FREQ = 10
    
    # Experiment info
    EXPERIMENT_NAME = "exp9_boltzmann"
    DESCRIPTION = "Boltzmann (softmax) exploration with temperature decay instead of epsilon-greedy"
    
    # Paths
    MODEL_DIR = '../models'
    RESULTS_DIR = '../results'
    VIDEO_DIR = '../videos'