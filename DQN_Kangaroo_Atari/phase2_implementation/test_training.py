#!/usr/bin/env python3
"""
Quick test of training pipeline with only 10 episodes
"""
from train_dqn import DQNTrainer
from config import Config

# Override config for quick test
class TestConfig(Config):
    TOTAL_EPISODES = 10
    LOG_FREQ = 2
    SAVE_FREQ = 5

if __name__ == "__main__":
    print("=" * 60)
    print("QUICK TRAINING TEST - 10 EPISODES")
    print("=" * 60)
    
    trainer = DQNTrainer(TestConfig)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTest interrupted!")
    finally:
        trainer.close()
    
    print("\n✓ Training pipeline test complete!")