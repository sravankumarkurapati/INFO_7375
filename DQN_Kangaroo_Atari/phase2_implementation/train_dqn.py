#!/usr/bin/env python3
"""
Main training script for DQN on Kangaroo
Complete training loop with logging and visualization
"""
import gymnasium as gym
import ale_py
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime

from dqn_agent import DQNAgent
from frame_preprocessing import FramePreprocessor, FrameStack
from config import Config

# Register ALE environments
gym.register_envs(ale_py)

class DQNTrainer:
    """Handles the complete training process"""
    
    def __init__(self, config):
        """
        Initialize trainer
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Create directories
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        
        # Create environment
        self.env = gym.make(config.ENV_NAME, render_mode='rgb_array')
        
        # Get action space size
        self.num_actions = self.env.action_space.n
        
        # Initialize preprocessing
        self.preprocessor = FramePreprocessor(
            height=config.FRAME_HEIGHT,
            width=config.FRAME_WIDTH
        )
        self.frame_stack = FrameStack(num_frames=config.NUM_STACKED_FRAMES)
        
        # Initialize agent
        self.agent = DQNAgent(
            state_shape=(config.NUM_STACKED_FRAMES, config.FRAME_HEIGHT, config.FRAME_WIDTH),
            num_actions=self.num_actions,
            learning_rate=config.LEARNING_RATE,
            gamma=config.GAMMA,
            epsilon_start=config.EPSILON_START,
            epsilon_end=config.EPSILON_END,
            epsilon_decay=config.EPSILON_DECAY,
            buffer_capacity=config.BUFFER_CAPACITY,
            batch_size=config.BATCH_SIZE,
            target_update_freq=config.TARGET_UPDATE_FREQ
        )
        
        # Training stats
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_losses = []
        self.epsilon_history = []
        
        print("\n" + "=" * 60)
        print("DQN TRAINER INITIALIZED")
        print("=" * 60)
        print(f"Environment: {config.ENV_NAME}")
        print(f"Action space: {self.num_actions} actions")
        print(f"State shape: ({config.NUM_STACKED_FRAMES}, {config.FRAME_HEIGHT}, {config.FRAME_WIDTH})")
        print("=" * 60 + "\n")
    
    def train(self):
        """Main training loop"""
        
        print("=" * 60)
        print(f"STARTING TRAINING: {self.config.TOTAL_EPISODES} EPISODES")
        print("=" * 60 + "\n")
        
        start_time = time.time()
        
        for episode in range(1, self.config.TOTAL_EPISODES + 1):
            episode_reward, episode_length, episode_loss = self._train_episode()
            
            # Store stats
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episode_losses.append(episode_loss)
            self.epsilon_history.append(self.agent.epsilon)
            
            # Decay epsilon
            self.agent.decay_epsilon()
            
            # Logging
            if episode % self.config.LOG_FREQ == 0:
                avg_reward = np.mean(self.episode_rewards[-self.config.LOG_FREQ:])
                avg_length = np.mean(self.episode_lengths[-self.config.LOG_FREQ:])
                avg_loss = np.mean([l for l in self.episode_losses[-self.config.LOG_FREQ:] if l is not None])
                
                elapsed_time = time.time() - start_time
                
                print(f"Episode {episode}/{self.config.TOTAL_EPISODES}")
                print(f"  Avg Reward (last {self.config.LOG_FREQ}): {avg_reward:.2f}")
                print(f"  Avg Length: {avg_length:.1f}")
                print(f"  Avg Loss: {avg_loss:.6f}" if not np.isnan(avg_loss) else "  Avg Loss: N/A")
                print(f"  Epsilon: {self.agent.epsilon:.4f}")
                print(f"  Buffer Size: {len(self.agent.memory)}")
                print(f"  Time Elapsed: {elapsed_time/60:.1f} min")
                print("-" * 60)
            
            # Save model
            if episode % self.config.SAVE_FREQ == 0:
                self._save_checkpoint(episode)
        
        # Final save
        self._save_checkpoint('final')
        
        # Save training plots
        self._plot_training_results()
        
        total_time = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f"TRAINING COMPLETE!")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"{'=' * 60}\n")
    
    def _train_episode(self):
        """
        Train for one episode
        
        Returns:
            tuple: (episode_reward, episode_length, avg_loss)
        """
        # Reset environment
        observation, _ = self.env.reset()
        processed_frame = self.preprocessor.preprocess(observation)
        self.frame_stack.reset(processed_frame)
        state = self.frame_stack.get_state()
        
        episode_reward = 0
        episode_loss = []
        step = 0
        
        for step in range(self.config.MAX_STEPS):
            # Select action
            # State needs to be transposed from (H, W, C) to (C, H, W) for PyTorch
            state_input = np.transpose(state, (2, 0, 1))
            action = self.agent.select_action(state_input)
            
            # Take action
            next_observation, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Process next frame
            processed_next_frame = self.preprocessor.preprocess(next_observation)
            self.frame_stack.append(processed_next_frame)
            next_state = self.frame_stack.get_state()
            
            # Store transition
            next_state_input = np.transpose(next_state, (2, 0, 1))
            self.agent.store_transition(state_input, action, reward, next_state_input, done)
            
            # Train agent
            if step % self.config.TRAIN_FREQ == 0:
                loss = self.agent.train()
                if loss is not None:
                    episode_loss.append(loss)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        avg_loss = np.mean(episode_loss) if episode_loss else None
        
        return episode_reward, step + 1, avg_loss
    
    def _save_checkpoint(self, episode):
        """Save model checkpoint"""
        filepath = os.path.join(self.config.MODEL_DIR, f'dqn_kangaroo_ep{episode}.pth')
        self.agent.save(filepath)
    
    def _plot_training_results(self):
        """Plot and save training results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('DQN Training Results - Kangaroo', fontsize=16)
        
        # Plot 1: Episode Rewards
        ax1 = axes[0, 0]
        ax1.plot(self.episode_rewards, alpha=0.6, label='Episode Reward')
        if len(self.episode_rewards) >= 100:
            moving_avg = np.convolve(self.episode_rewards, np.ones(100)/100, mode='valid')
            ax1.plot(range(99, len(self.episode_rewards)), moving_avg, 
                    linewidth=2, label='Moving Average (100 episodes)')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Training Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Episode Lengths
        ax2 = axes[0, 1]
        ax2.plot(self.episode_lengths, alpha=0.6, label='Episode Length')
        if len(self.episode_lengths) >= 100:
            moving_avg = np.convolve(self.episode_lengths, np.ones(100)/100, mode='valid')
            ax2.plot(range(99, len(self.episode_lengths)), moving_avg, 
                    linewidth=2, label='Moving Average (100 episodes)')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.set_title('Episode Lengths')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Training Loss
        ax3 = axes[1, 0]
        losses_filtered = [l for l in self.episode_losses if l is not None]
        if losses_filtered:
            ax3.plot(losses_filtered, alpha=0.6, label='Loss')
            if len(losses_filtered) >= 100:
                moving_avg = np.convolve(losses_filtered, np.ones(100)/100, mode='valid')
                ax3.plot(range(99, len(losses_filtered)), moving_avg, 
                        linewidth=2, label='Moving Average (100 episodes)')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Loss')
        ax3.set_title('Training Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Epsilon Decay
        ax4 = axes[1, 1]
        ax4.plot(self.epsilon_history, linewidth=2, color='green')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Epsilon')
        ax4.set_title('Exploration Rate (Epsilon)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(self.config.RESULTS_DIR, f'training_results_{timestamp}.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"\n✓ Training results saved to {filepath}")
        plt.close()
    
    def close(self):
        """Clean up resources"""
        self.env.close()


def main():
    """Main entry point"""
    # Print configuration
    Config.print_config()
    
    # Create trainer
    trainer = DQNTrainer(Config)
    
    try:
        # Start training
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    finally:
        trainer.close()
        print("Environment closed.")


if __name__ == "__main__":
    main()