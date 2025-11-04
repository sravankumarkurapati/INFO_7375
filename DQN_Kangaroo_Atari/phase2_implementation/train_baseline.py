#!/usr/bin/env python3
"""
Baseline Training Run - 5000 Episodes
Enhanced monitoring and logging
"""
import gymnasium as gym
import ale_py
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import json
from datetime import datetime

from dqn_agent import DQNAgent
from frame_preprocessing import FramePreprocessor, FrameStack
from config import Config

gym.register_envs(ale_py)

class BaselineTrainer:
    """Baseline training with enhanced monitoring"""
    
    def __init__(self, config):
        self.config = config
        
        # Create directories
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        
        # Create environment
        self.env = gym.make(config.ENV_NAME, render_mode='rgb_array')
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
        
        # Checkpoint tracking
        self.best_avg_reward = -float('inf')
        
        # Timing
        self.start_time = None
        self.episode_times = []
        
        print("\n" + "=" * 70)
        print("BASELINE DQN TRAINER INITIALIZED")
        print("=" * 70)
        print(f"Environment: {config.ENV_NAME}")
        print(f"Total Episodes: {config.TOTAL_EPISODES}")
        print(f"Action Space: {self.num_actions} actions")
        print(f"State Shape: ({config.NUM_STACKED_FRAMES}, {config.FRAME_HEIGHT}, {config.FRAME_WIDTH})")
        print(f"Learning Rate: {config.LEARNING_RATE}")
        print(f"Gamma: {config.GAMMA}")
        print(f"Epsilon: {config.EPSILON_START} → {config.EPSILON_END}")
        print("=" * 70 + "\n")
    
    def train(self):
        """Main training loop"""
        
        print("=" * 70)
        print(f"STARTING BASELINE TRAINING: {self.config.TOTAL_EPISODES} EPISODES")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70 + "\n")
        
        self.start_time = time.time()
        
        for episode in range(1, self.config.TOTAL_EPISODES + 1):
            episode_start = time.time()
            
            episode_reward, episode_length, episode_loss = self._train_episode()
            
            episode_time = time.time() - episode_start
            self.episode_times.append(episode_time)
            
            # Store stats
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episode_losses.append(episode_loss)
            self.epsilon_history.append(self.agent.epsilon)
            
            # Decay epsilon
            self.agent.decay_epsilon()
            
            # Logging
            if episode % self.config.LOG_FREQ == 0:
                self._log_progress(episode)
            
            # Save model periodically
            if episode % self.config.SAVE_FREQ == 0:
                self._save_checkpoint(episode)
            
            # Save best model
            if episode >= 100:
                avg_reward_100 = np.mean(self.episode_rewards[-100:])
                if avg_reward_100 > self.best_avg_reward:
                    self.best_avg_reward = avg_reward_100
                    self._save_checkpoint('best')
        
        # Final saves
        self._save_checkpoint('final')
        self._save_training_stats()
        self._plot_training_results()
        
        total_time = time.time() - self.start_time
        print(f"\n{'=' * 70}")
        print(f"BASELINE TRAINING COMPLETE!")
        print(f"Total Time: {total_time/3600:.2f} hours")
        print(f"Average Episode Time: {np.mean(self.episode_times):.2f} seconds")
        print(f"Best Average Reward (100 episodes): {self.best_avg_reward:.2f}")
        print(f"{'=' * 70}\n")
    
    def _train_episode(self):
        """Train for one episode"""
        observation, _ = self.env.reset()
        processed_frame = self.preprocessor.preprocess(observation)
        self.frame_stack.reset(processed_frame)
        state = self.frame_stack.get_state()
        
        episode_reward = 0
        episode_loss = []
        step = 0
        
        for step in range(self.config.MAX_STEPS):
            # Select action
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
    
    def _log_progress(self, episode):
        """Log training progress"""
        window = self.config.LOG_FREQ
        avg_reward = np.mean(self.episode_rewards[-window:])
        avg_length = np.mean(self.episode_lengths[-window:])
        losses = [l for l in self.episode_losses[-window:] if l is not None]
        avg_loss = np.mean(losses) if losses else 0
        
        elapsed_time = time.time() - self.start_time
        eta_seconds = (elapsed_time / episode) * (self.config.TOTAL_EPISODES - episode)
        eta_hours = eta_seconds / 3600
        
        print(f"Episode {episode}/{self.config.TOTAL_EPISODES}")
        print(f"  Reward (avg last {window}): {avg_reward:.2f}")
        print(f"  Length (avg): {avg_length:.1f}")
        print(f"  Loss (avg): {avg_loss:.6f}")
        print(f"  Epsilon: {self.agent.epsilon:.4f}")
        print(f"  Buffer: {len(self.agent.memory)}/{self.config.BUFFER_CAPACITY}")
        print(f"  Time: {elapsed_time/60:.1f}min | ETA: {eta_hours:.1f}h")
        print("-" * 70)
    
    def _save_checkpoint(self, episode):
        """Save model checkpoint"""
        filepath = os.path.join(self.config.MODEL_DIR, f'dqn_kangaroo_baseline_ep{episode}.pth')
        self.agent.save(filepath)
    
    def _save_training_stats(self):
        """Save training statistics to JSON"""
        stats = {
            'total_episodes': self.config.TOTAL_EPISODES,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'epsilon_history': self.epsilon_history,
            'config': {
                'learning_rate': self.config.LEARNING_RATE,
                'gamma': self.config.GAMMA,
                'epsilon_start': self.config.EPSILON_START,
                'epsilon_end': self.config.EPSILON_END,
                'epsilon_decay': self.config.EPSILON_DECAY,
                'max_steps': self.config.MAX_STEPS
            },
            'final_stats': {
                'mean_reward': float(np.mean(self.episode_rewards)),
                'std_reward': float(np.std(self.episode_rewards)),
                'mean_length': float(np.mean(self.episode_lengths)),
                'best_avg_reward_100': float(self.best_avg_reward)
            }
        }
        
        filepath = os.path.join(self.config.RESULTS_DIR, 'baseline_training_stats.json')
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✓ Training stats saved to {filepath}")
    
    def _plot_training_results(self):
        """Plot and save training results"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('DQN Baseline Training - Kangaroo (5000 Episodes)', fontsize=16, fontweight='bold')
        
        # Plot 1: Episode Rewards
        ax1 = axes[0, 0]
        ax1.plot(self.episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
        if len(self.episode_rewards) >= 100:
            moving_avg = np.convolve(self.episode_rewards, np.ones(100)/100, mode='valid')
            ax1.plot(range(99, len(self.episode_rewards)), moving_avg, 
                    linewidth=2, color='red', label='Moving Avg (100 ep)')
        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Total Reward', fontsize=12)
        ax1.set_title('Training Rewards', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Episode Lengths
        ax2 = axes[0, 1]
        ax2.plot(self.episode_lengths, alpha=0.3, color='green', label='Episode Length')
        if len(self.episode_lengths) >= 100:
            moving_avg = np.convolve(self.episode_lengths, np.ones(100)/100, mode='valid')
            ax2.plot(range(99, len(self.episode_lengths)), moving_avg, 
                    linewidth=2, color='darkgreen', label='Moving Avg (100 ep)')
        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('Steps', fontsize=12)
        ax2.set_title('Episode Lengths', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Training Loss
        ax3 = axes[1, 0]
        losses_filtered = [l for l in self.episode_losses if l is not None]
        if losses_filtered:
            ax3.plot(losses_filtered, alpha=0.3, color='orange', label='Loss')
            if len(losses_filtered) >= 100:
                moving_avg = np.convolve(losses_filtered, np.ones(100)/100, mode='valid')
                ax3.plot(range(99, len(losses_filtered)), moving_avg, 
                        linewidth=2, color='darkorange', label='Moving Avg (100 ep)')
        ax3.set_xlabel('Episode', fontsize=12)
        ax3.set_ylabel('Loss', fontsize=12)
        ax3.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Epsilon Decay
        ax4 = axes[1, 1]
        ax4.plot(self.epsilon_history, linewidth=2, color='purple')
        ax4.set_xlabel('Episode', fontsize=12)
        ax4.set_ylabel('Epsilon', fontsize=12)
        ax4.set_title('Exploration Rate (Epsilon)', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.config.RESULTS_DIR, 'baseline_training_results.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✓ Training plots saved to {filepath}")
        plt.close()
    
    def close(self):
        """Clean up resources"""
        self.env.close()


def main():
    """Main entry point"""
    Config.print_config()
    
    trainer = BaselineTrainer(Config)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        print("Saving current progress...")
        trainer._save_checkpoint('interrupted')
        trainer._save_training_stats()
        trainer._plot_training_results()
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        trainer.close()
        print("Environment closed.")


if __name__ == "__main__":
    main()