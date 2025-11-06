#!/usr/bin/env python3
"""
Run a single experiment with specified configuration
"""
import sys
import os
import gymnasium as gym
import ale_py
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append('../phase2_implementation')

from dqn_agent import DQNAgent
from frame_preprocessing import FramePreprocessor, FrameStack

gym.register_envs(ale_py)


class ExperimentRunner:
    """Runs a single experiment with given configuration"""
    
    def __init__(self, config):
        self.config = config
        
        # Create experiment directory
        self.exp_dir = os.path.join('experiment_results', config.EXPERIMENT_NAME)
        os.makedirs(self.exp_dir, exist_ok=True)
        
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
        
        print("\n" + "=" * 70)
        print(f"EXPERIMENT: {config.EXPERIMENT_NAME}")
        print("=" * 70)
        print(f"Description: {config.DESCRIPTION}")
        print(f"Episodes: {config.TOTAL_EPISODES}")
        print(f"Max Steps: {config.MAX_STEPS}")
        print(f"Learning Rate: {config.LEARNING_RATE}")
        print(f"Gamma: {config.GAMMA}")
        print(f"Epsilon Decay: {config.EPSILON_DECAY}")
        print("=" * 70 + "\n")
    
    def train(self):
        """Run the experiment"""
        print(f"Starting experiment: {self.config.EXPERIMENT_NAME}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
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
                self._log_progress(episode, start_time)
            
            # Save model
            if episode % self.config.SAVE_FREQ == 0:
                self._save_checkpoint(episode)
        
        # Final save
        self._save_checkpoint('final')
        self._save_stats()
        self._plot_results()
        
        total_time = time.time() - start_time
        
        print(f"\n{'=' * 70}")
        print(f"EXPERIMENT COMPLETE: {self.config.EXPERIMENT_NAME}")
        print(f"Total Time: {total_time/60:.1f} minutes")
        print(f"Mean Reward: {np.mean(self.episode_rewards):.2f}")
        print(f"Max Reward: {np.max(self.episode_rewards):.2f}")
        print(f"Mean Steps: {np.mean(self.episode_lengths):.1f}")
        print(f"{'=' * 70}\n")
    
    def _train_episode(self):
        """Train one episode"""
        observation, _ = self.env.reset()
        processed_frame = self.preprocessor.preprocess(observation)
        self.frame_stack.reset(processed_frame)
        state = self.frame_stack.get_state()
        
        episode_reward = 0
        episode_loss = []
        step = 0
        
        for step in range(self.config.MAX_STEPS):
            state_input = np.transpose(state, (2, 0, 1))
            action = self.agent.select_action(state_input)
            
            next_observation, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            processed_next_frame = self.preprocessor.preprocess(next_observation)
            self.frame_stack.append(processed_next_frame)
            next_state = self.frame_stack.get_state()
            
            next_state_input = np.transpose(next_state, (2, 0, 1))
            self.agent.store_transition(state_input, action, reward, next_state_input, done)
            
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
    
    def _log_progress(self, episode, start_time):
        """Log training progress"""
        window = self.config.LOG_FREQ
        avg_reward = np.mean(self.episode_rewards[-window:])
        avg_length = np.mean(self.episode_lengths[-window:])
        max_reward = np.max(self.episode_rewards)
        
        elapsed = time.time() - start_time
        eta = (elapsed / episode) * (self.config.TOTAL_EPISODES - episode)
        
        print(f"Ep {episode}/{self.config.TOTAL_EPISODES} | "
              f"Reward: {avg_reward:.1f} | "
              f"Max: {max_reward:.1f} | "
              f"Steps: {avg_length:.0f} | "
              f"ε: {self.agent.epsilon:.3f} | "
              f"Time: {elapsed/60:.0f}m | "
              f"ETA: {eta/60:.0f}m")
    
    def _save_checkpoint(self, episode):
        """Save model checkpoint"""
        filepath = os.path.join(self.exp_dir, f'model_ep{episode}.pth')
        self.agent.save(filepath)
    
    def _save_stats(self):
        """Save experiment statistics"""
        stats = {
            'experiment_name': self.config.EXPERIMENT_NAME,
            'description': self.config.DESCRIPTION,
            'config': {
                'total_episodes': self.config.TOTAL_EPISODES,
                'max_steps': self.config.MAX_STEPS,
                'learning_rate': self.config.LEARNING_RATE,
                'gamma': self.config.GAMMA,
                'epsilon_start': self.config.EPSILON_START,
                'epsilon_end': self.config.EPSILON_END,
                'epsilon_decay': self.config.EPSILON_DECAY
            },
            'results': {
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths,
                'epsilon_history': self.epsilon_history
            },
            'summary': {
                'mean_reward': float(np.mean(self.episode_rewards)),
                'std_reward': float(np.std(self.episode_rewards)),
                'max_reward': float(np.max(self.episode_rewards)),
                'min_reward': float(np.min(self.episode_rewards)),
                'mean_length': float(np.mean(self.episode_lengths)),
                'final_epsilon': float(self.epsilon_history[-1])
            }
        }
        
        filepath = os.path.join(self.exp_dir, 'experiment_stats.json')
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✓ Stats saved to {filepath}")
    
    def _plot_results(self):
        """Plot experiment results"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{self.config.EXPERIMENT_NAME}\n{self.config.DESCRIPTION}', 
                     fontsize=14, fontweight='bold')
        
        # Rewards
        ax1 = axes[0, 0]
        ax1.plot(self.episode_rewards, alpha=0.4, label='Episode Reward')
        if len(self.episode_rewards) >= 50:
            ma = np.convolve(self.episode_rewards, np.ones(50)/50, mode='valid')
            ax1.plot(range(49, len(self.episode_rewards)), ma, 
                    linewidth=2, label='MA(50)')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Lengths
        ax2 = axes[0, 1]
        ax2.plot(self.episode_lengths, alpha=0.4, label='Episode Length')
        if len(self.episode_lengths) >= 50:
            ma = np.convolve(self.episode_lengths, np.ones(50)/50, mode='valid')
            ax2.plot(range(49, len(self.episode_lengths)), ma, 
                    linewidth=2, label='MA(50)')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.set_title('Episode Lengths')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Loss
        ax3 = axes[1, 0]
        losses = [l for l in self.episode_losses if l is not None]
        if losses:
            ax3.plot(losses, alpha=0.4, label='Loss')
            if len(losses) >= 50:
                ma = np.convolve(losses, np.ones(50)/50, mode='valid')
                ax3.plot(range(49, len(losses)), ma, linewidth=2, label='MA(50)')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Loss')
        ax3.set_title('Training Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Epsilon
        ax4 = axes[1, 1]
        ax4.plot(self.epsilon_history, linewidth=2, color='purple')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Epsilon')
        ax4.set_title('Exploration Rate')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.exp_dir, 'training_plot.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to {filepath}")
        plt.close()
    
    def close(self):
        """Clean up"""
        self.env.close()


def main():
    """Run experiment from command line"""
    if len(sys.argv) < 2:
        print("Usage: python3 run_single_experiment.py <experiment_number>")
        print("\nAvailable experiments:")
        from experiment_configs import EXPERIMENTS
        for i, exp in enumerate(EXPERIMENTS, 1):
            print(f"  {i}: {exp.EXPERIMENT_NAME} - {exp.DESCRIPTION}")
        sys.exit(1)
    
    exp_num = int(sys.argv[1]) - 1
    
    from experiment_configs import EXPERIMENTS
    
    if exp_num < 0 or exp_num >= len(EXPERIMENTS):
        print(f"Error: Experiment number must be between 1 and {len(EXPERIMENTS)}")
        sys.exit(1)
    
    config_class = EXPERIMENTS[exp_num]
    
    runner = ExperimentRunner(config_class)
    
    try:
        runner.train()
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted!")
        runner._save_checkpoint('interrupted')
        runner._save_stats()
        runner._plot_results()
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        runner.close()


if __name__ == "__main__":
    main()