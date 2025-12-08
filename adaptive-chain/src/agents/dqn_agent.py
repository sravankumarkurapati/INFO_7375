"""
DQN Agent using Stable-Baselines3 - OPTIMIZED VERSION
"""
import os
from typing import Optional, Callable
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.buffers import ReplayBuffer
import torch


class TrainingCallback(BaseCallback):
    """Custom callback for tracking training progress"""
    
    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_costs = []
        
    def _on_step(self) -> bool:
        """Called at every step"""
        # Check if episode ended
        if self.locals.get('dones')[0]:
            # Get episode info
            info = self.locals.get('infos')[0]
            
            if 'episode' in info:
                ep_reward = info['episode']['r']
                ep_length = info['episode']['l']
                
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)
                
                # Calculate cost (negative of reward)
                ep_cost = -ep_reward
                self.episode_costs.append(ep_cost)
                
                if self.verbose > 0 and len(self.episode_rewards) % 10 == 0:
                    recent_costs = self.episode_costs[-10:]
                    avg_cost = np.mean(recent_costs)
                    print(f"  Episode {len(self.episode_rewards)}: "
                          f"Avg Cost (last 10) = ${avg_cost:,.0f}")
        
        return True


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule
    
    Args:
        initial_value: Initial learning rate
        
    Returns:
        Schedule function
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0 (end)
        
        Args:
            progress_remaining: Fraction of training remaining
            
        Returns:
            Current learning rate
        """
        return progress_remaining * initial_value
    
    return func


def create_dqn_agent(
    env,
    learning_rate: float = 0.0003,  # INCREASED from 0.0001
    buffer_size: int = 100000,  # DOUBLED from 50000
    learning_starts: int = 2000,  # INCREASED from 1000
    batch_size: int = 128,  # DOUBLED from 64
    gamma: float = 0.99,
    tau: float = 0.01,  # DOUBLED from 0.005
    exploration_fraction: float = 0.5,  # INCREASED from 0.3
    exploration_initial_eps: float = 1.0,
    exploration_final_eps: float = 0.1,  # INCREASED from 0.05
    use_prioritized_replay: bool = True,  # NEW
    use_lr_schedule: bool = True,  # NEW
    tensorboard_log: Optional[str] = None,
    verbose: int = 1
) -> DQN:
    """
    Create DQN agent with optimized hyperparameters
    
    OPTIMIZATIONS:
    - Higher learning rate for faster learning
    - Larger buffer for more diverse experiences
    - More exploration (50% of training)
    - Prioritized Experience Replay (optional)
    - Learning rate schedule (optional)
    - Deeper network architecture
    """
    print("🤖 Creating OPTIMIZED DQN Agent...")
    print(f"  Learning Rate: {learning_rate} (with schedule: {use_lr_schedule})")
    print(f"  Buffer Size: {buffer_size:,}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Gamma: {gamma}")
    print(f"  Exploration: {exploration_initial_eps} → {exploration_final_eps} ({exploration_fraction*100:.0f}% of training)")
    print(f"  Prioritized Replay: {use_prioritized_replay}")
    
    # Prepare learning rate
    if use_lr_schedule:
        lr = linear_schedule(learning_rate)
        print(f"  Using linear LR schedule")
    else:
        lr = learning_rate
    
    # Prepare replay buffer
    replay_buffer_kwargs = {}
    if use_prioritized_replay:
        try:
            from stable_baselines3.common.buffers import PrioritizedReplayBuffer
            replay_buffer_class = PrioritizedReplayBuffer
            replay_buffer_kwargs = dict(alpha=0.6, beta=0.4)
            print(f"  Using Prioritized Experience Replay (alpha=0.6, beta=0.4)")
        except ImportError:
            print(f"  ⚠️ PrioritizedReplayBuffer not available, using standard replay")
            replay_buffer_class = ReplayBuffer
    else:
        replay_buffer_class = ReplayBuffer
    
    # Create agent
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=lr,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
        policy_kwargs=dict(
            net_arch=[512, 512, 256]  # DEEPER: 3 layers instead of 2
        ),
        replay_buffer_class=replay_buffer_class,
        replay_buffer_kwargs=replay_buffer_kwargs if use_prioritized_replay else None,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
        device='auto'
    )
    
    print(f"✅ OPTIMIZED DQN Agent created")
    print(f"  Policy Network: [512, 512, 256] (3 layers)")
    print(f"  Device: {model.device}")
    
    return model


def train_dqn_agent(
    model: DQN,
    total_timesteps: int = 100000,
    callback: Optional[BaseCallback] = None,
    progress_bar: bool = False
) -> DQN:
    """Train DQN agent"""
    print(f"\n🚀 Starting OPTIMIZED Training...")
    print(f"  Total Timesteps: {total_timesteps:,}")
    estimated_time = (total_timesteps / 700) / 60  # ~700 steps/sec
    print(f"  Estimated Time: {estimated_time:.1f} minutes\n")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=progress_bar
    )
    
    print(f"\n✅ Training Complete!")
    
    return model


def evaluate_dqn_agent(
    model: DQN,
    env,
    num_episodes: int = 10,
    deterministic: bool = True,
    verbose: bool = True
) -> dict:
    """Evaluate trained DQN agent"""
    if verbose:
        print(f"\n{'='*70}")
        print(f"Evaluating DQN Agent")
        print(f"{'='*70}")
    
    episode_rewards = []
    episode_costs = []
    episode_summaries = []
    
    for episode in range(num_episodes):
        obs, info = env.reset(seed=42 + episode)
        done = False
        episode_reward = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        # Get episode summary
        summary = env.get_episode_summary()
        episode_rewards.append(episode_reward)
        episode_costs.append(summary['total_cost'])
        episode_summaries.append(summary)
        
        if verbose:
            print(f"  Episode {episode + 1}/{num_episodes}: "
                  f"Reward=${episode_reward:,.0f}, Cost=${summary['total_cost']:,.0f}")
    
    # Calculate statistics
    results = {
        'policy_name': 'DQN Agent (Optimized)',
        'num_episodes': num_episodes,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_cost': np.mean(episode_costs),
        'std_cost': np.std(episode_costs),
        'min_cost': np.min(episode_costs),
        'max_cost': np.max(episode_costs),
        'episode_rewards': episode_rewards,
        'episode_costs': episode_costs,
        'summaries': episode_summaries
    }
    
    if verbose:
        print(f"\n📈 Results for DQN Agent (Optimized):")
        print(f"  Mean Cost: ${results['mean_cost']:,.2f} ± ${results['std_cost']:,.2f}")
        print(f"  Min Cost:  ${results['min_cost']:,.2f}")
        print(f"  Max Cost:  ${results['max_cost']:,.2f}")
    
    return results


def save_dqn_model(model: DQN, path: str, results: Optional[dict] = None):
    """Save trained model and results"""
    import json
    
    # Create directory
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save model
    model.save(path)
    print(f"💾 Model saved to: {path}")
    
    # Save results if provided
    if results:
        results_path = path.replace('.zip', '_results.json')
        
        # Convert numpy types to Python types
        results_to_save = {
            'policy_name': results['policy_name'],
            'mean_cost': float(results['mean_cost']),
            'std_cost': float(results['std_cost']),
            'mean_reward': float(results['mean_reward']),
            'episode_costs': [float(c) for c in results['episode_costs']]
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"📊 Results saved to: {results_path}")


def load_dqn_model(path: str, env) -> DQN:
    """Load trained model"""
    model = DQN.load(path, env=env)
    print(f"📂 Model loaded from: {path}")
    return model