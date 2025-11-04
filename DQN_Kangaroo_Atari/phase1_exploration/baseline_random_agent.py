#!/usr/bin/env python3
import gymnasium as gym
import ale_py
import numpy as np
import matplotlib.pyplot as plt

gym.register_envs(ale_py)

def run_random_agent(num_episodes=100, max_steps=1000):
    """
    Run a random agent to establish baseline performance
    """
    env = gym.make('ALE/Kangaroo-v5', render_mode='rgb_array')
    
    episode_rewards = []
    episode_lengths = []
    
    print("=" * 60)
    print("RUNNING BASELINE RANDOM AGENT")
    print(f"Episodes: {num_episodes}, Max steps per episode: {max_steps}")
    print("=" * 60)
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        episode_reward = 0
        steps = 0
        
        for step in range(max_steps):
            # Take random action
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}/{num_episodes} - "
                  f"Avg Reward (last 10): {avg_reward:.2f}")
    
    env.close()
    
    # Calculate statistics
    print("\n" + "=" * 60)
    print("BASELINE RANDOM AGENT RESULTS")
    print("=" * 60)
    print(f"Total episodes: {num_episodes}")
    print(f"\nReward Statistics:")
    print(f"   - Mean reward: {np.mean(episode_rewards):.2f}")
    print(f"   - Std reward: {np.std(episode_rewards):.2f}")
    print(f"   - Min reward: {np.min(episode_rewards):.2f}")
    print(f"   - Max reward: {np.max(episode_rewards):.2f}")
    print(f"\nEpisode Length Statistics:")
    print(f"   - Mean steps: {np.mean(episode_lengths):.2f}")
    print(f"   - Std steps: {np.std(episode_lengths):.2f}")
    print(f"   - Min steps: {np.min(episode_lengths):.0f}")
    print(f"   - Max steps: {np.max(episode_lengths):.0f}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot rewards
    ax1.plot(episode_rewards, alpha=0.6, label='Episode Reward')
    ax1.plot(np.convolve(episode_rewards, np.ones(10)/10, mode='valid'), 
             linewidth=2, label='Moving Average (10 episodes)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Random Agent Performance - Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot episode lengths
    ax2.plot(episode_lengths, alpha=0.6, label='Episode Length')
    ax2.plot(np.convolve(episode_lengths, np.ones(10)/10, mode='valid'), 
             linewidth=2, label='Moving Average (10 episodes)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('Random Agent Performance - Episode Length')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('baseline_random_agent.png', dpi=150, bbox_inches='tight')
    print("\n✓ Results saved to 'baseline_random_agent.png'")
    plt.close()
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_steps': np.mean(episode_lengths),
        'std_steps': np.std(episode_lengths)
    }

if __name__ == "__main__":
    baseline_stats = run_random_agent(num_episodes=100, max_steps=1000)