#!/usr/bin/env python3
import gymnasium as gym
import ale_py
import numpy as np
import matplotlib.pyplot as plt

gym.register_envs(ale_py)

def explore_kangaroo():
    """Detailed exploration of Kangaroo environment"""
    
    env = gym.make('ALE/Kangaroo-v5', render_mode='rgb_array')
    observation, info = env.reset()
    
    print("=" * 60)
    print("KANGAROO ENVIRONMENT ANALYSIS")
    print("=" * 60)
    
    # ===== STATE SPACE =====
    print("\n📊 STATE SPACE DETAILS:")
    print(f"   - Raw observation shape: {observation.shape}")
    print(f"   - Raw observation size: {np.prod(observation.shape)} values")
    print(f"   - Data type: {observation.dtype}")
    print(f"   - Value range: [{observation.min()}, {observation.max()}]")
    
    # Calculate Q-table size (theoretical)
    state_size = np.prod(observation.shape)
    action_size = env.action_space.n
    
    print(f"\n   💡 Theoretical Q-table considerations:")
    print(f"      - State space size: {state_size} continuous values")
    print(f"      - If each pixel has 256 possible values:")
    print(f"        Possible states = 256^{state_size} (IMPOSSIBLY LARGE!)")
    print(f"      - This is why we need DEEP Q-Learning (function approximation)")
    print(f"        instead of tabular Q-learning!")
    
    # ===== ACTION SPACE =====
    print(f"\n🎮 ACTION SPACE DETAILS:")
    print(f"   - Number of discrete actions: {action_size}")
    print(f"   - Action meanings:")
    for i, action_name in enumerate(env.unwrapped.get_action_meanings()):
        print(f"      Action {i}: {action_name}")
    
    # ===== Q-TABLE SIZE =====
    print(f"\n📋 Q-TABLE INFORMATION:")
    print(f"   - For DQN: We use a neural network to approximate Q-values")
    print(f"   - Input to network: {observation.shape} (210x160x3 RGB image)")
    print(f"   - Output from network: {action_size} Q-values (one per action)")
    print(f"   - Network parameters: ~Millions (much smaller than full Q-table!)")
    
    env.close()
    print("\n" + "=" * 60)

def visualize_sample_frames():
    """Visualize sample frames from Kangaroo"""
    
    env = gym.make('ALE/Kangaroo-v5', render_mode='rgb_array')
    observation, info = env.reset()
    
    frames = [observation]
    
    # Collect a few frames
    for _ in range(5):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        frames.append(observation)
        
        if terminated or truncated:
            observation, info = env.reset()
    
    # Plot frames
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Sample Frames from Kangaroo Environment', fontsize=16)
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(frames):
            ax.imshow(frames[idx])
            ax.set_title(f'Frame {idx}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('kangaroo_sample_frames.png', dpi=150, bbox_inches='tight')
    print("\n✓ Sample frames saved to 'kangaroo_sample_frames.png'")
    plt.close()
    
    env.close()

if __name__ == "__main__":
    explore_kangaroo()
    print("\n🖼️  Generating sample frame visualizations...")
    visualize_sample_frames()