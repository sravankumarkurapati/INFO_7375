#!/usr/bin/env python3
import gymnasium as gym
import ale_py

# Register ALE environments
gym.register_envs(ale_py)

def test_kangaroo_environment():
    """Test basic Kangaroo environment setup"""
    print("=" * 50)
    print("TESTING KANGAROO ENVIRONMENT")
    print("=" * 50)
    
    # Create environment
    env = gym.make('ALE/Kangaroo-v5', render_mode='rgb_array')
    
    # Reset environment
    observation, info = env.reset()
    
    # Print environment details
    print(f"\n1. Observation Space: {env.observation_space}")
    print(f"   - Shape: {observation.shape}")
    print(f"   - Data type: {observation.dtype}")
    
    print(f"\n2. Action Space: {env.action_space}")
    print(f"   - Number of actions: {env.action_space.n}")
    print(f"   - Action meanings: {env.unwrapped.get_action_meanings()}")
    
    print(f"\n3. Initial Info: {info}")
    
    # Take a random action
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    
    print(f"\n4. After Random Action {action}:")
    print(f"   - Reward: {reward}")
    print(f"   - Terminated: {terminated}")
    print(f"   - Truncated: {truncated}")
    
    env.close()
    print("\n✓ Environment test successful!")
    print("=" * 50)

if __name__ == "__main__":
    test_kangaroo_environment()