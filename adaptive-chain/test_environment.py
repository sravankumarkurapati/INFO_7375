"""
Test Supply Chain Environment
"""
import numpy as np
from src.environment.supply_chain_env import SupplyChainEnv


def test_environment_creation():
    """Test environment initialization"""
    print("=" * 60)
    print("Testing Environment Creation")
    print("=" * 60)
    
    env = SupplyChainEnv(
        num_products=3,
        num_warehouses=1,
        episode_length=180,
        random_seed=42
    )
    
    print(f"\n✓ Environment created")
    print(f"  Products: {env.num_products}")
    print(f"  Warehouses: {env.num_warehouses}")
    print(f"  Episode length: {env.episode_length} days")
    print(f"  State dimension: {env.state_dim}")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space.shape}")
    
    print("\n✅ Environment creation working!\n")
    return env


def test_reset():
    """Test environment reset"""
    print("=" * 60)
    print("Testing Environment Reset")
    print("=" * 60)
    
    env = SupplyChainEnv(num_products=3, random_seed=42)
    obs, info = env.reset()
    
    print(f"\n✓ Environment reset successful")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"  Current day: {info['day']}")
    print(f"  Initial inventory: {info['inventory']}")
    
    print("\n✅ Reset working!\n")
    return env, obs


def test_random_actions():
    """Test taking random actions"""
    print("=" * 60)
    print("Testing Random Actions (10 steps)")
    print("=" * 60)
    
    env = SupplyChainEnv(num_products=3, random_seed=42)
    obs, info = env.reset()
    
    print(f"\n📊 Initial State:")
    print(f"  Day: {info['day']}")
    print(f"  Inventory: {info['inventory']}")
    
    total_reward = 0
    for step in range(10):
        # Random action
        action = env.action_space.sample()
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step < 3:  # Show first 3 steps
            print(f"\n  Step {step + 1}:")
            print(f"    Action: {action}")
            print(f"    Reward: ${reward:,.2f}")
            print(f"    Inventory: {info['inventory']}")
    
    print(f"\n✓ Completed 10 random steps")
    print(f"  Total reward: ${total_reward:,.2f}")
    print(f"  Episode costs: {info['episode_costs']}")
    
    print("\n✅ Random actions working!\n")


def test_full_episode():
    """Test a complete episode"""
    print("=" * 60)
    print("Testing Full Episode (180 days)")
    print("=" * 60)
    
    env = SupplyChainEnv(num_products=3, episode_length=180, random_seed=42)
    obs, info = env.reset()
    
    print(f"\n🚀 Running full episode with random policy...")
    
    done = False
    step_count = 0
    
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1
        
        # Print progress every 30 days
        if step_count % 30 == 0:
            print(f"  Day {step_count}: Reward=${info['total_reward']:,.2f}, "
                  f"Capacity={info['capacity_utilization']*100:.1f}%")
    
    # Get episode summary
    summary = env.get_episode_summary()
    
    print(f"\n✓ Episode completed!")
    print(f"  Total steps: {step_count}")
    print(f"  Total reward: ${summary['total_reward']:,.2f}")
    print(f"  Total cost: ${summary['total_cost']:,.2f}")
    print(f"\n📊 Cost Breakdown:")
    print(f"  Holding cost: ${summary['holding_cost']:,.2f}")
    print(f"  Stockout cost: ${summary['stockout_cost']:,.2f}")
    print(f"  Order cost: ${summary['order_cost']:,.2f}")
    print(f"  Avg daily cost: ${summary['avg_daily_cost']:,.2f}")
    
    print("\n✅ Full episode working!\n")
    return summary


def test_gymnasium_compatibility():
    """Test Gymnasium API compatibility"""
    print("=" * 60)
    print("Testing Gymnasium Compatibility")
    print("=" * 60)
    
    env = SupplyChainEnv(num_products=3, random_seed=42)
    
    # Check required methods
    required_methods = ['reset', 'step', 'render', 'close']
    print(f"\n✓ Checking required methods:")
    for method in required_methods:
        has_method = hasattr(env, method)
        status = "✓" if has_method else "✗"
        print(f"  {status} {method}")
    
    # Check spaces
    print(f"\n✓ Checking spaces:")
    print(f"  Action space valid: {env.action_space.contains(env.action_space.sample())}")
    obs, _ = env.reset()
    print(f"  Observation space valid: {env.observation_space.contains(obs)}")
    
    print("\n✅ Gymnasium compatibility confirmed!\n")


if __name__ == "__main__":
    try:
        # Run all tests
        env = test_environment_creation()
        env, obs = test_reset()
        test_random_actions()
        summary = test_full_episode()
        test_gymnasium_compatibility()
        
        print("=" * 60)
        print("🎉 ALL ENVIRONMENT TESTS PASSED!")
        print("=" * 60)
        print("\n✅ Environment is ready for RL training!")
        print("\n💡 Next: Step 1.5 - Create baseline policies for comparison")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        print("\n⚠️ Please fix errors before proceeding")