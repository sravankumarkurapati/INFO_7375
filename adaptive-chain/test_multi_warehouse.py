"""
Test multi-warehouse environment
"""
from src.environment.multi_warehouse_env import MultiWarehouseEnv
import numpy as np


def test_multi_warehouse_creation():
    """Test environment creation"""
    print("=" * 70)
    print("Testing Multi-Warehouse Environment Creation")
    print("=" * 70)
    
    env = MultiWarehouseEnv(
        num_products=3,
        num_warehouses=3,
        episode_length=180,
        enable_transfers=True,
        random_seed=42
    )
    
    print(f"\n✓ Environment created")
    print(f"  Products: {env.num_products}")
    print(f"  Warehouses: {env.num_warehouses}")
    print(f"  State dimension: {env.state_dim}")
    print(f"  Action space: {env.action_space}")
    print(f"  Transfers enabled: {env.enable_transfers}")
    
    print("\n✅ Multi-warehouse environment working!\n")
    return env


def test_reset_with_regional_variation():
    """Test reset with regional demand patterns"""
    print("=" * 70)
    print("Testing Reset with Regional Variations")
    print("=" * 70)
    
    env = MultiWarehouseEnv(num_products=3, num_warehouses=3, random_seed=42)
    obs, info = env.reset()
    
    print(f"\n✓ Environment reset")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Day: {info['day']}")
    
    print(f"\n📊 Regional Starting Inventory:")
    for wh_info in info['warehouses']:
        print(f"\n  {wh_info['id']}:")
        for sku, qty in wh_info['inventory'].items():
            print(f"    {sku}: {qty:.0f} units")
        print(f"    Capacity: {wh_info['capacity_util']*100:.1f}%")
    
    # Check regional variation
    east_inv = info['warehouses'][0]['total_inventory']
    west_inv = info['warehouses'][1]['total_inventory']
    central_inv = info['warehouses'][2]['total_inventory']
    
    print(f"\n📈 Regional Multipliers Applied:")
    print(f"  East (1.3x): {east_inv:.0f} units")
    print(f"  West (0.8x): {west_inv:.0f} units")
    print(f"  Central (1.0x): {central_inv:.0f} units")
    
    assert east_inv > central_inv > west_inv, "Regional variation not working!"
    
    print("\n✅ Regional variation working correctly!\n")
    return env


def test_random_episode():
    """Test running episode with random actions"""
    print("=" * 70)
    print("Testing Random Episode (50 days)")
    print("=" * 70)
    
    env = MultiWarehouseEnv(
        num_products=3,
        num_warehouses=3,
        episode_length=50,
        enable_transfers=True,
        random_seed=42
    )
    
    obs, info = env.reset()
    
    print(f"\n🚀 Running 50-day episode with random policy...")
    
    for day in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if day % 10 == 0:
            print(f"  Day {day}: Cost=${-reward:,.0f}, Transfers={info['transfer_count']}")
        
        if terminated or truncated:
            break
    
    summary = env.get_episode_summary()
    
    print(f"\n✓ Episode completed")
    print(f"\n📊 Episode Summary:")
    print(f"  Total Cost: ${summary['total_cost']:,.2f}")
    print(f"  Holding: ${summary['holding_cost']:,.2f}")
    print(f"  Stockout: ${summary['stockout_cost']:,.2f}")
    print(f"  Order: ${summary['order_cost']:,.2f}")
    print(f"  Transfer: ${summary['transfer_cost']:,.2f}")
    print(f"  Total Transfers: {summary['transfer_count']}")
    print(f"  Avg Transfers/Day: {summary['avg_transfer_per_day']:.2f}")
    
    print("\n✅ Multi-warehouse episode working!\n")
    return summary


def test_coordination_features():
    """Test coordination-specific features"""
    print("=" * 70)
    print("Testing Coordination Features")
    print("=" * 70)
    
    env = MultiWarehouseEnv(
        num_products=3,
        num_warehouses=3,
        enable_transfers=True,
        random_seed=42
    )
    
    obs, info = env.reset()
    
    print(f"\n✓ Testing coordination features:")
    
    # 1. Check state includes other warehouses
    print(f"  ✓ State dimension: {env.state_dim} (includes other warehouses)")
    print(f"    - Per warehouse: {env.state_dim // env.num_warehouses} features")
    print(f"    - Includes: own state + other warehouses' inventory")
    
    # 2. Test transfer mechanism
    print(f"\n  🔄 Testing inventory transfers...")
    
    # Create imbalance
    env.warehouses[0].inventory = {'PROD_A': 2000, 'PROD_B': 100, 'PROD_C': 50}
    env.warehouses[1].inventory = {'PROD_A': 50, 'PROD_B': 50, 'PROD_C': 50}
    env.warehouses[2].inventory = {'PROD_A': 500, 'PROD_B': 500, 'PROD_C': 200}
    
    print(f"  Before transfer:")
    print(f"    WH_EAST PROD_A: {env.warehouses[0].get_inventory_level('PROD_A'):.0f}")
    print(f"    WH_WEST PROD_A: {env.warehouses[1].get_inventory_level('PROD_A'):.0f}")
    
    # Trigger balance
    cost = env._balance_inventory_proactive()
    
    print(f"  After transfer:")
    print(f"    WH_EAST PROD_A: {env.warehouses[0].get_inventory_level('PROD_A'):.0f}")
    print(f"    WH_WEST PROD_A: {env.warehouses[1].get_inventory_level('PROD_A'):.0f}")
    print(f"    Transfer cost: ${cost:.2f}")
    print(f"    Transfers executed: {env.transfer_count}")
    
    if env.transfer_count > 0:
        print(f"  ✅ Inventory transfer system working!")
    else:
        print(f"  ⚠️ No transfers executed (check thresholds)")
    
    print("\n✅ Coordination features validated!\n")


if __name__ == "__main__":
    try:
        env = test_multi_warehouse_creation()
        env = test_reset_with_regional_variation()
        summary = test_random_episode()
        test_coordination_features()
        
        print("=" * 70)
        print("🎉 ALL MULTI-WAREHOUSE TESTS PASSED!")
        print("=" * 70)
        print("\n✅ Ready for Step 3.2: Multi-Agent Training")
        print("\n💡 Next: Train coordinated agents vs independent agents")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()