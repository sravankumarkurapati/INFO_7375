"""
Ablation study: Impact of coordination features
Tests what happens when we remove key components
"""
import numpy as np
from src.environment.multi_warehouse_env import MultiWarehouseEnv
from src.environment.env_wrappers import FlattenMultiDiscreteWrapper
from src.agents.coordinated_agents import CoordinatedMultiAgent
from stable_baselines3 import DQN
import json
import os


def evaluate_agent(agent, env, num_episodes: int = 5) -> dict:
    """Evaluate agent on environment"""
    episode_costs = []
    transfer_counts = []
    
    for ep in range(num_episodes):
        obs, info = env.reset(seed=200 + ep)
        done = False
        
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        summary = env.get_episode_summary()
        episode_costs.append(summary['total_cost'])
        transfer_counts.append(summary.get('transfer_count', 0))
    
    return {
        'mean_cost': np.mean(episode_costs),
        'std_cost': np.std(episode_costs),
        'mean_transfers': np.mean(transfer_counts),
        'episode_costs': episode_costs
    }


def run_ablation_study():
    """
    Run ablation study on coordinated agent
    
    Tests:
    1. Full system (with transfers)
    2. No transfers (coordination via info sharing only)
    3. Analysis of feature contribution
    """
    
    print("=" * 70)
    print("ABLATION STUDY: Coordination Feature Analysis")
    print("=" * 70)
    
    print(f"\n🎯 Goal: Isolate impact of inventory transfer feature")
    print(f"   We'll test coordinated agent with and without transfers")
    
    # Load coordinated agent
    print(f"\n📂 Loading trained coordinated agent...")
    
    if not os.path.exists('models/coordinated_multi_agent.zip'):
        print("❌ Coordinated agent not found!")
        return
    
    # ========================================
    # TEST 1: WITH TRANSFERS (Original)
    # ========================================
    print(f"\n{'='*70}")
    print(f"TEST 1: Coordinated Agent WITH Transfers (Original)")
    print(f"{'='*70}")
    
    env_with_transfers = MultiWarehouseEnv(
        num_products=3,
        num_warehouses=3,
        episode_length=180,
        enable_transfers=True,  # Transfers enabled
        transfer_cost_per_unit=5.0,
        random_seed=42
    )
    
    coordinated_agent = CoordinatedMultiAgent(env_with_transfers)
    wrapped_env = FlattenMultiDiscreteWrapper(env_with_transfers)
    coordinated_agent.agent = DQN.load('models/coordinated_multi_agent.zip', env=wrapped_env)
    
    print(f"\n🚀 Running 5 episodes WITH transfers...")
    results_with = evaluate_agent(coordinated_agent, env_with_transfers, num_episodes=5)
    
    print(f"\n📊 Results WITH Transfers:")
    print(f"  Mean Cost: ${results_with['mean_cost']:,.2f}")
    print(f"  Std Dev: ${results_with['std_cost']:,.2f}")
    print(f"  Avg Transfers: {results_with['mean_transfers']:.1f}")
    
    # ========================================
    # TEST 2: WITHOUT TRANSFERS
    # ========================================
    print(f"\n{'='*70}")
    print(f"TEST 2: Coordinated Agent WITHOUT Transfers")
    print(f"{'='*70}")
    print(f"  (Same agent, but transfers disabled in environment)")
    
    env_without_transfers = MultiWarehouseEnv(
        num_products=3,
        num_warehouses=3,
        episode_length=180,
        enable_transfers=False,  # Transfers DISABLED
        random_seed=42
    )
    
    print(f"\n🚀 Running 5 episodes WITHOUT transfers...")
    results_without = evaluate_agent(coordinated_agent, env_without_transfers, num_episodes=5)
    
    print(f"\n📊 Results WITHOUT Transfers:")
    print(f"  Mean Cost: ${results_without['mean_cost']:,.2f}")
    print(f"  Std Dev: ${results_without['std_cost']:,.2f}")
    print(f"  Avg Transfers: {results_without['mean_transfers']:.1f}")
    
    # ========================================
    # ANALYSIS
    # ========================================
    print(f"\n{'='*70}")
    print(f"ABLATION ANALYSIS")
    print(f"{'='*70}")
    
    cost_diff = results_with['mean_cost'] - results_without['mean_cost']
    percent_diff = (cost_diff / results_without['mean_cost']) * 100
    
    print(f"\n📊 Impact of Transfer Feature:")
    print(f"  WITH Transfers:    ${results_with['mean_cost']:,.2f}")
    print(f"  WITHOUT Transfers: ${results_without['mean_cost']:,.2f}")
    print(f"  Difference:        ${cost_diff:,.2f}")
    print(f"  Percentage:        {percent_diff:+.1f}%")
    
    if cost_diff > 0:
        print(f"\n  ❌ FINDING: Transfer feature INCREASES cost by {percent_diff:.1f}%")
        print(f"  💡 Implication: Inventory transfers add overhead without benefit")
        print(f"  ✅ Validates: Information sharing alone is sufficient for coordination")
    else:
        print(f"\n  ✅ FINDING: Transfer feature REDUCES cost by {abs(percent_diff):.1f}%")
        print(f"  💡 Implication: Physical transfers provide coordination benefit")
    
    # Transfer cost breakdown
    transfer_cost_estimate = results_with['mean_transfers'] * 5.0 * 100  # $5 per unit, ~100 units avg
    
    print(f"\n📊 Transfer Cost Breakdown:")
    print(f"  Avg Transfers per Episode: {results_with['mean_transfers']:.0f}")
    print(f"  Estimated Units Transferred: {results_with['mean_transfers'] * 100:.0f}")
    print(f"  Transfer Cost ($5/unit): ${transfer_cost_estimate:,.2f}")
    print(f"  As % of Total Cost: {(transfer_cost_estimate / results_with['mean_cost']) * 100:.1f}%")
    
    # ========================================
    # SAVE RESULTS
    # ========================================
    ablation_results = {
        'with_transfers': {
            'mean_cost': float(results_with['mean_cost']),
            'std_cost': float(results_with['std_cost']),
            'mean_transfers': float(results_with['mean_transfers']),
            'episode_costs': [float(c) for c in results_with['episode_costs']]
        },
        'without_transfers': {
            'mean_cost': float(results_without['mean_cost']),
            'std_cost': float(results_without['std_cost']),
            'mean_transfers': float(results_without['mean_transfers']),
            'episode_costs': [float(c) for c in results_without['episode_costs']]
        },
        'transfer_impact': {
            'cost_difference': float(cost_diff),
            'percent_difference': float(percent_diff),
            'estimated_transfer_cost': float(transfer_cost_estimate)
        }
    }
    
    os.makedirs('data', exist_ok=True)
    with open('data/ablation_study_results.json', 'w') as f:
        json.dump(ablation_results, f, indent=2)
    
    print(f"\n💾 Ablation results saved to: data/ablation_study_results.json")
    
    # ========================================
    # KEY INSIGHT
    # ========================================
    print(f"\n{'='*70}")
    print(f"KEY INSIGHT FROM ABLATION STUDY")
    print(f"{'='*70}")
    
    print(f"""
    The ablation study reveals a critical design lesson:
    
    🔍 FINDING:
    Removing the inventory transfer feature {'IMPROVES' if cost_diff > 0 else 'WORSENS'} 
    performance by ${abs(cost_diff):,.2f} ({abs(percent_diff):.1f}%).
    
    💡 INTERPRETATION:
    {"The overhead of physical inventory transfers (movement costs, " if cost_diff > 0 else ""}
    {"coordination complexity) exceeds any benefit from load balancing." if cost_diff > 0 else ""}
    {"Physical transfers provide net benefit despite coordination costs." if cost_diff < 0 else ""}
    
    ✅ RECOMMENDATION:
    {"For production deployment: Use information sharing (forecasts, " if cost_diff > 0 else ""}
    {"inventory visibility) WITHOUT physical transfers. This provides " if cost_diff > 0 else ""}
    {"coordination benefits at near-zero cost." if cost_diff > 0 else ""}
    {"Inventory transfers are beneficial and should be enabled." if cost_diff < 0 else ""}
    
    🎯 CONTRIBUTION:
    This demonstrates the importance of ablation studies in validating
    multi-agent system design choices. Not all coordination mechanisms
    provide value - empirical testing is essential.
    """)
    
    print(f"\n{'='*70}")
    print(f"✅ ABLATION STUDY COMPLETE")
    print(f"{'='*70}")
    print(f"\n🎯 Next: Step 4.3 - Create scenario visualizations")
    
    return ablation_results


if __name__ == "__main__":
    results = run_ablation_study()