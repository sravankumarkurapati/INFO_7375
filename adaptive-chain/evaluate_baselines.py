"""
Evaluate all baseline policies
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.environment.supply_chain_env import SupplyChainEnv
from src.agents.baseline_policies import create_baseline_policies, evaluate_policy
import json


def run_baseline_evaluation(num_episodes: int = 10):
    """
    Run evaluation of all baseline policies
    
    Args:
        num_episodes: Number of episodes per policy
    """
    print("=" * 70)
    print("BASELINE POLICY EVALUATION")
    print("=" * 70)
    
    # Create environment
    env = SupplyChainEnv(
        num_products=3,
        num_warehouses=1,
        episode_length=180,
        random_seed=42
    )
    
    print(f"\n📊 Environment: {env.num_products} products, {env.episode_length} days")
    print(f"🎯 Evaluating each policy over {num_episodes} episodes\n")
    
    # Create all baseline policies
    policies = create_baseline_policies(env)
    
    # Evaluate each policy
    all_results = {}
    
    for policy_name, policy in policies.items():
        print(f"\n{'='*70}")
        print(f"Evaluating: {policy.name}")
        print(f"{'='*70}")
        
        results = evaluate_policy(env, policy, num_episodes=num_episodes, verbose=True)
        all_results[policy_name] = results
        
        print(f"\n📈 Results for {policy.name}:")
        print(f"  Mean Cost: ${results['mean_cost']:,.2f} ± ${results['std_cost']:,.2f}")
        print(f"  Min Cost:  ${results['min_cost']:,.2f}")
        print(f"  Max Cost:  ${results['max_cost']:,.2f}")
    
    # Create comparison
    print(f"\n{'='*70}")
    print("POLICY COMPARISON")
    print(f"{'='*70}\n")
    
    comparison_df = create_comparison_table(all_results)
    print(comparison_df.to_string())
    
    # Save results
    save_results(all_results, comparison_df)
    
    # Create visualizations
    create_visualizations(all_results)
    
    print(f"\n{'='*70}")
    print("✅ BASELINE EVALUATION COMPLETE")
    print(f"{'='*70}")
    print("\n📁 Results saved to:")
    print("  - data/baseline_results.json")
    print("  - data/baseline_comparison.csv")
    print("  - data/baseline_comparison.png")
    print("  - data/baseline_cost_distribution.png")
    
    return all_results


def create_comparison_table(results: dict) -> pd.DataFrame:
    """Create comparison table"""
    data = []
    
    for policy_name, policy_results in results.items():
        data.append({
            'Policy': policy_results['policy_name'],
            'Mean Cost ($)': f"{policy_results['mean_cost']:,.0f}",
            'Std Dev ($)': f"{policy_results['std_cost']:,.0f}",
            'Min Cost ($)': f"{policy_results['min_cost']:,.0f}",
            'Max Cost ($)': f"{policy_results['max_cost']:,.0f}",
            'Mean Reward ($)': f"{policy_results['mean_reward']:,.0f}"
        })
    
    df = pd.DataFrame(data)
    return df


def save_results(results: dict, comparison_df: pd.DataFrame):
    """Save results to files"""
    import os
    os.makedirs('data', exist_ok=True)
    
    # Save detailed results (JSON)
    results_to_save = {}
    for policy_name, policy_results in results.items():
        results_to_save[policy_name] = {
            'policy_name': policy_results['policy_name'],
            'mean_cost': float(policy_results['mean_cost']),
            'std_cost': float(policy_results['std_cost']),
            'mean_reward': float(policy_results['mean_reward']),
            'episode_costs': [float(c) for c in policy_results['episode_costs']]
        }
    
    with open('data/baseline_results.json', 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    # Save comparison table (CSV)
    comparison_df.to_csv('data/baseline_comparison.csv', index=False)


def create_visualizations(results: dict):
    """Create comparison visualizations"""
    import os
    os.makedirs('data', exist_ok=True)
    
    # Figure 1: Bar chart comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    policies = list(results.keys())
    means = [results[p]['mean_cost'] for p in policies]
    stds = [results[p]['std_cost'] for p in policies]
    names = [results[p]['policy_name'] for p in policies]
    
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    bars = ax.bar(names, means, yerr=stds, capsize=10, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Total Cost ($)', fontsize=12, fontweight='bold')
    ax.set_title('Baseline Policy Comparison\n(Lower is Better)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'${mean:,.0f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('data/baseline_comparison.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: data/baseline_comparison.png")
    
    # Figure 2: Box plot distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    
    data_for_boxplot = [results[p]['episode_costs'] for p in policies]
    bp = ax.boxplot(data_for_boxplot, labels=names, patch_artist=True)
    
    # Color boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Total Cost ($)', fontsize=12, fontweight='bold')
    ax.set_title('Cost Distribution Across Episodes', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/baseline_cost_distribution.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: data/baseline_cost_distribution.png")
    
    plt.close('all')


if __name__ == "__main__":
    results = run_baseline_evaluation(num_episodes=10)