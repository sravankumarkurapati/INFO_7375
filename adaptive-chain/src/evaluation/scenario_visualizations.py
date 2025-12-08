"""
Create visualizations for scenario testing results
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def plot_scenario_heatmap(save_path: str = 'data/scenario_heatmap.png'):
    """Create heatmap of all agents across all scenarios"""
    
    # Load results
    try:
        df = pd.read_csv('data/scenario_results.csv')
    except FileNotFoundError:
        print("❌ Scenario results not found")
        return
    
    # Pivot for heatmap
    pivot = df.pivot_table(
        index='scenario',
        columns='policy',
        values='mean_cost',
        aggfunc='mean'
    )
    
    # Normalize for better visualization (lower is better = green)
    pivot_normalized = (pivot - pivot.min().min()) / (pivot.max().max() - pivot.min().min())
    pivot_normalized = 1 - pivot_normalized  # Invert so green = better
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap
    im = ax.imshow(pivot_normalized.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([col.replace('_', ' ').title() for col in pivot.columns], rotation=45, ha='right')
    ax.set_yticklabels([idx.replace('_', ' ').title() for idx in pivot.index])
    
    # Add values
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            cost = pivot.iloc[i, j]
            text = ax.text(j, i, f'${cost/1e6:.1f}M',
                          ha="center", va="center", color="black", fontweight='bold', fontsize=9)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Performance (Green = Better)', fontsize=11, fontweight='bold')
    
    ax.set_title('Agent Performance Across Scenarios\n(All Costs in Millions)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path}")
    plt.close()


def plot_scenario_comparison_bars(save_path: str = 'data/scenario_comparison_bars.png'):
    """Create grouped bar chart for scenario comparison"""
    
    try:
        df = pd.read_csv('data/scenario_results.csv')
    except FileNotFoundError:
        print("❌ Scenario results not found")
        return
    
    # Get unique scenarios and policies
    scenarios = df['scenario'].unique()
    policies = df['policy'].unique()
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    colors = {
        'random': '#e74c3c',
        'reorder_point': '#27ae60',
        'eoq': '#3498db',
        'dqn': '#9b59b6',
        'independent_multi_agent': '#e67e22',
        'coordinated_multi_agent': '#c0392b'
    }
    
    # Plot each scenario
    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        
        scenario_data = df[df['scenario'] == scenario]
        
        policy_names = []
        costs = []
        policy_colors = []
        
        for _, row in scenario_data.iterrows():
            policy_names.append(row['policy'].replace('_', '\n'))
            costs.append(row['mean_cost'])
            policy_colors.append(colors.get(row['policy'], '#95a5a6'))
        
        bars = ax.bar(range(len(policy_names)), costs, color=policy_colors, 
                     alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax.set_xticks(range(len(policy_names)))
        ax.set_xticklabels(policy_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Cost ($)', fontsize=10, fontweight='bold')
        ax.set_title(scenario.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add values on bars
        for bar, cost in zip(bars, costs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${cost/1e6:.1f}M',
                   ha='center', va='bottom', fontsize=7, fontweight='bold')
        
        # Highlight winner
        min_cost_idx = costs.index(min(costs))
        bars[min_cost_idx].set_edgecolor('gold')
        bars[min_cost_idx].set_linewidth(3)
    
    # Remove extra subplot
    if len(scenarios) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.suptitle('Agent Performance Across All Scenarios', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path}")
    plt.close()


def plot_ablation_visualization(save_path: str = 'data/ablation_visualization.png'):
    """Visualize ablation study results"""
    
    try:
        with open('data/ablation_study_results.json', 'r') as f:
            ablation = json.load(f)
    except FileNotFoundError:
        print("❌ Ablation results not found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Cost comparison
    ax1 = axes[0]
    
    conditions = ['WITH\nTransfers', 'WITHOUT\nTransfers']
    costs = [
        ablation['with_transfers']['mean_cost'],
        ablation['without_transfers']['mean_cost']
    ]
    transfers = [
        ablation['with_transfers']['mean_transfers'],
        0
    ]
    
    bars = ax1.bar(conditions, [c/1e6 for c in costs], 
                   color=['#e74c3c', '#3498db'], alpha=0.7, edgecolor='black', linewidth=2)
    
    ax1.set_ylabel('Cost ($M)', fontsize=12, fontweight='bold')
    ax1.set_title('Ablation Study: Transfer Feature Impact', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add values
    for bar, cost, transfer_count in zip(bars, costs, transfers):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'${cost/1e6:.2f}M\n({int(transfer_count)} transfers)',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Annotate difference
    diff = ablation['transfer_impact']['cost_difference']
    pct = ablation['transfer_impact']['percent_difference']
    
    arrow_color = 'green' if diff < 0 else 'red'
    arrow_direction = '↓' if diff < 0 else '↑'
    
    ax1.annotate(f'{arrow_direction} ${abs(diff)/1e6:.2f}M\n({abs(pct):.1f}%)',
                xy=(0.5, max(costs)/1e6 * 0.6),
                ha='center', fontsize=12, fontweight='bold', color=arrow_color,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor=arrow_color, linewidth=2))
    
    # Plot 2: Transfer breakdown
    ax2 = axes[1]
    
    components = ['Base\nOperation', 'Transfer\nCost', 'Transfer\nOverhead']
    values = [
        costs[1] / 1e6,  # Without transfers (baseline)
        ablation['transfer_impact']['estimated_transfer_cost'] / 1e6,  # Direct cost
        max(0, (costs[0] - costs[1] - ablation['transfer_impact']['estimated_transfer_cost'])) / 1e6  # Overhead
    ]
    
    colors_breakdown = ['#3498db', '#f39c12', '#e74c3c']
    bars = ax2.bar(components, values, color=colors_breakdown, alpha=0.7, edgecolor='black', linewidth=2)
    
    ax2.set_ylabel('Cost ($M)', fontsize=12, fontweight='bold')
    ax2.set_title('Transfer Cost Breakdown', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, value in zip(bars, values):
        if value > 0:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'${value:.2f}M',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path}")
    plt.close()


def create_all_scenario_visualizations():
    """Generate all scenario-related visualizations"""
    
    print("=" * 70)
    print("CREATING SCENARIO VISUALIZATIONS")
    print("=" * 70)
    
    print(f"\n📊 Generating charts...")
    
    print(f"\n1. Scenario Performance Heatmap:")
    plot_scenario_heatmap()
    
    print(f"\n2. Scenario Comparison Bars:")
    plot_scenario_comparison_bars()
    
    print(f"\n3. Ablation Study Visualization:")
    plot_ablation_visualization()
    
    print(f"\n{'='*70}")
    print(f"✅ ALL SCENARIO VISUALIZATIONS CREATED")
    print(f"{'='*70}")
    print(f"\n📁 Created files:")
    print(f"  ✓ data/scenario_heatmap.png")
    print(f"  ✓ data/scenario_comparison_bars.png")
    print(f"  ✓ data/ablation_visualization.png")
    
    print(f"\n🎯 Phase 4 Complete!")
    print(f"\n✅ Ready for Phase 5: Streamlit Interactive UI")


if __name__ == "__main__":
    create_all_scenario_visualizations()