"""
Visualization utilities for supply chain data
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_demand_patterns(demand_df: pd.DataFrame, title: str = "Demand Patterns"):
    """
    Plot demand over time for all products
    
    Args:
        demand_df: DataFrame with columns [day, sku, demand]
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for sku in demand_df['sku'].unique():
        sku_data = demand_df[demand_df['sku'] == sku]
        ax.plot(sku_data['day'], sku_data['demand'], 
               label=sku, marker='o', markersize=2, alpha=0.7)
    
    ax.set_xlabel('Day', fontsize=12)
    ax.set_ylabel('Demand (units)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_demand_statistics(demand_df: pd.DataFrame):
    """
    Plot demand statistics by product
    
    Args:
        demand_df: DataFrame with columns [day, sku, demand]
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Box plot
    demand_df.boxplot(column='demand', by='sku', ax=axes[0])
    axes[0].set_title('Demand Distribution by Product')
    axes[0].set_xlabel('Product SKU')
    axes[0].set_ylabel('Demand (units)')
    plt.sca(axes[0])
    plt.xticks(rotation=0)
    
    # Statistics table
    stats = demand_df.groupby('sku')['demand'].agg([
        ('Mean', 'mean'),
        ('Std', 'std'),
        ('Min', 'min'),
        ('Max', 'max')
    ]).round(2)
    
    axes[1].axis('tight')
    axes[1].axis('off')
    table = axes[1].table(cellText=stats.values,
                         rowLabels=stats.index,
                         colLabels=stats.columns,
                         cellLoc='center',
                         loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    axes[1].set_title('Demand Statistics', fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig


def visualize_disruptions(disruptions: List[Dict], num_days: int):
    """
    Visualize disruption timeline
    
    Args:
        disruptions: List of disruption events
        num_days: Total simulation days
    """
    if not disruptions:
        print("No disruptions to visualize")
        return None
    
    fig, ax = plt.subplots(figsize=(14, 4))
    
    colors = {
        'supplier_delay': 'red',
        'demand_spike': 'orange',
        'capacity_reduction': 'purple',
        'transportation_issue': 'blue'
    }
    
    for i, disruption in enumerate(disruptions):
        start = disruption['start_day']
        duration = disruption['end_day'] - disruption['start_day']
        dtype = disruption['type']
        
        ax.barh(i, duration, left=start, height=0.8,
               color=colors.get(dtype, 'gray'),
               alpha=0.7,
               label=dtype if dtype not in [d.get_label() for d in ax.containers] else "")
        
        # Add text
        ax.text(start + duration/2, i, disruption['type'].replace('_', ' ').title(),
               ha='center', va='center', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Day', fontsize=12)
    ax.set_ylabel('Disruption Event', fontsize=12)
    ax.set_title('Disruption Timeline', fontsize=14, fontweight='bold')
    ax.set_xlim(0, num_days)
    ax.set_ylim(-0.5, len(disruptions) - 0.5)
    ax.invert_yaxis()
    
    # Remove duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.tight_layout()
    return fig