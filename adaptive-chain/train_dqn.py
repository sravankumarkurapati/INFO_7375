"""
Train OPTIMIZED DQN agent for supply chain optimization
"""
import os
import sys
from src.environment.supply_chain_env import SupplyChainEnv
from src.environment.env_wrappers import FlattenMultiDiscreteWrapper
from src.agents.dqn_agent import (
    create_dqn_agent,
    train_dqn_agent,
    evaluate_dqn_agent,
    save_dqn_model,
    TrainingCallback
)
from stable_baselines3.common.monitor import Monitor
import json


def main():
    """Main training function - OPTIMIZED VERSION"""
    
    print("=" * 70)
    print("DQN TRAINING - OPTIMIZED VERSION")
    print("=" * 70)
    
    # Configuration - OPTIMIZED
    TOTAL_TIMESTEPS = 200000  # DOUBLED from 100K
    EVAL_EPISODES = 10
    MODEL_PATH = "models/dqn_optimized.zip"  # Different name
    
    print(f"\n📋 OPTIMIZED Configuration:")
    print(f"  Total Timesteps: {TOTAL_TIMESTEPS:,} (2x original)")
    print(f"  Evaluation Episodes: {EVAL_EPISODES}")
    print(f"  Model Save Path: {MODEL_PATH}")
    print(f"\n🎯 Target: Beat Reorder Point ($1,061,199)")
    
    # Create training environment
    print(f"\n🏗️ Creating Training Environment...")
    base_train_env = SupplyChainEnv(
        num_products=3,
        num_warehouses=1,
        episode_length=180,
        random_seed=42,
        verbose=False
    )
    # Wrap to flatten action space for DQN
    train_env = FlattenMultiDiscreteWrapper(base_train_env)
    train_env = Monitor(train_env)
    
    print(f"  ✓ Environment created (OPTIMIZED)")
    
    # Create evaluation environment
    print(f"\n🏗️ Creating Evaluation Environment...")
    base_eval_env = SupplyChainEnv(
        num_products=3,
        num_warehouses=1,
        episode_length=180,
        random_seed=123,
        verbose=False
    )
    eval_env = FlattenMultiDiscreteWrapper(base_eval_env)
    print(f"  ✓ Environment created")
    
    # Create OPTIMIZED DQN agent
    print(f"\n" + "="*70)
    model = create_dqn_agent(
        env=train_env,
        learning_rate=0.0003,  # INCREASED
        buffer_size=100000,  # DOUBLED
        learning_starts=2000,  # INCREASED
        batch_size=128,  # DOUBLED
        gamma=0.99,
        tau=0.01,  # DOUBLED
        exploration_fraction=0.5,  # INCREASED
        exploration_final_eps=0.1,  # INCREASED
        use_prioritized_replay=True,  # NEW
        use_lr_schedule=True,  # NEW
        tensorboard_log="./logs/dqn_optimized/"
    )
    
    # Create callback
    callback = TrainingCallback(check_freq=1000, verbose=1)
    
    # Train agent
    print(f"\n" + "="*70)
    model = train_dqn_agent(
        model=model,
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        progress_bar=False  # Disable to avoid dependency issues
    )
    
    # Evaluate trained agent
    print(f"\n" + "="*70)
    results = evaluate_dqn_agent(
        model=model,
        env=eval_env,
        num_episodes=EVAL_EPISODES,
        deterministic=True,
        verbose=True
    )
    
    # Save model and results
    print(f"\n" + "="*70)
    save_dqn_model(model, MODEL_PATH, results)
    
    # Compare with baselines
    print(f"\n" + "="*70)
    print("COMPARISON WITH BASELINES")
    print(f"="*70)
    
    # Load baseline results
    try:
        with open('data/baseline_results.json', 'r') as f:
            baseline_results = json.load(f)
        
        print(f"\n{'Policy':<35} {'Mean Cost':<15} {'vs Random':<20}")
        print("-" * 70)
        
        random_cost = baseline_results['random']['mean_cost']
        
        # Random
        print(f"{'Random Policy':<35} ${random_cost:>13,.0f} {'(baseline)':<20}")
        
        # EOQ
        eoq_cost = baseline_results['eoq']['mean_cost']
        eoq_improvement = (random_cost - eoq_cost) / random_cost * 100
        print(f"{'EOQ Policy':<35} ${eoq_cost:>13,.0f} {f'-{eoq_improvement:.1f}%':<20}")
        
        # Reorder Point
        reorder_cost = baseline_results['reorder_point']['mean_cost']
        reorder_improvement = (random_cost - reorder_cost) / random_cost * 100
        print(f"{'Reorder Point Policy':<35} ${reorder_cost:>13,.0f} {f'-{reorder_improvement:.1f}%':<20}")
        
        # DQN Optimized
        dqn_cost = results['mean_cost']
        dqn_improvement = (random_cost - dqn_cost) / random_cost * 100
        print(f"{'DQN Agent (OPTIMIZED)':<35} ${dqn_cost:>13,.0f} {f'-{dqn_improvement:.1f}%':<20}")
        
        # Improvement over best baseline
        best_baseline = min(eoq_cost, reorder_cost)
        improvement_vs_best = (best_baseline - dqn_cost) / best_baseline * 100
        
        print(f"\n{'='*70}")
        print(f"🎯 DQN IMPROVEMENT OVER BEST BASELINE:")
        print(f"  Best Baseline: ${best_baseline:,.2f} (Reorder Point)")
        print(f"  DQN Agent (OPTIMIZED): ${dqn_cost:,.2f}")
        
        if improvement_vs_best > 0:
            print(f"  ✅ SUCCESS! {improvement_vs_best:.1f}% BETTER than best baseline!")
            print(f"  🎉 DQN has learned superior inventory management!")
        elif improvement_vs_best > -5:
            print(f"  ✅ COMPETITIVE! Within {abs(improvement_vs_best):.1f}% of best baseline")
            print(f"  💡 Excellent performance given classical OR strength")
        else:
            print(f"  ⚠️ Performance: {abs(improvement_vs_best):.1f}% behind best baseline")
            print(f"  💡 Consider: More training steps or hyperparameter tuning")
        
        # Calculate total improvement
        print(f"\n📊 Overall Performance:")
        print(f"  Total Cost Reduction: ${random_cost - dqn_cost:,.0f}")
        print(f"  Percentage Improvement: {dqn_improvement:.1f}%")
        
    except FileNotFoundError:
        print("\n⚠️ Baseline results not found. Run evaluate_baselines.py first.")
    
    print(f"\n{'='*70}")
    print("✅ OPTIMIZED TRAINING COMPLETE")
    print(f"{'='*70}")
    print("\n📁 Next steps:")
    print("  1. Check learning curves: tensorboard --logdir=./logs/dqn_optimized/")
    print("  2. View results: cat models/dqn_optimized_results.json")
    print("  3. Compare: Old DQN vs Optimized DQN")
    print("  4. Continue to Phase 3: Multi-agent RL")


if __name__ == "__main__":
    main()