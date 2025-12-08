"""
Multi-agent training with checkpoint resume capability
FULL TRAINING with crash resistance
"""
import numpy as np
import os
import glob
from src.environment.multi_warehouse_env import MultiWarehouseEnv
from src.environment.env_wrappers import FlattenMultiDiscreteWrapper
from src.agents.independent_agents import IndependentMultiAgent
from src.agents.coordinated_agents import CoordinatedMultiAgent
from src.agents.dqn_agent import TrainingCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3 import DQN
import json


def find_latest_checkpoint(checkpoint_dir: str, prefix: str) -> tuple:
    """
    Find the latest checkpoint file
    
    Returns:
        (checkpoint_path, steps_completed) or (None, 0)
    """
    if not os.path.exists(checkpoint_dir):
        return None, 0
    
    pattern = os.path.join(checkpoint_dir, f"{prefix}_*_steps.zip")
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        return None, 0
    
    # Extract step numbers and find latest
    checkpoint_steps = []
    for ckpt in checkpoints:
        try:
            # Extract number from "coordinated_agent_50000_steps.zip"
            parts = os.path.basename(ckpt).split('_')
            steps = int(parts[-2])
            checkpoint_steps.append((ckpt, steps))
        except (ValueError, IndexError):
            continue
    
    if not checkpoint_steps:
        return None, 0
    
    # Return checkpoint with most steps
    latest = max(checkpoint_steps, key=lambda x: x[1])
    return latest[0], latest[1]


def evaluate_multi_agent(agent, env, num_episodes: int = 10, agent_name: str = "Agent"):
    """Evaluate multi-agent system"""
    print(f"\n{'='*70}")
    print(f"Evaluating {agent_name}")
    print(f"{'='*70}")
    
    episode_costs = []
    episode_rewards = []
    episode_summaries = []
    
    for ep in range(num_episodes):
        obs, info = env.reset(seed=42 + ep)
        done = False
        ep_reward = 0
        
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        
        summary = env.get_episode_summary()
        episode_costs.append(summary['total_cost'])
        episode_rewards.append(ep_reward)
        episode_summaries.append(summary)
        
        print(f"  Episode {ep+1}/{num_episodes}: Cost=${summary['total_cost']:,.0f}, "
              f"Transfers={summary.get('transfer_count', 0)}")
    
    results = {
        'agent_name': agent_name,
        'mean_cost': np.mean(episode_costs),
        'std_cost': np.std(episode_costs),
        'mean_reward': np.mean(episode_rewards),
        'episode_costs': episode_costs,
        'summaries': episode_summaries
    }
    
    print(f"\n📈 Results for {agent_name}:")
    print(f"  Mean Cost: ${results['mean_cost']:,.2f} ± ${results['std_cost']:,.2f}")
    if episode_summaries and 'transfer_count' in episode_summaries[0]:
        avg_transfers = np.mean([s.get('transfer_count', 0) for s in episode_summaries])
        print(f"  Avg Transfers: {avg_transfers:.1f}")
    
    return results


def main():
    """Main training with full checkpoint resume capability"""
    
    print("=" * 70)
    print("MULTI-AGENT RL TRAINING - RESUMABLE VERSION")
    print("=" * 70)
    
    # Configuration - FULL TRAINING
    TRAIN_TIMESTEPS_INDEPENDENT = 100000
    TRAIN_TIMESTEPS_COORDINATED = 150000  # FULL 150K
    CHECKPOINT_FREQ = 10000  # Save every 10K steps
    EVAL_EPISODES = 10
    
    CHECKPOINT_DIR = './models/checkpoints/'
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    print(f"\n📋 Configuration:")
    print(f"  Independent Training: {TRAIN_TIMESTEPS_INDEPENDENT:,} steps (~555 episodes)")
    print(f"  Coordinated Training: {TRAIN_TIMESTEPS_COORDINATED:,} steps (~833 episodes)")
    print(f"  Checkpoint Frequency: Every {CHECKPOINT_FREQ:,} steps")
    print(f"  Evaluation Episodes: {EVAL_EPISODES}")
    print(f"  💾 Checkpoints: {CHECKPOINT_DIR}")
    
    # Create environments
    print(f"\n🏗️ Creating Multi-Warehouse Environments...")
    
    # For independent training
    env_independent = MultiWarehouseEnv(
        num_products=3,
        num_warehouses=3,
        episode_length=180,
        enable_transfers=False,
        random_seed=42
    )
    
    # For coordinated training
    env_coordinated = MultiWarehouseEnv(
        num_products=3,
        num_warehouses=3,
        episode_length=180,
        enable_transfers=True,
        random_seed=42
    )
    env_coordinated_wrapped = FlattenMultiDiscreteWrapper(env_coordinated)
    env_coordinated_wrapped = Monitor(env_coordinated_wrapped)
    
    # For evaluation
    env_eval = MultiWarehouseEnv(
        num_products=3,
        num_warehouses=3,
        episode_length=180,
        enable_transfers=True,
        random_seed=123
    )
    
    print(f"  ✓ Environments created")
    
    # ========================================
    # TRAIN INDEPENDENT AGENTS
    # ========================================
    print(f"\n{'='*70}")
    print(f"PHASE 1: Training Independent Agents")
    print(f"{'='*70}")
    
    independent_model_path = 'models/independent_multi_agent.zip'
    
    # Check if already trained
    if os.path.exists(independent_model_path):
        print(f"\n✅ Found existing independent agent: {independent_model_path}")
        print(f"  Skipping training (already complete)")
        
        independent_agent = IndependentMultiAgent(
            base_env=env_independent,
            learning_rate=0.0003,
            verbose=0
        )
        
        # Create a single-warehouse env for loading
        from src.environment.supply_chain_env import SupplyChainEnv
        temp_env = SupplyChainEnv(num_products=3, episode_length=180, random_seed=42)
        temp_env = FlattenMultiDiscreteWrapper(temp_env)
        
        independent_agent.agent = DQN.load(
            independent_model_path,
            env=temp_env
        )
        print(f"  📂 Loaded existing model")
        
    else:
        print(f"\n🚀 Training Independent Agent from scratch...")
        independent_agent = IndependentMultiAgent(
            base_env=env_independent,
            learning_rate=0.0003,
            verbose=0
        )
        
        independent_agent.train(total_timesteps=TRAIN_TIMESTEPS_INDEPENDENT)
        independent_agent.save(independent_model_path)
    
    # ========================================
    # TRAIN COORDINATED AGENTS WITH RESUME
    # ========================================
    print(f"\n{'='*70}")
    print(f"PHASE 2: Training Coordinated Agents (WITH Checkpoints)")
    print(f"{'='*70}")
    
    coordinated_model_path = 'models/coordinated_multi_agent.zip'
    
    # Check for existing final model
    if os.path.exists(coordinated_model_path):
        print(f"\n✅ Found existing coordinated agent: {coordinated_model_path}")
        user_input = input(f"  Retrain from scratch? (y/n): ").lower()
        
        if user_input != 'y':
            print(f"  📂 Using existing model")
            coordinated_agent = CoordinatedMultiAgent(
                env=env_coordinated,
                learning_rate=0.0003,
                buffer_size=100000,
                verbose=0
            )
            coordinated_agent.agent = DQN.load(
                coordinated_model_path,
                env=env_coordinated_wrapped
            )
            
            # Skip to evaluation
            print(f"\n  ⏭️ Skipping to evaluation...")
            results_independent = evaluate_multi_agent(
                independent_agent, env_eval, EVAL_EPISODES,
                "Independent Agents (No Coordination)"
            )
            results_coordinated = evaluate_multi_agent(
                coordinated_agent, env_eval, EVAL_EPISODES,
                "Coordinated Agents (WITH Coordination)"
            )
            
            # Jump to results
            display_results(results_independent, results_coordinated)
            return
    
    # Check for checkpoint to resume from
    checkpoint_path, steps_completed = find_latest_checkpoint(
        CHECKPOINT_DIR, 
        'coordinated_agent'
    )
    
    if checkpoint_path:
        print(f"\n🔄 RESUMING from checkpoint!")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Steps completed: {steps_completed:,}")
        print(f"  Remaining: {TRAIN_TIMESTEPS_COORDINATED - steps_completed:,} steps")
        
        # Load from checkpoint
        coordinated_agent = CoordinatedMultiAgent(
            env=env_coordinated,
            learning_rate=0.0003,
            buffer_size=100000,
            verbose=0
        )
        coordinated_agent.agent = DQN.load(checkpoint_path, env=env_coordinated_wrapped)
        
        # Train for remaining steps
        remaining_steps = TRAIN_TIMESTEPS_COORDINATED - steps_completed
        
    else:
        print(f"\n🚀 Training Coordinated Agent from scratch...")
        print(f"  No checkpoint found - starting fresh")
        
        coordinated_agent = CoordinatedMultiAgent(
            env=env_coordinated,
            learning_rate=0.0003,
            buffer_size=100000,
            verbose=0
        )
        
        remaining_steps = TRAIN_TIMESTEPS_COORDINATED
    
    # Create callbacks
    training_callback = TrainingCallback(check_freq=1000, verbose=1)
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix='coordinated_agent',
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=1
    )
    combined_callback = CallbackList([training_callback, checkpoint_callback])
    
    print(f"\n  💾 Checkpoints will save to: {CHECKPOINT_DIR}")
    print(f"  📊 Progress updates every 10 episodes")
    print(f"  ⏱️ Estimated time: {remaining_steps / 700 / 60:.1f} minutes")
    
    # Train
    coordinated_agent.train(
        total_timesteps=remaining_steps,
        callback=combined_callback
    )
    
    # Save final model
    coordinated_agent.save(coordinated_model_path)
    print(f"\n💾 Final model saved to: {coordinated_model_path}")
    
    # ========================================
    # EVALUATION & COMPARISON
    # ========================================
    print(f"\n{'='*70}")
    print(f"EVALUATION: Comparing All Approaches")
    print(f"{'='*70}")
    
    # Evaluate independent
    results_independent = evaluate_multi_agent(
        independent_agent,
        env_eval,
        num_episodes=EVAL_EPISODES,
        agent_name="Independent Agents (No Coordination)"
    )
    
    # Evaluate coordinated
    results_coordinated = evaluate_multi_agent(
        coordinated_agent,
        env_eval,
        num_episodes=EVAL_EPISODES,
        agent_name="Coordinated Agents (WITH Coordination)"
    )
    
    # Display results
    display_results(results_independent, results_coordinated)
    
    return {
        'independent': results_independent,
        'coordinated': results_coordinated
    }


def display_results(results_independent, results_coordinated):
    """Display comparison results"""
    
    print(f"\n{'='*70}")
    print(f"MULTI-AGENT COMPARISON RESULTS")
    print(f"{'='*70}")
    
    independent_cost = results_independent['mean_cost']
    coordinated_cost = results_coordinated['mean_cost']
    
    improvement = (independent_cost - coordinated_cost) / independent_cost * 100
    
    print(f"\n📊 System-Wide Performance (3 Warehouses):")
    print(f"  Independent Agents: ${independent_cost:,.2f}")
    print(f"  Coordinated Agents: ${coordinated_cost:,.2f}")
    
    if improvement > 0:
        print(f"\n  ✅ COORDINATION BENEFIT: {improvement:.1f}% improvement!")
        print(f"  💰 Cost Savings: ${independent_cost - coordinated_cost:,.2f}")
        print(f"  🎯 Demonstrates RL's coordination advantage!")
    else:
        print(f"\n  ⚠️ Coordination: {abs(improvement):.1f}% difference")
        print(f"  💡 Results show independent policies may be sufficient")
    
    # Compare with baselines
    print(f"\n📊 Complete System Comparison:")
    try:
        with open('data/baseline_results.json', 'r') as f:
            baseline_results = json.load(f)
            
        random_cost = baseline_results['random']['mean_cost']
        reorder_cost = baseline_results['reorder_point']['mean_cost']
        
        # Scale to 3 warehouses
        random_3wh = random_cost * 3
        reorder_3wh = reorder_cost * 3
        
        print(f"\n  {'Policy':<40} {'Cost':<20} {'vs Random':<15}")
        print(f"  {'-'*70}")
        print(f"  {'Random (3 warehouses)':<40} ${random_3wh:>18,.0f} {'baseline':<15}")
        print(f"  {'Reorder Point (3 warehouses)':<40} ${reorder_3wh:>18,.0f} {f'{(random_3wh-reorder_3wh)/random_3wh*100:.1f}%':<15}")
        print(f"  {'Independent DQN Agents':<40} ${independent_cost:>18,.0f} {f'{(random_3wh-independent_cost)/random_3wh*100:.1f}%':<15}")
        print(f"  {'Coordinated DQN Agents':<40} ${coordinated_cost:>18,.0f} {f'{(random_3wh-coordinated_cost)/random_3wh*100:.1f}%':<15}")
        
        coord_vs_reorder = (reorder_3wh - coordinated_cost) / reorder_3wh * 100
        
        print(f"\n  🎯 Key Insights:")
        print(f"    • Coordination vs Random: {(random_3wh-coordinated_cost)/random_3wh*100:.1f}% better")
        print(f"    • Coordination vs Reorder Point: {coord_vs_reorder:.1f}% {'better' if coord_vs_reorder > 0 else 'worse'}")
        print(f"    • Coordination Benefit: {improvement:.1f}% over independent RL")
        
    except FileNotFoundError:
        print(f"  ⚠️ Baseline results not found")
    
    # Save results
    all_results = {
        'independent': {
            'mean_cost': float(results_independent['mean_cost']),
            'std_cost': float(results_independent['std_cost']),
            'episode_costs': [float(c) for c in results_independent['episode_costs']]
        },
        'coordinated': {
            'mean_cost': float(results_coordinated['mean_cost']),
            'std_cost': float(results_coordinated['std_cost']),
            'episode_costs': [float(c) for c in results_coordinated['episode_costs']]
        },
        'coordination_improvement': float(improvement)
    }
    
    os.makedirs('data', exist_ok=True)
    with open('data/multi_agent_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Results saved to: data/multi_agent_results.json")
    
    print(f"\n{'='*70}")
    print(f"✅ MULTI-AGENT TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"\n📁 Saved Files:")
    print(f"  ✓ models/independent_multi_agent.zip")
    print(f"  ✓ models/coordinated_multi_agent.zip")
    print(f"  ✓ models/checkpoints/ (15 checkpoints)")
    print(f"  ✓ data/multi_agent_results.json")
    
    print(f"\n🎯 Next Steps:")
    print(f"  → Step 3.3: Create visualizations")
    print(f"  → Step 3.4: Statistical analysis")
    print(f"  → Phase 4: Comprehensive evaluation")


if __name__ == "__main__":
    results = main()