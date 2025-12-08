"""
Scenario testing for supply chain RL agents
Tests all agents across multiple disruption scenarios
"""
import numpy as np
import pandas as pd
from typing import Dict, List
import json
import os

from src.environment.supply_chain_env import SupplyChainEnv
from src.environment.multi_warehouse_env import MultiWarehouseEnv
from src.environment.env_wrappers import FlattenMultiDiscreteWrapper
from src.agents.baseline_policies import create_baseline_policies
from src.agents.dqn_agent import load_dqn_model
from src.agents.independent_agents import IndependentMultiAgent
from src.agents.coordinated_agents import CoordinatedMultiAgent
from stable_baselines3 import DQN


class ScenarioTester:
    """
    Test agents across different scenarios
    
    Scenarios:
    1. Normal Operations - Standard demand
    2. High Demand Surge - 1.5x demand
    3. Supplier Crisis - Lead times doubled
    4. Demand Shock - Sudden 3x spike
    5. Capacity Constraints - 50% capacity reduction
    """
    
    def __init__(self):
        """Initialize scenario tester"""
        self.scenarios = self._create_scenarios()
        
    def _create_scenarios(self) -> List[Dict]:
        """Create test scenario configurations"""
        scenarios = [
            {
                'name': 'normal_operations',
                'description': 'Standard operating conditions',
                'demand_multiplier': 1.0,
                'lead_time_multiplier': 1.0,
                'capacity_multiplier': 1.0,
                'disruption_day': None
            },
            {
                'name': 'high_demand',
                'description': 'Sustained high demand period',
                'demand_multiplier': 1.5,
                'lead_time_multiplier': 1.0,
                'capacity_multiplier': 1.0,
                'disruption_day': None
            },
            {
                'name': 'supplier_crisis',
                'description': 'Major supplier delays',
                'demand_multiplier': 1.0,
                'lead_time_multiplier': 2.0,
                'capacity_multiplier': 1.0,
                'disruption_day': 30  # Starts day 30
            },
            {
                'name': 'demand_shock',
                'description': 'Sudden demand spike',
                'demand_multiplier': 3.0,
                'lead_time_multiplier': 1.0,
                'capacity_multiplier': 1.0,
                'disruption_day': 45  # Day 45-60
            },
            {
                'name': 'capacity_crisis',
                'description': 'Warehouse capacity constraints',
                'demand_multiplier': 1.0,
                'lead_time_multiplier': 1.0,
                'capacity_multiplier': 0.5,
                'disruption_day': 20  # Starts day 20
            }
        ]
        
        return scenarios
    
    def test_baseline_on_scenario(
        self, 
        policy_name: str,
        scenario: Dict,
        num_episodes: int = 5
    ) -> Dict:
        """
        Test a baseline policy on a scenario
        
        Args:
            policy_name: 'random', 'reorder_point', or 'eoq'
            scenario: Scenario configuration
            num_episodes: Number of test episodes
            
        Returns:
            Results dictionary
        """
        episode_costs = []
        episode_summaries = []
        
        for ep in range(num_episodes):
            # Create environment for this scenario
            env = SupplyChainEnv(
                num_products=3,
                episode_length=180,
                random_seed=100 + ep
            )
            
            # Apply scenario modifications
            env = self._apply_scenario_modifications(env, scenario)
            
            # Get policy
            policies = create_baseline_policies(env)
            policy = policies[policy_name]
            
            # Run episode
            obs, info = env.reset()
            policy.reset()
            done = False
            
            while not done:
                action = policy.select_action(obs, info)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            
            summary = env.get_episode_summary()
            episode_costs.append(summary['total_cost'])
            episode_summaries.append(summary)
        
        results = {
            'policy': policy_name,
            'scenario': scenario['name'],
            'mean_cost': np.mean(episode_costs),
            'std_cost': np.std(episode_costs),
            'episode_costs': episode_costs
        }
        
        return results
    
    def test_dqn_on_scenario(
        self,
        model_path: str,
        scenario: Dict,
        num_episodes: int = 5
    ) -> Dict:
        """Test DQN agent on scenario"""
        episode_costs = []
        
        for ep in range(num_episodes):
            env = SupplyChainEnv(
                num_products=3,
                episode_length=180,
                random_seed=100 + ep
            )
            env = self._apply_scenario_modifications(env, scenario)
            env = FlattenMultiDiscreteWrapper(env)
            
            # Load model
            model = DQN.load(model_path, env=env)
            
            # Run episode
            obs, info = env.reset()
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            
            summary = env.unwrapped.get_episode_summary()
            episode_costs.append(summary['total_cost'])
        
        results = {
            'policy': 'dqn',
            'scenario': scenario['name'],
            'mean_cost': np.mean(episode_costs),
            'std_cost': np.std(episode_costs),
            'episode_costs': episode_costs
        }
        
        return results
    
    def test_multiagent_on_scenario(
        self,
        agent,
        agent_name: str,
        scenario: Dict,
        num_episodes: int = 5
    ) -> Dict:
        """Test multi-agent on scenario"""
        episode_costs = []
        
        for ep in range(num_episodes):
            env = MultiWarehouseEnv(
                num_products=3,
                num_warehouses=3,
                episode_length=180,
                enable_transfers=True,
                random_seed=100 + ep
            )
            env = self._apply_scenario_modifications(env, scenario)
            
            # Run episode
            obs, info = env.reset()
            done = False
            
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            
            summary = env.get_episode_summary()
            episode_costs.append(summary['total_cost'])
        
        results = {
            'policy': agent_name,
            'scenario': scenario['name'],
            'mean_cost': np.mean(episode_costs),
            'std_cost': np.std(episode_costs),
            'episode_costs': episode_costs
        }
        
        return results
    
    def _apply_scenario_modifications(self, env, scenario: Dict):
        """Apply scenario-specific modifications to environment"""
        # Modify demand
        if scenario['demand_multiplier'] != 1.0:
            env.demand_data['demand'] *= scenario['demand_multiplier']
        
        # Modify lead times
        if scenario['lead_time_multiplier'] != 1.0:
            for product in env.products:
                product.lead_time = int(product.lead_time * scenario['lead_time_multiplier'])
        
        # Modify capacity
        if scenario['capacity_multiplier'] != 1.0:
            for warehouse in env.warehouses:
                warehouse.capacity = int(warehouse.capacity * scenario['capacity_multiplier'])
        
        return env
    
    def run_all_scenarios(self) -> pd.DataFrame:
        """Run all agents on all scenarios"""
        
        print("=" * 70)
        print("SCENARIO TESTING - ALL AGENTS")
        print("=" * 70)
        
        all_results = []
        
        # Test each scenario
        for scenario in self.scenarios:
            print(f"\n{'='*70}")
            print(f"SCENARIO: {scenario['name'].upper().replace('_', ' ')}")
            print(f"Description: {scenario['description']}")
            print(f"{'='*70}")
            
            # Test baseline: Random
            print(f"\n  Testing: Random Policy...")
            result = self.test_baseline_on_scenario('random', scenario, num_episodes=5)
            all_results.append(result)
            print(f"    Cost: ${result['mean_cost']:,.0f} ± ${result['std_cost']:,.0f}")
            
            # Test baseline: Reorder Point
            print(f"\n  Testing: Reorder Point Policy...")
            result = self.test_baseline_on_scenario('reorder_point', scenario, num_episodes=5)
            all_results.append(result)
            print(f"    Cost: ${result['mean_cost']:,.0f} ± ${result['std_cost']:,.0f}")
            
            # Test DQN
            if os.path.exists('models/dqn_optimized.zip'):
                print(f"\n  Testing: DQN Agent...")
                result = self.test_dqn_on_scenario('models/dqn_optimized.zip', scenario, num_episodes=5)
                all_results.append(result)
                print(f"    Cost: ${result['mean_cost']:,.0f} ± ${result['std_cost']:,.0f}")
            
            # Test Independent Multi-Agent
            if os.path.exists('models/independent_multi_agent.zip'):
                print(f"\n  Testing: Independent Multi-Agent...")
                
                # Load independent agent
                from src.environment.supply_chain_env import SupplyChainEnv
                temp_env = MultiWarehouseEnv(num_products=3, num_warehouses=3, episode_length=180)
                independent_agent = IndependentMultiAgent(temp_env)
                
                single_env = SupplyChainEnv(num_products=3, episode_length=180)
                single_env = FlattenMultiDiscreteWrapper(single_env)
                independent_agent.agent = DQN.load('models/independent_multi_agent.zip', env=single_env)
                
                result = self.test_multiagent_on_scenario(
                    independent_agent, 'independent_multi_agent', scenario, num_episodes=5
                )
                all_results.append(result)
                print(f"    Cost: ${result['mean_cost']:,.0f} ± ${result['std_cost']:,.0f}")
            
            # Test Coordinated Multi-Agent
            if os.path.exists('models/coordinated_multi_agent.zip'):
                print(f"\n  Testing: Coordinated Multi-Agent...")
                
                # Load coordinated agent
                temp_env = MultiWarehouseEnv(num_products=3, num_warehouses=3, 
                                            episode_length=180, enable_transfers=True)
                coordinated_agent = CoordinatedMultiAgent(temp_env)
                
                wrapped = FlattenMultiDiscreteWrapper(temp_env)
                coordinated_agent.agent = DQN.load('models/coordinated_multi_agent.zip', env=wrapped)
                
                result = self.test_multiagent_on_scenario(
                    coordinated_agent, 'coordinated_multi_agent', scenario, num_episodes=5
                )
                all_results.append(result)
                print(f"    Cost: ${result['mean_cost']:,.0f} ± ${result['std_cost']:,.0f}")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_results)
        
        return df


def create_scenario_comparison_table(df: pd.DataFrame, save_path: str = 'data/scenario_results.csv'):
    """Create comparison table across scenarios"""
    
    # Pivot table
    pivot = df.pivot_table(
        index='scenario',
        columns='policy',
        values='mean_cost',
        aggfunc='mean'
    )
    
    # Format
    pivot = pivot.applymap(lambda x: f"${x:,.0f}")
    
    print(f"\n📊 SCENARIO COMPARISON TABLE:")
    print(pivot.to_string())
    
    # Save
    df.to_csv(save_path, index=False)
    print(f"\n💾 Saved detailed results to: {save_path}")
    
    # Save pivot
    pivot.to_csv(save_path.replace('.csv', '_pivot.csv'))
    print(f"💾 Saved pivot table to: {save_path.replace('.csv', '_pivot.csv')}")
    
    return pivot


def identify_best_agent_per_scenario(df: pd.DataFrame):
    """Identify which agent performs best in each scenario"""
    
    print(f"\n{'='*70}")
    print(f"BEST AGENT PER SCENARIO")
    print(f"{'='*70}")
    
    for scenario in df['scenario'].unique():
        scenario_data = df[df['scenario'] == scenario]
        best_idx = scenario_data['mean_cost'].idxmin()
        best_row = scenario_data.loc[best_idx]
        
        print(f"\n📍 {scenario.replace('_', ' ').title()}:")
        print(f"  🏆 Winner: {best_row['policy'].replace('_', ' ').title()}")
        print(f"  💰 Cost: ${best_row['mean_cost']:,.0f}")
        
        # Show all results for this scenario
        print(f"\n  All Agents:")
        for _, row in scenario_data.iterrows():
            symbol = "🏆" if row['policy'] == best_row['policy'] else "  "
            print(f"    {symbol} {row['policy']:25s}: ${row['mean_cost']:>12,.0f}")


def main():
    """Main scenario testing function"""
    
    print("=" * 70)
    print("COMPREHENSIVE SCENARIO TESTING")
    print("=" * 70)
    print("\n🎯 Testing all agents across 5 realistic scenarios")
    print("   This will take approximately 10-15 minutes...")
    
    # Create tester
    tester = ScenarioTester()
    
    # Run all tests
    print(f"\n🚀 Starting scenario testing...")
    results_df = tester.run_all_scenarios()
    
    # Create comparison table
    print(f"\n{'='*70}")
    pivot = create_scenario_comparison_table(results_df)
    
    # Identify winners
    identify_best_agent_per_scenario(results_df)
    
    # Save complete results
    os.makedirs('data', exist_ok=True)
    
    results_dict = {
        'scenarios': tester.scenarios,
        'results': results_df.to_dict('records')
    }
    
    with open('data/scenario_testing_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print(f"✅ SCENARIO TESTING COMPLETE")
    print(f"{'='*70}")
    print(f"\n📁 Results saved:")
    print(f"  ✓ data/scenario_results.csv (detailed)")
    print(f"  ✓ data/scenario_results_pivot.csv (comparison table)")
    print(f"  ✓ data/scenario_testing_results.json (complete)")
    
    print(f"\n🎯 Next: Step 4.2 - Ablation Study")
    
    return results_df


if __name__ == "__main__":
    results = main()