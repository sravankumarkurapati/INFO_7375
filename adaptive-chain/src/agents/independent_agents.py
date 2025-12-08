"""
Independent multi-agent training (no coordination baseline)
"""
import numpy as np
from typing import List, Dict
from stable_baselines3 import DQN
from src.environment.multi_warehouse_env import MultiWarehouseEnv
from src.environment.env_wrappers import FlattenMultiDiscreteWrapper


class IndependentMultiAgent:
    """
    Independent agents - each warehouse has separate DQN
    No coordination, no communication
    Baseline for comparison
    """
    
    def __init__(
        self,
        base_env: MultiWarehouseEnv,
        learning_rate: float = 0.0003,
        verbose: int = 0
    ):
        """
        Initialize independent agents
        
        Args:
            base_env: Base multi-warehouse environment
            learning_rate: Learning rate for each agent
            verbose: Verbosity level
        """
        self.num_warehouses = base_env.num_warehouses
        self.num_products = base_env.num_products
        self.action_quantities = base_env.action_quantities
        
        print(f"\n🤖 Creating {self.num_warehouses} Independent DQN Agents...")
        
        # We'll simulate independent agents by:
        # 1. Training one DQN on single-warehouse env
        # 2. Using same policy for all warehouses
        # This is the "no coordination" baseline
        
        from src.environment.supply_chain_env import SupplyChainEnv
        
        # Create single-warehouse environment
        single_env = SupplyChainEnv(
            num_products=self.num_products,
            episode_length=180,
            random_seed=42
        )
        single_env = FlattenMultiDiscreteWrapper(single_env)
        
        # Train one agent (represents independent policy)
        self.agent = DQN(
            policy="MlpPolicy",
            env=single_env,
            learning_rate=learning_rate,
            buffer_size=50000,
            batch_size=64,
            gamma=0.99,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=verbose
        )
        
        print(f"  ✓ Created independent agent template")
        print(f"  ✓ Each warehouse will use same policy (no coordination)")
    
    def train(self, total_timesteps: int = 100000):
        """Train the independent agent"""
        print(f"\n🚀 Training Independent Agent...")
        print(f"  Timesteps: {total_timesteps:,}")
        
        self.agent.learn(total_timesteps=total_timesteps, progress_bar=False)
        
        print(f"✅ Independent agent trained")
    
    def predict(self, full_observation: np.ndarray, deterministic: bool = True):
        """
        Predict actions for all warehouses using same policy
        
        Args:
            full_observation: Full state (57-dim for 3 warehouses)
            deterministic: Use deterministic policy
            
        Returns:
            Actions for all warehouses (9 actions: 3 warehouses × 3 products)
        """
        # Multi-warehouse state is 57 dims (19 per warehouse):
        # Per warehouse: inventory(3) + pending(3) + forecast(3) + days(3) + util(1) + others_inv(6) = 19
        # 
        # Single-warehouse agent expects 13 dims:
        # inventory(3) + pending(3) + forecast(3) + days(3) + util(1) = 13
        #
        # Extract first warehouse's core state (exclude others' inventory)
        
        single_wh_state = full_observation[:13]  # First 13 dims = first warehouse's core state
        
        # Predict action for single warehouse using trained policy
        # Returns flattened action (0-63 representing 3 product decisions)
        flat_action, _ = self.agent.predict(single_wh_state, deterministic=deterministic)
        
        # Unflatten to get individual product actions
        # flat_action is integer 0-63 representing 4^3 combinations
        # Convert to [prod1_idx, prod2_idx, prod3_idx]
        product_actions = []
        remaining = int(flat_action)
        num_action_choices = len(self.action_quantities)  # 4
        
        for i in range(self.num_products):
            product_actions.append(remaining % num_action_choices)
            remaining = remaining // num_action_choices
        
        # Reverse because we extracted in reverse order
        product_actions = np.array(product_actions[::-1], dtype=np.int32)
        
        # Repeat same product actions for all warehouses (independent = same policy)
        # Result shape: [wh1_p1, wh1_p2, wh1_p3, wh2_p1, wh2_p2, wh2_p3, wh3_p1, wh3_p2, wh3_p3]
        full_action = np.tile(product_actions, self.num_warehouses)
        
        return full_action, None
    
    def save(self, path: str):
        """Save agent"""
        self.agent.save(path)
        print(f"💾 Independent agent saved to: {path}")
    
    def load(self, path: str):
        """Load agent"""
        from src.environment.supply_chain_env import SupplyChainEnv
        single_env = SupplyChainEnv(num_products=self.num_products, episode_length=180)
        single_env = FlattenMultiDiscreteWrapper(single_env)
        
        self.agent = DQN.load(path, env=single_env)
        print(f"📂 Independent agent loaded from: {path}")