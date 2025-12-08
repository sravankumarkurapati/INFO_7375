"""
Baseline policies for comparison with RL agents
"""
import numpy as np
from typing import Dict, List, Tuple
import math


class BaselinePolicy:
    """Base class for baseline policies"""
    
    def __init__(self, env):
        """
        Initialize policy
        
        Args:
            env: Supply chain environment
        """
        self.env = env
        self.products = env.products
        self.warehouses = env.warehouses
        
    def select_action(self, observation: np.ndarray, info: dict) -> np.ndarray:
        """
        Select action based on policy
        
        Args:
            observation: Current state
            info: Additional information
            
        Returns:
            Action array
        """
        raise NotImplementedError
    
    def reset(self):
        """Reset policy state"""
        pass


class RandomPolicy(BaselinePolicy):
    """
    Random policy - selects actions uniformly at random
    This is our worst-case baseline
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.name = "Random Policy"
        
    def select_action(self, observation: np.ndarray, info: dict) -> np.ndarray:
        """Select random action"""
        return self.env.action_space.sample()


class ReorderPointPolicy(BaselinePolicy):
    """
    (s, Q) Reorder Point Policy
    - s = reorder point (when to order)
    - Q = order quantity (how much to order)
    
    Classic inventory management policy:
    - When inventory drops below s, order Q units
    - s typically set to cover demand during lead time + safety stock
    - Q typically set using EOQ formula or fixed quantity
    """
    
    def __init__(self, env, service_level: float = 0.95):
        """
        Initialize reorder point policy
        
        Args:
            env: Supply chain environment
            service_level: Desired service level (default 95%)
        """
        super().__init__(env)
        self.name = "Reorder Point (s,Q) Policy"
        self.service_level = service_level
        
        # Calculate reorder points and order quantities for each product
        self.reorder_points = {}
        self.order_quantities = {}
        
        for product in self.products:
            # Reorder point = (average demand × lead time) + safety stock
            avg_demand_during_lead_time = product.base_demand * product.lead_time
            
            # Safety stock using z-score for service level
            z_score = self._get_z_score(service_level)
            safety_stock = z_score * product.demand_std * math.sqrt(product.lead_time)
            
            self.reorder_points[product.sku] = avg_demand_during_lead_time + safety_stock
            
            # Order quantity using simplified EOQ
            # Q* = sqrt(2 * D * S / H)
            # D = annual demand, S = order cost, H = holding cost per unit per year
            annual_demand = product.base_demand * 365
            eoq = math.sqrt(2 * annual_demand * product.order_cost / (product.holding_cost * 365))
            
            # Round to nearest action quantity
            self.order_quantities[product.sku] = self._round_to_action(eoq)
            
    def _get_z_score(self, service_level: float) -> float:
        """Get z-score for given service level"""
        # Approximate z-scores for common service levels
        z_scores = {
            0.90: 1.28,
            0.95: 1.65,
            0.975: 1.96,
            0.99: 2.33
        }
        return z_scores.get(service_level, 1.65)
    
    def _round_to_action(self, quantity: float) -> int:
        """Round quantity to nearest available action"""
        action_quantities = self.env.action_quantities
        closest_idx = min(range(len(action_quantities)), 
                         key=lambda i: abs(action_quantities[i] - quantity))
        return closest_idx
    
    def select_action(self, observation: np.ndarray, info: dict) -> np.ndarray:
        """
        Select action based on reorder point policy
        
        Args:
            observation: Current state
            info: Additional information with inventory levels
            
        Returns:
            Action array
        """
        action = np.zeros(len(self.products), dtype=np.int32)
        
        for i, product in enumerate(self.products):
            current_inventory = info['inventory'].get(product.sku, 0)
            reorder_point = self.reorder_points[product.sku]
            
            # If inventory below reorder point, place order
            if current_inventory < reorder_point:
                action[i] = self.order_quantities[product.sku]
            else:
                action[i] = 0  # Don't order
                
        return action


class EOQPolicy(BaselinePolicy):
    """
    Economic Order Quantity (EOQ) Policy
    
    Classic Operations Research solution:
    - Q* = sqrt(2DS/H)
    - D = annual demand
    - S = order cost per order
    - H = holding cost per unit per year
    
    Orders EOQ quantity when inventory drops below certain threshold
    """
    
    def __init__(self, env, review_period: int = 7):
        """
        Initialize EOQ policy
        
        Args:
            env: Supply chain environment
            review_period: Days between inventory reviews
        """
        super().__init__(env)
        self.name = "EOQ Policy"
        self.review_period = review_period
        self.days_since_review = {p.sku: 0 for p in self.products}
        
        # Calculate EOQ for each product
        self.eoq_quantities = {}
        self.reorder_levels = {}
        
        for product in self.products:
            # EOQ formula: Q* = sqrt(2DS/H)
            annual_demand = product.base_demand * 365
            eoq = math.sqrt(
                (2 * annual_demand * product.order_cost) / 
                (product.holding_cost * 365)
            )
            
            self.eoq_quantities[product.sku] = self._round_to_action(eoq)
            
            # Reorder level = demand during lead time + review period
            self.reorder_levels[product.sku] = (
                product.base_demand * (product.lead_time + review_period)
            )
    
    def _round_to_action(self, quantity: float) -> int:
        """Round quantity to nearest available action"""
        action_quantities = self.env.action_quantities
        closest_idx = min(range(len(action_quantities)), 
                         key=lambda i: abs(action_quantities[i] - quantity))
        return closest_idx
    
    def select_action(self, observation: np.ndarray, info: dict) -> np.ndarray:
        """
        Select action based on EOQ policy
        
        Args:
            observation: Current state
            info: Additional information
            
        Returns:
            Action array
        """
        action = np.zeros(len(self.products), dtype=np.int32)
        
        for i, product in enumerate(self.products):
            current_inventory = info['inventory'].get(product.sku, 0)
            reorder_level = self.reorder_levels[product.sku]
            
            # Update days since last review
            self.days_since_review[product.sku] += 1
            
            # Check if it's time to review and inventory is low
            if (self.days_since_review[product.sku] >= self.review_period and 
                current_inventory < reorder_level):
                
                action[i] = self.eoq_quantities[product.sku]
                self.days_since_review[product.sku] = 0  # Reset counter
            else:
                action[i] = 0
                
        return action
    
    def reset(self):
        """Reset policy state"""
        self.days_since_review = {p.sku: 0 for p in self.products}


def create_baseline_policies(env) -> Dict[str, BaselinePolicy]:
    """
    Create all baseline policies
    
    Args:
        env: Supply chain environment
        
    Returns:
        Dictionary of {policy_name: policy_instance}
    """
    policies = {
        'random': RandomPolicy(env),
        'reorder_point': ReorderPointPolicy(env, service_level=0.95),
        'eoq': EOQPolicy(env, review_period=7)
    }
    
    return policies


def evaluate_policy(env, policy: BaselinePolicy, num_episodes: int = 10, verbose: bool = True) -> Dict:
    """
    Evaluate a baseline policy
    
    Args:
        env: Supply chain environment
        policy: Policy to evaluate
        num_episodes: Number of episodes to run
        verbose: Print progress
        
    Returns:
        Dictionary with evaluation results
    """
    episode_rewards = []
    episode_costs = []
    episode_summaries = []
    
    for episode in range(num_episodes):
        obs, info = env.reset(seed=42 + episode)
        policy.reset()
        
        done = False
        episode_reward = 0
        
        while not done:
            action = policy.select_action(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        # Get episode summary
        summary = env.get_episode_summary()
        episode_rewards.append(episode_reward)
        episode_costs.append(summary['total_cost'])
        episode_summaries.append(summary)
        
        if verbose:
            print(f"  Episode {episode + 1}/{num_episodes}: "
                  f"Reward=${episode_reward:,.0f}, Cost=${summary['total_cost']:,.0f}")
    
    # Calculate statistics
    results = {
        'policy_name': policy.name,
        'num_episodes': num_episodes,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_cost': np.mean(episode_costs),
        'std_cost': np.std(episode_costs),
        'min_cost': np.min(episode_costs),
        'max_cost': np.max(episode_costs),
        'episode_rewards': episode_rewards,
        'episode_costs': episode_costs,
        'summaries': episode_summaries
    }
    
    return results