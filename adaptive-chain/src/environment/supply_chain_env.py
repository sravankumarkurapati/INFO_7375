"""
Gymnasium environment for supply chain management - OPTIMIZED VERSION
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Optional, Tuple
import pandas as pd

from .product import Product, create_sample_products
from .warehouse import Warehouse, create_sample_warehouses
from .data_generator import create_training_data


class SupplyChainEnv(gym.Env):
    """
    Supply Chain Management Environment - OPTIMIZED
    
    Improvements:
    - Reduced action space (4 actions instead of 5)
    - Better reward shaping with inventory balance
    - Penalty for excessive ordering
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        num_products: int = 3,
        num_warehouses: int = 3,
        episode_length: int = 180,
        random_seed: Optional[int] = None,
        verbose: bool = False
    ):
        """Initialize supply chain environment"""
        super().__init__()
        
        self.num_products = num_products
        self.num_warehouses = 1  # Start with single warehouse for DQN
        self.episode_length = episode_length
        self.random_seed = random_seed
        self.verbose = verbose
        
        # Create products and warehouses
        all_products = create_sample_products()
        self.products = all_products[:num_products]
        
        all_warehouses = create_sample_warehouses()
        self.warehouses = [all_warehouses[0]]  # Use first warehouse
        
        # Generate demand data
        training_data = create_training_data(
            num_days=episode_length,
            random_seed=random_seed if random_seed else 42
        )
        self.demand_data = training_data['demand']
        self.disruptions = training_data['disruptions']
        
        # State space dimensions
        self.state_dim = num_products * 4 + 1  # +1 for capacity utilization
        
        # OPTIMIZED: Reduced action space (4 actions instead of 5)
        self.action_quantities = [0, 100, 200, 500]  # 4^3 = 64 combinations
        self.action_space = spaces.MultiDiscrete([len(self.action_quantities)] * num_products)
        
        # Observation space
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.state_dim,),
            dtype=np.float32
        )
        
        # Target inventory levels (for reward shaping)
        self.target_inventory = {
            p.sku: p.base_demand * 5  # 5 days coverage
            for p in self.products
        }
        
        # Episode tracking
        self.current_day = 0
        self.total_reward = 0.0
        self.episode_costs = {
            'holding': 0.0,
            'stockout': 0.0,
            'order': 0.0,
            'capacity_violation': 0.0,
            'imbalance': 0.0  # NEW
        }
        
        # History tracking
        self.history = []
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        if seed is not None:
            self.random_seed = seed
            np.random.seed(seed)
        
        # Reset day counter
        self.current_day = 0
        self.total_reward = 0.0
        self.episode_costs = {
            'holding': 0.0,
            'stockout': 0.0,
            'order': 0.0,
            'capacity_violation': 0.0,
            'imbalance': 0.0
        }
        self.history = []
        
        # Reset warehouses
        for warehouse in self.warehouses:
            warehouse.inventory = {}
            warehouse.pending_orders = {}
            warehouse.capacity_violations = 0
            
            # Initialize with starting inventory (target level)
            for product in self.products:
                initial_stock = self.target_inventory[product.sku]
                warehouse.add_inventory(product.sku, initial_stock)
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one time step"""
        warehouse = self.warehouses[0]
        
        # 1. Process incoming orders (from previous actions)
        received, rejected = warehouse.receive_orders(self.current_day)
        
        # Penalize capacity violations
        capacity_violation_cost = 0.0
        if rejected:
            for sku, qty in rejected.items():
                product = next(p for p in self.products if p.sku == sku)
                capacity_violation_cost += qty * product.unit_cost * 0.5
        
        # 2. Generate customer demand for today
        daily_demand = self._get_daily_demand()
        
        # 3. Fulfill demand (or record stockouts)
        stockout_cost = 0.0
        for product in self.products:
            demand = daily_demand.get(product.sku, 0.0)
            fulfilled = warehouse.remove_inventory(product.sku, demand)
            stockout = demand - fulfilled
            
            if stockout > 0:
                stockout_cost += stockout * product.stockout_cost
        
        # 4. Execute new orders (agent's action)
        order_cost = 0.0
        for i, product in enumerate(self.products):
            action_idx = action[i]
            order_quantity = self.action_quantities[action_idx]
            
            if order_quantity > 0:
                # Place order
                arrival_day = self.current_day + product.lead_time
                warehouse.place_order(product.sku, order_quantity, arrival_day)
                
                # Calculate order cost
                order_cost += product.order_cost + (order_quantity * product.unit_cost)
        
        # 5. Calculate holding costs
        holding_cost = 0.0
        for product in self.products:
            inventory = warehouse.get_inventory_level(product.sku)
            holding_cost += inventory * product.holding_cost
        
        # 6. NEW: Inventory imbalance penalty (reward shaping)
        imbalance_cost = 0.0
        for product in self.products:
            current_inv = warehouse.get_inventory_level(product.sku)
            target_inv = self.target_inventory[product.sku]
            
            # Penalize being too far from target
            deviation = abs(current_inv - target_inv)
            imbalance_cost += deviation * 0.1  # Small penalty for imbalance
        
        # 7. Calculate total cost and reward
        total_cost = (
            holding_cost + 
            stockout_cost + 
            order_cost + 
            capacity_violation_cost + 
            imbalance_cost
        )
        reward = -total_cost  # Negative cost as reward
        
        # Track costs
        self.episode_costs['holding'] += holding_cost
        self.episode_costs['stockout'] += stockout_cost
        self.episode_costs['order'] += order_cost
        self.episode_costs['capacity_violation'] += capacity_violation_cost
        self.episode_costs['imbalance'] += imbalance_cost
        self.total_reward += reward
        
        # 8. Record history
        self.history.append({
            'day': self.current_day,
            'inventory': {p.sku: warehouse.get_inventory_level(p.sku) for p in self.products},
            'demand': daily_demand,
            'action': action.copy(),
            'costs': {
                'holding': holding_cost,
                'stockout': stockout_cost,
                'order': order_cost,
                'capacity_violation': capacity_violation_cost,
                'imbalance': imbalance_cost
            },
            'reward': reward,
            'received': received,
            'rejected': rejected
        })
        
        # 9. Advance to next day
        self.current_day += 1
        
        # 10. Check if episode is done
        terminated = self.current_day >= self.episode_length
        truncated = False
        
        # 11. Get next observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation"""
        warehouse = self.warehouses[0]
        state = []
        
        for product in self.products:
            # Inventory level (normalized by capacity)
            inventory = warehouse.get_inventory_level(product.sku)
            inventory_norm = inventory / warehouse.capacity
            state.append(inventory_norm)
            
            # Pending orders (normalized)
            pending = sum(q for q, _ in warehouse.pending_orders.get(product.sku, []))
            pending_norm = pending / warehouse.capacity
            state.append(pending_norm)
            
            # Demand forecast (7-day average, normalized)
            forecast = self._get_demand_forecast(product.sku, days=7)
            forecast_norm = forecast / (product.base_demand * 2)
            state.append(min(1.0, forecast_norm))
            
            # Days until next delivery (normalized by max lead time)
            days_until = self._get_days_until_delivery(product.sku)
            days_norm = days_until / 10.0
            state.append(min(1.0, days_norm))
        
        # Warehouse capacity utilization
        utilization = warehouse.get_capacity_utilization()
        state.append(utilization)
        
        return np.array(state, dtype=np.float32)
    
    def _get_daily_demand(self) -> Dict[str, float]:
        """Get demand for current day"""
        daily_demand = {}
        
        for product in self.products:
            # Get demand from generated data
            demand_row = self.demand_data[
                (self.demand_data['day'] == self.current_day) &
                (self.demand_data['sku'] == product.sku)
            ]
            
            if not demand_row.empty:
                demand = demand_row.iloc[0]['demand']
            else:
                # Fallback to generated demand
                demand = product.generate_demand(self.current_day, self.random_seed)
            
            daily_demand[product.sku] = demand
        
        return daily_demand
    
    def _get_demand_forecast(self, sku: str, days: int = 7) -> float:
        """Get average demand forecast for next N days"""
        future_demand = self.demand_data[
            (self.demand_data['sku'] == sku) &
            (self.demand_data['day'] >= self.current_day) &
            (self.demand_data['day'] < self.current_day + days)
        ]
        
        if not future_demand.empty:
            return future_demand['demand'].mean()
        else:
            # Fallback to product base demand
            product = next(p for p in self.products if p.sku == sku)
            return product.base_demand
    
    def _get_days_until_delivery(self, sku: str) -> float:
        """Get days until next delivery arrives"""
        warehouse = self.warehouses[0]
        pending = warehouse.pending_orders.get(sku, [])
        
        if not pending:
            return 0.0
        
        # Get earliest delivery
        next_delivery_day = min(day for _, day in pending)
        days_until = max(0, next_delivery_day - self.current_day)
        
        return float(days_until)
    
    def _get_info(self) -> dict:
        """Get additional information"""
        warehouse = self.warehouses[0]
        
        info = {
            'day': self.current_day,
            'total_reward': self.total_reward,
            'episode_costs': self.episode_costs.copy(),
            'inventory': {p.sku: warehouse.get_inventory_level(p.sku) for p in self.products},
            'capacity_utilization': warehouse.get_capacity_utilization(),
            'capacity_violations': warehouse.capacity_violations
        }
        
        return info
    
    def render(self, mode='human'):
        """Render the environment"""
        if mode == 'human':
            warehouse = self.warehouses[0]
            print(f"\n=== Day {self.current_day} ===")
            print(f"Total Reward: ${self.total_reward:,.2f}")
            print(f"Capacity: {warehouse.get_capacity_utilization()*100:.1f}%")
            print("\nInventory:")
            for product in self.products:
                inv = warehouse.get_inventory_level(product.sku)
                target = self.target_inventory[product.sku]
                print(f"  {product.sku}: {inv:.0f} units (target: {target:.0f})")
    
    def get_episode_summary(self) -> dict:
        """Get summary of completed episode"""
        warehouse = self.warehouses[0]
        total_cost = sum(self.episode_costs.values())
        
        summary = {
            'total_reward': self.total_reward,
            'total_cost': total_cost,
            'holding_cost': self.episode_costs['holding'],
            'stockout_cost': self.episode_costs['stockout'],
            'order_cost': self.episode_costs['order'],
            'capacity_violation_cost': self.episode_costs['capacity_violation'],
            'imbalance_cost': self.episode_costs['imbalance'],
            'avg_daily_cost': total_cost / self.episode_length,
            'episode_length': self.episode_length,
            'capacity_violations': warehouse.capacity_violations,
            'history': self.history
        }
        
        return summary