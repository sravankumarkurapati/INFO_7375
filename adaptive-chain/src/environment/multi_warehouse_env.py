"""
Multi-warehouse supply chain environment with coordination
INNOVATION: Inventory transfers + shared learning
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Optional, Tuple
import pandas as pd

from .product import Product, create_sample_products
from .warehouse import Warehouse, create_sample_warehouses
from .data_generator import create_training_data


class MultiWarehouseEnv(gym.Env):
    """
    Multi-warehouse supply chain with coordination
    
    Key Innovations:
    1. Inventory transfers between warehouses
    2. Regional demand patterns
    3. Shared state observation (communication)
    4. Global reward with local shaping
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        num_products: int = 3,
        num_warehouses: int = 3,
        episode_length: int = 180,
        enable_transfers: bool = True,
        transfer_cost_per_unit: float = 5.0,
        random_seed: Optional[int] = None,
        verbose: bool = False
    ):
        super().__init__()
        
        self.num_products = num_products
        self.num_warehouses = num_warehouses
        self.episode_length = episode_length
        self.enable_transfers = enable_transfers
        self.transfer_cost = transfer_cost_per_unit
        self.random_seed = random_seed
        self.verbose = verbose
        
        # Create products and warehouses
        all_products = create_sample_products()
        self.products = all_products[:num_products]
        
        all_warehouses = create_sample_warehouses()
        self.warehouses = all_warehouses[:num_warehouses]
        
        # Generate demand data
        training_data = create_training_data(
            num_days=episode_length,
            random_seed=random_seed if random_seed else 42
        )
        self.demand_data = training_data['demand']
        
        # Regional demand multipliers (East > Central > West)
        self.regional_multipliers = [1.3, 0.8, 1.0]  # East, West, Central
        
        # State space: Each warehouse sees:
        # - Own inventory (3)
        # - Pending orders (3)
        # - Demand forecast (3)
        # - Days until delivery (3)
        # - Own capacity util (1)
        # - Other warehouses' inventory (2 × 3 = 6)
        # = 19 features per warehouse × 3 warehouses = 57 total
        state_per_wh = num_products * 4 + 1 + (num_warehouses - 1) * num_products
        self.state_dim = state_per_wh * num_warehouses
        
        # Action: Each warehouse orders for each product
        action_quantities = [0, 100, 200, 500]
        self.action_quantities = action_quantities
        self.action_space = spaces.MultiDiscrete(
            [len(action_quantities)] * (num_products * num_warehouses)
        )
        
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.state_dim,),
            dtype=np.float32
        )
        
        # Target inventory
        self.target_inventory = {p.sku: p.base_demand * 5 for p in self.products}
        
        # Tracking
        self.current_day = 0
        self.total_reward = 0.0
        self.episode_costs = {
            'holding': 0.0,
            'stockout': 0.0,
            'order': 0.0,
            'transfer': 0.0,
            'imbalance': 0.0
        }
        self.transfer_count = 0
        self.history = []
        
    def _unflatten_action(self, flat_action: int) -> np.ndarray:
        """
        Unflatten action from single integer to array
        
        Args:
            flat_action: Integer representing all warehouse actions (0 to 262143)
            
        Returns:
            Array of 9 actions [wh1_p1, wh1_p2, wh1_p3, wh2_p1, wh2_p2, wh2_p3, wh3_p1, wh3_p2, wh3_p3]
        """
        num_actions = len(self.action_quantities)  # 4
        total_products = self.num_warehouses * self.num_products  # 9
        
        actions = []
        remaining = flat_action
        
        for i in range(total_products):
            actions.append(remaining % num_actions)
            remaining = remaining // num_actions
        
        return np.array(actions[::-1], dtype=np.int32)
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment"""
        super().reset(seed=seed)
        
        if seed is not None:
            self.random_seed = seed
            np.random.seed(seed)
        
        self.current_day = 0
        self.total_reward = 0.0
        self.episode_costs = {
            'holding': 0.0,
            'stockout': 0.0,
            'order': 0.0,
            'transfer': 0.0,
            'imbalance': 0.0
        }
        self.transfer_count = 0
        self.history = []
        
        # Initialize warehouses with regional variation
        for wh_idx, warehouse in enumerate(self.warehouses):
            warehouse.inventory = {}
            warehouse.pending_orders = {}
            warehouse.capacity_violations = 0
            
            multiplier = self.regional_multipliers[wh_idx]
            for product in self.products:
                initial = self.target_inventory[product.sku] * multiplier
                warehouse.add_inventory(product.sku, initial)
        
        obs = self._get_observation()
        info = self._get_info()
        return obs, info
    
    def step(self, action):
        """Execute one timestep"""
        # Handle both scalar (flattened) and array actions
        if np.isscalar(action) or (isinstance(action, np.ndarray) and action.size == 1):
            # Flattened action from wrapped env - unflatten it
            action_int = int(action)
            action_array = self._unflatten_action(action_int)
        else:
            # Already unflattened array
            action_array = np.array(action, dtype=np.int32)
        
        # Ensure we have 9 actions
        if action_array.size != self.num_warehouses * self.num_products:
            raise ValueError(f"Expected {self.num_warehouses * self.num_products} actions, got {action_array.size}")
        
        # Reshape: [wh0_p0, wh0_p1, wh0_p2, wh1_p0, wh1_p1, wh1_p2, wh2_p0, wh2_p1, wh2_p2]
        actions_per_warehouse = action_array.reshape(self.num_warehouses, self.num_products)
        
        total_cost = 0.0
        daily_stockouts = 0.0
        
        # 1. Receive orders
        for warehouse in self.warehouses:
            warehouse.receive_orders(self.current_day)
        
        # 2. INNOVATION: Proactive inventory balancing (weekly)
        if self.enable_transfers and self.current_day % 7 == 0:
            transfer_cost = self._balance_inventory_proactive()
            total_cost += transfer_cost
            self.episode_costs['transfer'] += transfer_cost
        
        # 3. Fulfill regional demand
        for wh_idx, warehouse in enumerate(self.warehouses):
            regional_demand = self._get_regional_demand(wh_idx)
            
            for product in self.products:
                demand = regional_demand.get(product.sku, 0.0)
                fulfilled = warehouse.remove_inventory(product.sku, demand)
                stockout = demand - fulfilled
                
                if stockout > 0:
                    daily_stockouts += stockout * product.stockout_cost
                    
                    # INNOVATION: Emergency transfer from nearest warehouse
                    if self.enable_transfers:
                        emergency_cost = self._emergency_transfer(
                            to_wh_idx=wh_idx,
                            product=product,
                            needed=stockout
                        )
                        total_cost += emergency_cost
        
        total_cost += daily_stockouts
        self.episode_costs['stockout'] += daily_stockouts
        
        # 4. Execute orders
        order_cost = 0.0
        for wh_idx, warehouse in enumerate(self.warehouses):
            wh_actions = actions_per_warehouse[wh_idx]
            
            for prod_idx, product in enumerate(self.products):
                action_idx = wh_actions[prod_idx]
                order_qty = self.action_quantities[action_idx]
                
                if order_qty > 0:
                    arrival_day = self.current_day + product.lead_time
                    warehouse.place_order(product.sku, order_qty, arrival_day)
                    order_cost += product.order_cost + (order_qty * product.unit_cost)
        
        total_cost += order_cost
        self.episode_costs['order'] += order_cost
        
        # 5. Holding costs
        holding_cost = 0.0
        for warehouse in self.warehouses:
            for product in self.products:
                inventory = warehouse.get_inventory_level(product.sku)
                holding_cost += inventory * product.holding_cost
        
        total_cost += holding_cost
        self.episode_costs['holding'] += holding_cost
        
        # 6. System-wide imbalance penalty
        imbalance_cost = self._calculate_system_imbalance()
        total_cost += imbalance_cost
        self.episode_costs['imbalance'] += imbalance_cost
        
        # 7. Reward (global system cost)
        reward = -total_cost
        self.total_reward += reward
        
        # 8. Advance day
        self.current_day += 1
        
        # 9. Check done
        terminated = self.current_day >= self.episode_length
        truncated = False
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get coordinated observation (COMMUNICATION PROTOCOL)
        
        Each warehouse sees:
        - Its own state
        - Other warehouses' inventory (coordination signal)
        """
        state = []
        
        for wh_idx, warehouse in enumerate(self.warehouses):
            # Own inventory
            for product in self.products:
                inv = warehouse.get_inventory_level(product.sku)
                state.append(min(1.0, inv / warehouse.capacity))
            
            # Pending orders
            for product in self.products:
                pending = sum(q for q, _ in warehouse.pending_orders.get(product.sku, []))
                state.append(min(1.0, pending / warehouse.capacity))
            
            # Demand forecast
            for product in self.products:
                forecast = self._get_demand_forecast(product.sku, wh_idx)
                state.append(min(1.0, forecast / (product.base_demand * 3)))
            
            # Days until delivery
            for product in self.products:
                days = self._get_days_until_delivery(warehouse, product.sku)
                state.append(min(1.0, days / 10.0))
            
            # Own capacity
            state.append(warehouse.get_capacity_utilization())
            
            # OTHER WAREHOUSES' INVENTORY (Communication!)
            for other_idx, other_wh in enumerate(self.warehouses):
                if other_idx != wh_idx:
                    for product in self.products:
                        other_inv = other_wh.get_inventory_level(product.sku)
                        state.append(min(1.0, other_inv / other_wh.capacity))
        
        return np.array(state, dtype=np.float32)
    
    def _get_regional_demand(self, warehouse_idx: int) -> Dict[str, float]:
        """Get demand for specific warehouse region"""
        base_demand = self._get_daily_demand()
        multiplier = self.regional_multipliers[warehouse_idx]
        
        return {sku: demand * multiplier for sku, demand in base_demand.items()}
    
    def _get_daily_demand(self) -> Dict[str, float]:
        """Get base daily demand"""
        daily_demand = {}
        for product in self.products:
            demand_row = self.demand_data[
                (self.demand_data['day'] == self.current_day) &
                (self.demand_data['sku'] == product.sku)
            ]
            if not demand_row.empty:
                demand = demand_row.iloc[0]['demand']
            else:
                demand = product.generate_demand(self.current_day, self.random_seed)
            daily_demand[product.sku] = demand
        return daily_demand
    
    def _get_demand_forecast(self, sku: str, wh_idx: int, days: int = 7) -> float:
        """Get regional demand forecast"""
        base_forecast = self.demand_data[
            (self.demand_data['sku'] == sku) &
            (self.demand_data['day'] >= self.current_day) &
            (self.demand_data['day'] < self.current_day + days)
        ]
        if not base_forecast.empty:
            avg_demand = base_forecast['demand'].mean()
        else:
            product = next(p for p in self.products if p.sku == sku)
            avg_demand = product.base_demand
        
        return avg_demand * self.regional_multipliers[wh_idx]
    
    def _get_days_until_delivery(self, warehouse, sku: str) -> float:
        """Days until next delivery"""
        pending = warehouse.pending_orders.get(sku, [])
        if not pending:
            return 0.0
        next_delivery = min(day for _, day in pending)
        return float(max(0, next_delivery - self.current_day))
    
    def _balance_inventory_proactive(self) -> float:
        """
        INNOVATION: Proactive inventory balancing
        Move excess inventory from low-demand to high-demand warehouses
        """
        transfer_cost = 0.0
        
        for product in self.products:
            # Get inventory levels across all warehouses
            inventories = [wh.get_inventory_level(product.sku) for wh in self.warehouses]
            targets = [self.target_inventory[product.sku] * mult 
                      for mult in self.regional_multipliers]
            
            # Find warehouses with excess and shortage
            excess_wh = []
            shortage_wh = []
            
            for i, (inv, target) in enumerate(zip(inventories, targets)):
                if inv > target * 1.5:  # 50% above target
                    excess = inv - target
                    excess_wh.append((i, excess))
                elif inv < target * 0.5:  # 50% below target
                    shortage = target - inv
                    shortage_wh.append((i, shortage))
            
            # Transfer from excess to shortage
            for from_idx, excess in excess_wh:
                for to_idx, shortage in shortage_wh:
                    if excess > 0 and shortage > 0:
                        transfer_qty = min(excess, shortage, 100)  # Max 100 units per transfer
                        
                        # Execute transfer
                        removed = self.warehouses[from_idx].remove_inventory(product.sku, transfer_qty)
                        added = self.warehouses[to_idx].add_inventory(product.sku, removed)
                        
                        if added:
                            transfer_cost += removed * self.transfer_cost
                            self.transfer_count += 1
                            
                            excess -= removed
                            shortage -= removed
        
        return transfer_cost
    
    def _emergency_transfer(self, to_wh_idx: int, product: Product, needed: float) -> float:
        """
        INNOVATION: Emergency transfer during stockout
        Try to fulfill from nearby warehouses
        """
        transfer_cost = 0.0
        remaining_need = needed
        
        # Try each other warehouse
        for from_idx, warehouse in enumerate(self.warehouses):
            if from_idx != to_wh_idx and remaining_need > 0:
                available = warehouse.get_inventory_level(product.sku)
                
                if available > product.base_demand * 2:  # Only if they have excess
                    transfer_qty = min(remaining_need, available - product.base_demand * 2)
                    
                    if transfer_qty > 0:
                        # Execute emergency transfer
                        removed = warehouse.remove_inventory(product.sku, transfer_qty)
                        added = self.warehouses[to_wh_idx].add_inventory(product.sku, removed)
                        
                        if added:
                            # Emergency transfer is expensive (2x normal)
                            transfer_cost += removed * self.transfer_cost * 2
                            remaining_need -= removed
                            self.transfer_count += 1
        
        # If still have shortage, incur stockout cost
        if remaining_need > 0:
            transfer_cost += remaining_need * product.stockout_cost
        
        return transfer_cost
    
    def _calculate_system_imbalance(self) -> float:
        """
        Calculate cost of inventory imbalance across system
        Encourages balanced distribution
        """
        imbalance_cost = 0.0
        
        for product in self.products:
            inventories = [wh.get_inventory_level(product.sku) for wh in self.warehouses]
            targets = [self.target_inventory[product.sku] * mult 
                      for mult in self.regional_multipliers]
            
            # Penalty for deviation from targets
            for inv, target in zip(inventories, targets):
                deviation = abs(inv - target)
                imbalance_cost += deviation * 0.05  # Small penalty
        
        return imbalance_cost
    
    def _get_info(self) -> dict:
        """Get info dictionary"""
        info = {
            'day': self.current_day,
            'total_reward': self.total_reward,
            'episode_costs': self.episode_costs.copy(),
            'transfer_count': self.transfer_count,
            'warehouses': []
        }
        
        for wh in self.warehouses:
            wh_info = {
                'id': wh.warehouse_id,
                'inventory': {p.sku: wh.get_inventory_level(p.sku) for p in self.products},
                'capacity_util': wh.get_capacity_utilization(),
                'total_inventory': wh.get_total_inventory()
            }
            info['warehouses'].append(wh_info)
        
        return info
    
    def get_episode_summary(self) -> dict:
        """Get episode summary"""
        total_cost = sum(self.episode_costs.values())
        
        summary = {
            'total_reward': self.total_reward,
            'total_cost': total_cost,
            'holding_cost': self.episode_costs['holding'],
            'stockout_cost': self.episode_costs['stockout'],
            'order_cost': self.episode_costs['order'],
            'transfer_cost': self.episode_costs['transfer'],
            'imbalance_cost': self.episode_costs['imbalance'],
            'avg_daily_cost': total_cost / self.episode_length,
            'transfer_count': self.transfer_count,
            'avg_transfer_per_day': self.transfer_count / self.episode_length
        }
        
        return summary
    
    def render(self, mode='human'):
        """Render environment"""
        if mode == 'human':
            print(f"\n=== Day {self.current_day} ===")
            print(f"System Cost: ${-self.total_reward:,.0f}")
            print(f"Transfers: {self.transfer_count}")
            for wh in self.warehouses:
                print(f"\n{wh.warehouse_id}:")
                for p in self.products:
                    inv = wh.get_inventory_level(p.sku)
                    print(f"  {p.sku}: {inv:.0f} units")