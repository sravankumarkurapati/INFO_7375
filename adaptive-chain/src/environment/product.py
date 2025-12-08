"""
Product entity representing items in the supply chain
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Product:
    """
    Represents a product/SKU in the supply chain
    
    Attributes:
        sku: Unique product identifier
        name: Human-readable product name
        base_demand: Average daily demand (units/day)
        demand_std: Standard deviation of demand
        holding_cost: Cost to store one unit for one day ($)
        stockout_cost: Penalty for not having stock when demanded ($)
        unit_cost: Purchase cost per unit ($)
        order_cost: Fixed cost per order ($)  ← ADDED THIS
        lead_time: Days to receive order after placing
    """
    
    sku: str
    name: str
    base_demand: float
    demand_std: float
    holding_cost: float
    stockout_cost: float
    unit_cost: float
    order_cost: float  # ← ADDED THIS LINE
    lead_time: int
    
    def __post_init__(self):
        """Validate product parameters"""
        assert self.base_demand > 0, "Base demand must be positive"
        assert self.demand_std >= 0, "Standard deviation cannot be negative"
        assert self.holding_cost >= 0, "Holding cost cannot be negative"
        assert self.stockout_cost >= 0, "Stockout cost cannot be negative"
        assert self.order_cost >= 0, "Order cost cannot be negative"  # ← ADDED THIS
        assert self.lead_time > 0, "Lead time must be positive"
    
    def generate_demand(self, day: int, random_seed: Optional[int] = None) -> float:
        """
        Generate realistic demand for a given day
        
        Args:
            day: Day number (for seasonal patterns)
            random_seed: Random seed for reproducibility
            
        Returns:
            Demand quantity (non-negative)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Base demand with noise
        demand = np.random.normal(self.base_demand, self.demand_std)
        
        # Add weekly seasonality (weekends have +20% demand)
        day_of_week = day % 7
        if day_of_week in [5, 6]:  # Weekend
            demand *= 1.2
        
        # Ensure non-negative
        demand = max(0, demand)
        
        return round(demand, 2)
    
    def __repr__(self):
        return f"Product({self.sku}, demand={self.base_demand:.1f}±{self.demand_std:.1f})"


def create_sample_products():
    """Create sample products for testing"""
    return [
        Product(
            sku="PROD_A",
            name="High-Volume Widget",
            base_demand=100.0,
            demand_std=15.0,
            holding_cost=2.0,
            stockout_cost=50.0,
            unit_cost=20.0,
            order_cost=100.0,  # ← ADDED THIS
            lead_time=3
        ),
        Product(
            sku="PROD_B",
            name="Medium-Volume Gadget",
            base_demand=50.0,
            demand_std=10.0,
            holding_cost=3.0,
            stockout_cost=80.0,
            unit_cost=35.0,
            order_cost=150.0,  # ← ADDED THIS
            lead_time=5
        ),
        Product(
            sku="PROD_C",
            name="Low-Volume Component",
            base_demand=20.0,
            demand_std=5.0,
            holding_cost=1.5,
            stockout_cost=100.0,
            unit_cost=15.0,
            order_cost=75.0,  # ← ADDED THIS
            lead_time=2
        ),
    ]