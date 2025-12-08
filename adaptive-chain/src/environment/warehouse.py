"""
Warehouse entity managing inventory
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class Warehouse:
    """
    Represents a warehouse/distribution center
    
    Attributes:
        warehouse_id: Unique identifier
        name: Human-readable name
        location: (latitude, longitude)
        capacity: Maximum storage capacity (units)
        inventory: Current inventory levels {sku: quantity}
        pending_orders: Orders in transit {sku: [(quantity, arrival_day), ...]}
    """
    
    warehouse_id: str
    name: str
    location: tuple  # (lat, lon)
    capacity: int
    inventory: Dict[str, float] = field(default_factory=dict)
    pending_orders: Dict[str, List[tuple]] = field(default_factory=dict)
    capacity_violations: int = field(default=0)  # Track violations
    
    def __post_init__(self):
        """Initialize inventory tracking"""
        assert self.capacity > 0, "Capacity must be positive"
    
    def get_inventory_level(self, sku: str) -> float:
        """Get current inventory for a product"""
        return self.inventory.get(sku, 0.0)
    
    def get_total_inventory(self) -> float:
        """Get total inventory across all products"""
        return sum(self.inventory.values())
    
    def has_capacity(self, quantity: float) -> bool:
        """Check if warehouse can accept more inventory"""
        return self.get_total_inventory() + quantity <= self.capacity
    
    def add_inventory(self, sku: str, quantity: float) -> bool:
        """
        Add inventory to warehouse
        
        Returns:
            True if successful, False if exceeds capacity
        """
        if not self.has_capacity(quantity):
            self.capacity_violations += 1
            return False
        
        if sku not in self.inventory:
            self.inventory[sku] = 0.0
        
        self.inventory[sku] += quantity
        return True
    
    def remove_inventory(self, sku: str, quantity: float) -> float:
        """
        Remove inventory (for customer orders)
        
        Returns:
            Actual quantity removed (may be less if insufficient stock)
        """
        current = self.get_inventory_level(sku)
        actual_quantity = min(quantity, current)
        
        self.inventory[sku] = current - actual_quantity
        return actual_quantity
    
    def place_order(self, sku: str, quantity: float, arrival_day: int):
        """Place an order that will arrive on a future day"""
        if sku not in self.pending_orders:
            self.pending_orders[sku] = []
        
        self.pending_orders[sku].append((quantity, arrival_day))
    
    def receive_orders(self, current_day: int) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Receive orders that have arrived
        
        Returns:
            Tuple of (received_dict, rejected_dict)
            - received: {sku: quantity} successfully received
            - rejected: {sku: quantity} rejected due to capacity
        """
        received = {}
        rejected = {}
        
        for sku in list(self.pending_orders.keys()):
            # Filter orders arriving today
            arriving = [(q, d) for q, d in self.pending_orders[sku] if d == current_day]
            still_pending = [(q, d) for q, d in self.pending_orders[sku] if d > current_day]
            
            if arriving:
                total_quantity = sum(q for q, _ in arriving)
                if self.add_inventory(sku, total_quantity):
                    received[sku] = total_quantity
                else:
                    # Capacity exceeded - order rejected
                    rejected[sku] = total_quantity
                    # Silent failure - the negative reward teaches the RL agent
            
            # Update pending orders
            if still_pending:
                self.pending_orders[sku] = still_pending
            else:
                del self.pending_orders[sku]
        
        return received, rejected
    
    def get_capacity_utilization(self) -> float:
        """Get warehouse capacity utilization (0.0 to 1.0)"""
        return self.get_total_inventory() / self.capacity
    
    def get_stats(self) -> Dict:
        """Get warehouse statistics"""
        return {
            'total_inventory': self.get_total_inventory(),
            'capacity_utilization': self.get_capacity_utilization(),
            'capacity_violations': self.capacity_violations,
            'num_products': len(self.inventory),
            'pending_orders': sum(len(orders) for orders in self.pending_orders.values())
        }
    
    def __repr__(self):
        return f"Warehouse({self.warehouse_id}, inv={self.get_total_inventory():.0f}/{self.capacity})"


def create_sample_warehouses():
    """Create sample warehouses for testing"""
    return [
        Warehouse(
            warehouse_id="WH_EAST",
            name="East Coast DC",
            location=(40.7128, -74.0060),  # New York
            capacity=5000
        ),
        Warehouse(
            warehouse_id="WH_WEST",
            name="West Coast DC",
            location=(34.0522, -118.2437),  # Los Angeles
            capacity=5000
        ),
        Warehouse(
            warehouse_id="WH_CENTRAL",
            name="Central DC",
            location=(41.8781, -87.6298),  # Chicago
            capacity=7000
        ),
    ]