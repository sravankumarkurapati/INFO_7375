"""
Generate realistic supply chain data for training and testing
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from datetime import datetime, timedelta


class DemandPatternGenerator:
    """
    Generate realistic demand patterns with multiple components:
    - Base demand (product-specific)
    - Seasonal variations (weekly, monthly)
    - Trends (gradual increase/decrease)
    - Random noise
    - Promotional events
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize generator
        
        Args:
            random_seed: Seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def generate_demand_series(
        self,
        base_demand: float,
        std_dev: float,
        num_days: int,
        trend_rate: float = 0.0,
        seasonality_amplitude: float = 0.15,
        promo_probability: float = 0.05,
        promo_multiplier: float = 2.0
    ) -> np.ndarray:
        """
        Generate a realistic demand time series
        
        Args:
            base_demand: Average daily demand
            std_dev: Standard deviation of noise
            num_days: Number of days to generate
            trend_rate: Daily trend rate (0.01 = 1% increase per day)
            seasonality_amplitude: Amplitude of seasonal variation (0.15 = ±15%)
            promo_probability: Probability of promotional event each day
            promo_multiplier: Demand multiplier during promotions
            
        Returns:
            Array of daily demand values
        """
        demand = np.zeros(num_days)
        
        for day in range(num_days):
            # 1. Base demand
            daily_demand = base_demand
            
            # 2. Trend component (gradual change over time)
            trend = base_demand * trend_rate * day
            daily_demand += trend
            
            # 3. Weekly seasonality (weekends higher)
            day_of_week = day % 7
            if day_of_week in [5, 6]:  # Weekend
                seasonal_factor = 1.0 + seasonality_amplitude
            else:
                seasonal_factor = 1.0
            daily_demand *= seasonal_factor
            
            # 4. Monthly seasonality (end of month spike)
            day_of_month = day % 30
            if day_of_month >= 25:  # Last 5 days of month
                monthly_factor = 1.0 + seasonality_amplitude * 0.5
                daily_demand *= monthly_factor
            
            # 5. Random noise
            noise = np.random.normal(0, std_dev)
            daily_demand += noise
            
            # 6. Promotional events (random spikes)
            if np.random.random() < promo_probability:
                daily_demand *= promo_multiplier
            
            # Ensure non-negative
            demand[day] = max(0, daily_demand)
        
        return demand
    
    def generate_multi_product_demand(
        self,
        products: List[Dict],
        num_days: int
    ) -> pd.DataFrame:
        """
        Generate demand for multiple products
        
        Args:
            products: List of product dictionaries with demand parameters
            num_days: Number of days to simulate
            
        Returns:
            DataFrame with columns: day, sku, demand
        """
        all_demand = []
        
        for product in products:
            sku = product['sku']
            base_demand = product.get('base_demand', 100.0)
            std_dev = product.get('demand_std', 15.0)
            trend = product.get('trend_rate', 0.0)
            
            demand_series = self.generate_demand_series(
                base_demand=base_demand,
                std_dev=std_dev,
                num_days=num_days,
                trend_rate=trend,
                seasonality_amplitude=0.15,
                promo_probability=0.05,
                promo_multiplier=2.0
            )
            
            for day, demand_val in enumerate(demand_series):
                all_demand.append({
                    'day': day,
                    'sku': sku,
                    'demand': round(demand_val, 2)
                })
        
        df = pd.DataFrame(all_demand)
        return df


class DisruptionGenerator:
    """
    Generate realistic supply chain disruptions
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize disruption generator"""
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def generate_disruptions(
        self,
        num_days: int,
        disruption_probability: float = 0.05
    ) -> List[Dict]:
        """
        Generate random disruptions throughout the simulation
        
        Args:
            num_days: Total simulation days
            disruption_probability: Probability of disruption each day
            
        Returns:
            List of disruption events
        """
        disruptions = []
        
        disruption_types = [
            {
                'type': 'supplier_delay',
                'description': 'Supplier delivery delayed',
                'lead_time_multiplier': 2.0,
                'duration': 5
            },
            {
                'type': 'demand_spike',
                'description': 'Unexpected demand surge',
                'demand_multiplier': 3.0,
                'duration': 3
            },
            {
                'type': 'capacity_reduction',
                'description': 'Warehouse capacity temporarily reduced',
                'capacity_multiplier': 0.7,
                'duration': 7
            },
            {
                'type': 'transportation_issue',
                'description': 'Transportation network disrupted',
                'cost_multiplier': 1.5,
                'duration': 4
            },
        ]
        
        for day in range(num_days):
            if np.random.random() < disruption_probability:
                disruption = np.random.choice(disruption_types)
                event = {
                    'start_day': day,
                    'end_day': day + disruption['duration'],
                    'type': disruption['type'],
                    'description': disruption['description'],
                    'parameters': {k: v for k, v in disruption.items() 
                                 if k not in ['type', 'description', 'duration']}
                }
                disruptions.append(event)
        
        return disruptions
    
    def create_test_scenarios(self) -> List[Dict]:
        """
        Create predefined test scenarios for evaluation
        
        Returns:
            List of test scenario configurations
        """
        scenarios = [
            {
                'name': 'normal_operations',
                'description': 'Standard operating conditions',
                'disruptions': [],
                'demand_multiplier': 1.0
            },
            {
                'name': 'high_demand',
                'description': 'Sustained high demand period',
                'disruptions': [],
                'demand_multiplier': 1.5
            },
            {
                'name': 'supplier_crisis',
                'description': 'Major supplier delay',
                'disruptions': [
                    {
                        'start_day': 30,
                        'end_day': 40,
                        'type': 'supplier_delay',
                        'parameters': {'lead_time_multiplier': 3.0}
                    }
                ],
                'demand_multiplier': 1.0
            },
            {
                'name': 'demand_shock',
                'description': 'Sudden demand spike',
                'disruptions': [
                    {
                        'start_day': 45,
                        'end_day': 50,
                        'type': 'demand_spike',
                        'parameters': {'demand_multiplier': 4.0}
                    }
                ],
                'demand_multiplier': 1.0
            },
            {
                'name': 'capacity_crisis',
                'description': 'Warehouse capacity constraints',
                'disruptions': [
                    {
                        'start_day': 20,
                        'end_day': 35,
                        'type': 'capacity_reduction',
                        'parameters': {'capacity_multiplier': 0.5}
                    }
                ],
                'demand_multiplier': 1.0
            }
        ]
        
        return scenarios


def create_training_data(num_days: int = 180, random_seed: int = 42) -> Dict:
    """
    Create complete training dataset
    
    Args:
        num_days: Number of days to simulate
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with demand data and disruptions
    """
    # Product configurations
    products = [
        {
            'sku': 'PROD_A',
            'base_demand': 100.0,
            'demand_std': 15.0,
            'trend_rate': 0.001  # Slight upward trend
        },
        {
            'sku': 'PROD_B',
            'base_demand': 50.0,
            'demand_std': 10.0,
            'trend_rate': -0.0005  # Slight downward trend
        },
        {
            'sku': 'PROD_C',
            'base_demand': 20.0,
            'demand_std': 5.0,
            'trend_rate': 0.0  # No trend
        },
    ]
    
    # Generate demand
    demand_gen = DemandPatternGenerator(random_seed=random_seed)
    demand_df = demand_gen.generate_multi_product_demand(products, num_days)
    
    # Generate disruptions
    disruption_gen = DisruptionGenerator(random_seed=random_seed)
    disruptions = disruption_gen.generate_disruptions(num_days, disruption_probability=0.03)
    
    # Get test scenarios
    test_scenarios = disruption_gen.create_test_scenarios()
    
    return {
        'demand': demand_df,
        'disruptions': disruptions,
        'test_scenarios': test_scenarios,
        'products': products
    }