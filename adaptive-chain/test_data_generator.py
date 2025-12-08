"""
Test data generation and visualization
"""
import sys
import os
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from environment.data_generator import (
    DemandPatternGenerator, 
    DisruptionGenerator,
    create_training_data
)
from environment.viz_utils import (
    plot_demand_patterns,
    plot_demand_statistics,
    visualize_disruptions
)


def test_demand_generation():
    """Test demand pattern generation"""
    print("=" * 60)
    print("Testing Demand Generation")
    print("=" * 60)
    
    generator = DemandPatternGenerator(random_seed=42)
    
    # Generate simple demand series
    demand = generator.generate_demand_series(
        base_demand=100.0,
        std_dev=15.0,
        num_days=30,
        trend_rate=0.001,
        seasonality_amplitude=0.15,
        promo_probability=0.1
    )
    
    print(f"\n✓ Generated {len(demand)} days of demand")
    print(f"  Mean: {demand.mean():.2f}")
    print(f"  Std:  {demand.std():.2f}")
    print(f"  Min:  {demand.min():.2f}")
    print(f"  Max:  {demand.max():.2f}")
    
    # Check for promotional spikes
    promo_days = sum(1 for d in demand if d > 150)
    print(f"  Promotional spike days: {promo_days}")
    
    print("\n✅ Demand generation working!\n")
    return True


def test_multi_product_demand():
    """Test multi-product demand generation"""
    print("=" * 60)
    print("Testing Multi-Product Demand")
    print("=" * 60)
    
    products = [
        {'sku': 'PROD_A', 'base_demand': 100.0, 'demand_std': 15.0},
        {'sku': 'PROD_B', 'base_demand': 50.0, 'demand_std': 10.0},
        {'sku': 'PROD_C', 'base_demand': 20.0, 'demand_std': 5.0},
    ]
    
    generator = DemandPatternGenerator(random_seed=42)
    demand_df = generator.generate_multi_product_demand(products, num_days=90)
    
    print(f"\n✓ Generated demand for {len(products)} products over 90 days")
    print(f"  Total records: {len(demand_df)}")
    print(f"  Date range: Day {demand_df['day'].min()} to Day {demand_df['day'].max()}")
    
    print("\n📊 Demand by Product:")
    for sku in products:
        sku_data = demand_df[demand_df['sku'] == sku['sku']]
        print(f"  {sku['sku']}: Mean={sku_data['demand'].mean():.1f}, "
              f"Std={sku_data['demand'].std():.1f}")
    
    print("\n✅ Multi-product demand working!\n")
    return demand_df


def test_disruptions():
    """Test disruption generation"""
    print("=" * 60)
    print("Testing Disruption Generation")
    print("=" * 60)
    
    generator = DisruptionGenerator(random_seed=42)
    disruptions = generator.generate_disruptions(
        num_days=180,
        disruption_probability=0.05
    )
    
    print(f"\n✓ Generated {len(disruptions)} disruption events")
    
    if disruptions:
        print("\n📋 Disruption Summary:")
        disruption_types = {}
        for d in disruptions:
            dtype = d['type']
            disruption_types[dtype] = disruption_types.get(dtype, 0) + 1
        
        for dtype, count in disruption_types.items():
            print(f"  {dtype}: {count} events")
        
        print(f"\n  First disruption: Day {disruptions[0]['start_day']}")
        print(f"  Type: {disruptions[0]['type']}")
        print(f"  Description: {disruptions[0]['description']}")
    
    print("\n✅ Disruption generation working!\n")
    return disruptions


def test_complete_dataset():
    """Test complete dataset generation"""
    print("=" * 60)
    print("Testing Complete Dataset Creation")
    print("=" * 60)
    
    data = create_training_data(num_days=180, random_seed=42)
    
    print(f"\n✓ Created complete training dataset")
    print(f"  Demand records: {len(data['demand'])}")
    print(f"  Disruptions: {len(data['disruptions'])}")
    print(f"  Test scenarios: {len(data['test_scenarios'])}")
    print(f"  Products: {len(data['products'])}")
    
    print("\n📋 Test Scenarios Available:")
    for scenario in data['test_scenarios']:
        print(f"  - {scenario['name']}: {scenario['description']}")
    
    print("\n✅ Complete dataset working!\n")
    return data


def test_visualizations(demand_df, disruptions):
    """Test visualization functions"""
    print("=" * 60)
    print("Testing Visualizations")
    print("=" * 60)
    
    print("\n📊 Creating visualizations...")
    
    # Plot demand patterns
    fig1 = plot_demand_patterns(demand_df, "90-Day Demand Patterns")
    print("  ✓ Demand pattern plot created")
    
    # Plot statistics
    fig2 = plot_demand_statistics(demand_df)
    print("  ✓ Statistics plot created")
    
    # Plot disruptions
    if disruptions:
        fig3 = visualize_disruptions(disruptions, num_days=180)
        print("  ✓ Disruption timeline created")
    
    # Save plots
    os.makedirs('data', exist_ok=True)
    fig1.savefig('data/demand_patterns.png', dpi=100, bbox_inches='tight')
    fig2.savefig('data/demand_statistics.png', dpi=100, bbox_inches='tight')
    if disruptions:
        fig3.savefig('data/disruption_timeline.png', dpi=100, bbox_inches='tight')
    
    print(f"\n  💾 Plots saved to data/ directory")
    print("\n✅ Visualizations working!\n")
    
    plt.close('all')  # Close all figures


if __name__ == "__main__":
    try:
        # Run all tests
        test_demand_generation()
        test_disruptions()
        demand_df = test_multi_product_demand()
        data = test_complete_dataset()
        test_visualizations(data['demand'], data['disruptions'])
        
        print("=" * 60)
        print("🎉 ALL DATA GENERATION TESTS PASSED!")
        print("=" * 60)
        print("\n✅ Ready to proceed to Step 1.4 (Gymnasium Environment)")
        print("\n💡 Next: We'll build the RL environment that uses this data")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        print("\n⚠️ Please fix errors before proceeding")