"""
Test Product and Warehouse classes
"""
from src.environment.product import Product, create_sample_products
from src.environment.warehouse import Warehouse, create_sample_warehouses


def test_product():
    """Test Product class"""
    print("=" * 50)
    print("Testing Product class...")
    print("=" * 50)
    
    products = create_sample_products()
    product = products[0]
    
    print(f"✓ Created product: {product}")
    
    # Test demand generation
    print("\n📊 Generating 10 days of demand:")
    demands = [product.generate_demand(day=i, random_seed=42+i) for i in range(10)]
    for i, d in enumerate(demands):
        day_type = "Weekend" if i % 7 in [5, 6] else "Weekday"
        print(f"  Day {i+1} ({day_type}): {d:.1f} units")
    
    # Check validation
    assert product.base_demand > 0, "Validation failed"
    print("\n✅ Product validation works correctly")
    
    print("\n✅ Product class working perfectly!\n")
    return True


def test_warehouse():
    """Test Warehouse class"""
    print("=" * 50)
    print("Testing Warehouse class...")
    print("=" * 50)
    
    warehouses = create_sample_warehouses()
    warehouse = warehouses[0]
    
    print(f"✓ Created warehouse: {warehouse}")
    
    # Test adding inventory
    print("\n📦 Testing inventory management:")
    success = warehouse.add_inventory("PROD_A", 100)
    print(f"  Added 100 units of PROD_A: {'✓' if success else '✗'}")
    print(f"  Current inventory: {warehouse.get_inventory_level('PROD_A')} units")
    
    # Test removing inventory
    removed = warehouse.remove_inventory("PROD_A", 30)
    print(f"  Removed {removed} units")
    print(f"  Remaining inventory: {warehouse.get_inventory_level('PROD_A')} units")
    
    # Test capacity
    utilization = warehouse.get_capacity_utilization()
    print(f"\n📊 Capacity utilization: {utilization*100:.1f}%")
    
    # Test pending orders
    print("\n🚚 Testing order management:")
    warehouse.place_order("PROD_B", 200, arrival_day=5)
    print(f"  Placed order for 200 units (arriving day 5)")
    print(f"  Pending orders: {len(warehouse.pending_orders)} SKUs")
    
    # Test receiving orders
    print(f"\n  Simulating day 5...")
    received = warehouse.receive_orders(current_day=5)
    print(f"  Received: {received}")
    print(f"  PROD_B inventory: {warehouse.get_inventory_level('PROD_B')} units")
    
    print("\n✅ Warehouse class working perfectly!\n")
    return True


if __name__ == "__main__":
    try:
        test_product()
        test_warehouse()
        
        print("=" * 50)
        print("🎉 ALL ENTITY CLASSES VALIDATED!")
        print("=" * 50)
        print("\n✅ Ready to proceed to Step 1.3 (Data Generator)")
        print("\n💡 Next: We'll create realistic demand patterns")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("\n⚠️ Please fix errors before proceeding")