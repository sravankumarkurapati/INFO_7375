"""
Test memory system - FIXED with proper paths
"""
import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.memory import MemoryManager
from utils.logger import logger

def test_memory_system():
    """Test memory operations"""
    
    print("\n🧪 Testing Memory System...")
    print("=" * 50)
    
    # Create memory manager
    memory = MemoryManager()
    
    # Test short-term memory
    print("\n📝 Testing short-term memory...")
    memory.add_query("test query about AI")
    memory.add_agent_output("test_agent", {"result": "test output"})
    memory.update_metric("test_metric", 0.85)
    
    print("✅ Short-term operations successful")
    
    # Test long-term memory
    print("\n💾 Testing long-term memory...")
    memory.add_successful_search("AI research", 15)
    memory.update_domain_knowledge("transformer", 3)
    memory.add_quality_score(8.5)
    
    print("✅ Long-term operations successful")
    
    # Get summary
    summary = memory.get_summary()
    print("\n📊 Memory Summary:")
    print(f"  Current session queries: {summary['current_session']['queries']}")
    print(f"  Total lifetime queries: {summary['long_term']['total_queries']}")
    print(f"  Average quality: {summary['long_term']['average_quality']:.2f}")
    
    # Export session
    export_path = memory.export_session()
    print(f"\n💾 Session exported to: {export_path}")
    
    # Save long-term memory
    memory.save_long_term()
    print("✅ Long-term memory saved")
    
    print("\n" + "=" * 50)
    print("✅ MEMORY SYSTEM TEST SUCCESSFUL!")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    try:
        test_memory_system()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)