"""Verify Phase 1 completion - FIXED"""
import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from utils.memory import memory_manager
from utils.logger import logger
from utils.validators import validators

def verify_phase1():
    print("\n🔍 VERIFYING PHASE 1 COMPLETION")
    print("=" * 50)
    
    checks = []
    
    # Check 1: Settings
    try:
        settings.validate()
        checks.append(("Settings", True))
        print("✅ Settings validated")
    except Exception as e:
        checks.append(("Settings", False))
        print(f"❌ Settings: {e}")
    
    # Check 2: Memory
    try:
        memory_manager.add_query("verification test")
        summary = memory_manager.get_summary()
        checks.append(("Memory", True))
        print("✅ Memory system working")
    except Exception as e:
        checks.append(("Memory", False))
        print(f"❌ Memory: {e}")
    
    # Check 3: Logger
    try:
        logger.info("Logger test")
        checks.append(("Logger", True))
        print("✅ Logger working")
    except Exception as e:
        checks.append(("Logger", False))
        print(f"❌ Logger: {e}")
    
    # Check 4: Validators
    try:
        test_papers = [{"title": "Test", "url": "http://test.com"}] * 10
        validators.validate_papers(test_papers)
        checks.append(("Validators", True))
        print("✅ Validators working")
    except Exception as e:
        checks.append(("Validators", False))
        print(f"❌ Validators: {e}")
    
    # Check 5: CrewAI
    try:
        from crewai import Agent, Task, Crew
        checks.append(("CrewAI", True))
        print("✅ CrewAI imports working")
    except Exception as e:
        checks.append(("CrewAI", False))
        print(f"❌ CrewAI: {e}")
    
    # Print results
    print("\n📋 Component Status:")
    all_passed = all(status for _, status in checks)
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 PHASE 1 COMPLETE - READY FOR PHASE 2!")
        print("=" * 50)
        print("\n✅ All systems operational:")
        print("   • Configuration system")
        print("   • Memory management")
        print("   • Logging utilities")
        print("   • Validation framework")
        print("   • CrewAI integration")
        print("\n🚀 Ready to build Agent 1: Paper Hunter!")
    else:
        print("⚠️  Some checks failed - review errors above")
    print("=" * 50)
    
    return all_passed

if __name__ == "__main__":
    success = verify_phase1()
    sys.exit(0 if success else 1)