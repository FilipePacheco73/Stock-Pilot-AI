"""
Master test runner for StockPilot
Run all validations from this script
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from tests.test_quick import test_quick


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("STOCKPILOT TEST SUITE")
    print("="*60)
    
    print("\n🧪 Running all tests...\n")
    
    tests = [
        ("Quick Validation", test_quick),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"[{len([t for t in tests[:tests.index((test_name, test_func))]]) + 1}/{len(tests)}] {test_name}")
            print("-" * 60)
            test_func()
            passed += 1
            
        except Exception as e:
            print(f"\n❌ TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"📊 TEST RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("\n✅ ALL TESTS PASSED!")
        print("\nNext steps:")
        print("  1. python main.py          # Run full training pipeline")
        print("  2. streamlit run dashboard/app.py  # Launch dashboard")
        print("\n")
    else:
        print(f"\n❌ {failed} test(s) failed!")
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
