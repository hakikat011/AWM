#!/usr/bin/env python3
"""
Test runner for the AWM Quantitative Trading System.
Runs unit tests and integration tests with proper reporting.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and return the result."""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
        else:
            print(f"❌ {description} - FAILED")
        
        return result.returncode == 0
    
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False

def main():
    """Run all tests for the quantitative trading system."""
    print("🚀 AWM Quantitative Trading System - Test Suite")
    print("=" * 60)
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Test commands
    test_commands = [
        {
            "command": "python -m pytest tests/unit/test_quantitative_strategies.py -v --tb=short",
            "description": "Unit Tests - Quantitative Strategies"
        },
        {
            "command": "python -m pytest tests/unit/test_signal_generation.py -v --tb=short",
            "description": "Unit Tests - Signal Generation"
        },
        {
            "command": "python -m pytest tests/unit/test_decision_engine.py -v --tb=short",
            "description": "Unit Tests - Decision Engine"
        },
        {
            "command": "python -m pytest tests/integration/test_quantitative_trading_pipeline.py -v --tb=short",
            "description": "Integration Tests - End-to-End Pipeline"
        }
    ]
    
    # Run tests
    results = []
    for test in test_commands:
        success = run_command(test["command"], test["description"])
        results.append((test["description"], success))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    failed = 0
    
    for description, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{description:<40} {status}")
        if success:
            passed += 1
        else:
            failed += 1
    
    print(f"\n📈 Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! The quantitative trading system is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
