#!/usr/bin/env python3
"""
Test runner for AgriDetect AI
Author: [Your Name]
Date: [Current Date]
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print('='*60)
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ SUCCESS")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print(f"Error: {e}")
        if e.stdout:
            print("Output:")
            print(e.stdout)
        if e.stderr:
            print("Error output:")
            print(e.stderr)
        return False

def run_tests(test_type=None, verbose=False, coverage=False):
    """Run tests based on type"""
    base_dir = Path(__file__).parent
    os.chdir(base_dir)
    
    # Base pytest command
    pytest_cmd = "python -m pytest"
    
    if verbose:
        pytest_cmd += " -v"
    
    if coverage:
        pytest_cmd += " --cov=. --cov-report=html --cov-report=term-missing"
    
    # Test specific types
    if test_type == "unit":
        pytest_cmd += " tests/test_model.py tests/test_advanced_features.py -m unit"
    elif test_type == "api":
        pytest_cmd += " tests/test_api.py -m api"
    elif test_type == "integration":
        pytest_cmd += " tests/ -m integration"
    elif test_type == "all":
        pytest_cmd += " tests/"
    else:
        pytest_cmd += " tests/"
    
    return run_command(pytest_cmd, f"Running {test_type or 'all'} tests")

def run_linting():
    """Run code linting"""
    commands = [
        ("python -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics", "Flake8 syntax check"),
        ("python -m flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics", "Flake8 style check"),
        ("python -m black --check .", "Black code formatting check"),
        ("python -m isort --check-only .", "Import sorting check")
    ]
    
    all_passed = True
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            all_passed = False
    
    return all_passed

def run_security_check():
    """Run security checks"""
    commands = [
        ("python -m bandit -r . -f json -o bandit-report.json", "Bandit security scan"),
        ("python -m safety check", "Safety dependency check")
    ]
    
    all_passed = True
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            all_passed = False
    
    return all_passed

def run_performance_tests():
    """Run performance tests"""
    return run_command(
        "python -m pytest tests/test_performance.py -v --durations=10",
        "Performance tests"
    )

def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="AgriDetect AI Test Runner")
    parser.add_argument("--type", choices=["unit", "api", "integration", "all"], 
                       help="Type of tests to run")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output")
    parser.add_argument("--coverage", "-c", action="store_true", 
                       help="Run with coverage")
    parser.add_argument("--lint", action="store_true", 
                       help="Run linting checks")
    parser.add_argument("--security", action="store_true", 
                       help="Run security checks")
    parser.add_argument("--performance", action="store_true", 
                       help="Run performance tests")
    parser.add_argument("--all", action="store_true", 
                       help="Run all checks")
    
    args = parser.parse_args()
    
    print("🌿 AgriDetect AI Test Runner")
    print("=" * 60)
    
    success = True
    
    # Run tests
    if args.all or args.type or not any([args.lint, args.security, args.performance]):
        if not run_tests(args.type, args.verbose, args.coverage):
            success = False
    
    # Run linting
    if args.all or args.lint:
        if not run_linting():
            success = False
    
    # Run security checks
    if args.all or args.security:
        if not run_security_check():
            success = False
    
    # Run performance tests
    if args.all or args.performance:
        if not run_performance_tests():
            success = False
    
    # Summary
    print("\n" + "="*60)
    if success:
        print("🎉 ALL CHECKS PASSED!")
        sys.exit(0)
    else:
        print("❌ SOME CHECKS FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    main()
