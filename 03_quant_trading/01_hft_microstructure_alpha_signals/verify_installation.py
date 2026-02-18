"""
Installation verification script.

Checks that all dependencies are installed and modules can be imported.
"""

import sys
from pathlib import Path


def check_dependencies():
    """Check if all required packages are installed."""
    print("Checking dependencies...")
    
    required_packages = [
        'numpy',
        'pandas',
        'scipy',
        'sklearn',
        'xgboost',
        'matplotlib',
        'seaborn',
        'yaml',
        'joblib',
        'tqdm'
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            elif package == 'yaml':
                __import__('yaml')
            else:
                __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\nAll dependencies installed!")
    return True


def check_project_structure():
    """Check if all required directories and files exist."""
    print("\nChecking project structure...")
    
    required_paths = [
        'config/data_config.yaml',
        'config/feature_config.yaml',
        'config/model_config.yaml',
        'config/backtest_config.yaml',
        'src/__init__.py',
        'src/data/loader.py',
        'src/features/ofi.py',
        'src/models/tree_models.py',
        'src/backtest/event_simulator.py',
        'run_pipeline.py',
        'README.md',
        'DOCUMENTATION.md'
    ]
    
    missing = []
    
    for path_str in required_paths:
        path = Path(path_str)
        if path.exists():
            print(f"  ✓ {path_str}")
        else:
            print(f"  ✗ {path_str} - MISSING")
            missing.append(path_str)
    
    if missing:
        print(f"\nMissing files: {', '.join(missing)}")
        return False
    
    print("\nAll required files present!")
    return True


def check_imports():
    """Check if project modules can be imported."""
    print("\nChecking module imports...")
    
    modules = [
        'src.data.loader',
        'src.features.ofi',
        'src.features.queue_imbalance',
        'src.labels.future_ticks',
        'src.models.baseline',
        'src.models.tree_models',
        'src.backtest.event_simulator',
        'src.analysis.alpha_decay'
    ]
    
    failed = []
    
    for module in modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except Exception as e:
            print(f"  ✗ {module} - ERROR: {str(e)[:50]}")
            failed.append(module)
    
    if failed:
        print(f"\nFailed imports: {', '.join(failed)}")
        return False
    
    print("\nAll modules import successfully!")
    return True


def main():
    """Run all verification checks."""
    print("="*60)
    print("HFT ALPHA RESEARCH PROJECT - INSTALLATION VERIFICATION")
    print("="*60)
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Project Structure", check_project_structure),
        ("Module Imports", check_imports)
    ]
    
    results = []
    
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\nError in {name}: {e}")
            results.append((name, False))
    
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:20s}: {status}")
        if not result:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n✓ All checks passed! You're ready to run the pipeline.")
        print("\nNext steps:")
        print("  1. Review README.md for project overview")
        print("  2. Check DOCUMENTATION.md for technical details")
        print("  3. Run: python run_pipeline.py")
        return 0
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
