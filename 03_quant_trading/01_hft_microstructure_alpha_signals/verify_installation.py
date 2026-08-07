"""Installation and repository verification for the HFT microstructure project."""

from __future__ import annotations

import sys
from pathlib import Path


def check_dependencies() -> bool:
    print("Checking dependencies...")
    required_packages = [
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "xgboost",
        "matplotlib",
        "seaborn",
        "yaml",
        "joblib",
        "tqdm",
    ]
    missing: list[str] = []
    for package in required_packages:
        try:
            __import__("sklearn" if package == "sklearn" else "yaml" if package == "yaml" else package)
            print(f"  [OK] {package}")
        except ImportError:
            print(f"  [MISSING] {package}")
            missing.append(package)

    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    print("\nAll dependencies installed.")
    return True


def check_project_structure() -> bool:
    print("\nChecking project structure...")
    required_paths = [
        "config/data_config.yaml",
        "config/feature_config.yaml",
        "config/model_config.yaml",
        "config/backtest_config.yaml",
        "src/__init__.py",
        "src/data/loader.py",
        "src/features/ofi.py",
        "src/models/tree_models.py",
        "src/backtest/event_simulator.py",
        "src/backtest/ev_execution.py",
        "src/analysis/alpha_decay.py",
        "run_pipeline.py",
        "README.md",
        "reports/summary_report.md",
    ]
    missing: list[str] = []
    for path_str in required_paths:
        if Path(path_str).exists():
            print(f"  [OK] {path_str}")
        else:
            print(f"  [MISSING] {path_str}")
            missing.append(path_str)

    if missing:
        print(f"\nMissing files: {', '.join(missing)}")
        return False
    print("\nAll required files present.")
    return True


def check_imports() -> bool:
    print("\nChecking module imports...")
    modules = [
        "src.data.loader",
        "src.features.ofi",
        "src.features.queue_imbalance",
        "src.labels.future_ticks",
        "src.models.baseline",
        "src.models.tree_models",
        "src.backtest.event_simulator",
        "src.backtest.ev_execution",
        "src.analysis.alpha_decay",
    ]
    failed: list[str] = []
    for module in modules:
        try:
            __import__(module)
            print(f"  [OK] {module}")
        except Exception as exc:
            print(f"  [FAILED] {module}: {str(exc)[:80]}")
            failed.append(module)

    if failed:
        print(f"\nFailed imports: {', '.join(failed)}")
        return False
    print("\nAll modules import successfully.")
    return True


def main() -> int:
    print("=" * 60)
    print("HFT ALPHA RESEARCH PROJECT - INSTALLATION VERIFICATION")
    print("=" * 60)

    checks = [
        ("Dependencies", check_dependencies),
        ("Project Structure", check_project_structure),
        ("Module Imports", check_imports),
    ]
    results = []
    for name, check_func in checks:
        try:
            results.append((name, check_func()))
        except Exception as exc:
            print(f"\nError in {name}: {exc}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{name:20s}: {status}")
        all_passed = all_passed and result
    print("=" * 60)

    if all_passed:
        print("\nAll checks passed. Next run: python run_pipeline.py")
        return 0
    print("\nSome checks failed. Fix the issues above before running the pipeline.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
