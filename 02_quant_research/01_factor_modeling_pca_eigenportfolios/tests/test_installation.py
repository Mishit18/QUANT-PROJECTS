"""Installation and smoke tests for the PCA factor modeling project."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"


def test_required_packages_import() -> None:
    """All runtime dependencies should be importable."""

    required_packages = [
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "statsmodels",
        "yfinance",
        "matplotlib",
        "seaborn",
        "yaml",
    ]

    missing = []
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing.append(package)

    assert not missing, f"Missing packages: {', '.join(missing)}"


def test_required_directories_exist() -> None:
    """The repository should include the expected project layout."""

    required_dirs = [
        "config",
        "src",
        "analysis",
        "data",
        "results",
        "plots",
        "reports",
    ]

    missing = [
        directory
        for directory in required_dirs
        if not (PROJECT_ROOT / directory).exists()
    ]

    assert not missing, f"Missing directories: {', '.join(missing)}"


def test_config_file_is_valid() -> None:
    """The default config should parse and expose all required sections."""

    config_path = PROJECT_ROOT / "config" / "config.yaml"
    assert config_path.exists(), "config/config.yaml not found"

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    required_sections = [
        "data",
        "pca",
        "factors",
        "regression",
        "regime",
        "portfolio",
        "analysis",
        "paths",
    ]
    missing = [section for section in required_sections if section not in config]

    assert not missing, f"Missing config sections: {', '.join(missing)}"


def test_custom_modules_import() -> None:
    """Core project modules should import without side effects."""

    sys.path.insert(0, str(SRC_DIR))
    modules = [
        "utils",
        "data_pipeline",
        "pca_model",
        "factor_construction",
        "regression",
        "regime_analysis",
        "portfolio_controls",
        "visualization",
    ]

    failed = []
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception as exc:  # pragma: no cover - message is the value here.
            failed.append(f"{module}: {exc}")

    assert not failed, "Module import failures: " + "; ".join(failed)


def main() -> int:
    """Run the same checks without requiring pytest."""

    checks = [
        ("package imports", test_required_packages_import),
        ("directory layout", test_required_directories_exist),
        ("configuration", test_config_file_is_valid),
        ("custom modules", test_custom_modules_import),
    ]

    print("=" * 72)
    print("PCA FACTOR MODELING INSTALLATION CHECK")
    print("=" * 72)

    failures = []
    for label, check in checks:
        try:
            check()
            print(f"PASS: {label}")
        except Exception as exc:
            print(f"FAIL: {label} - {exc}")
            failures.append(label)

    print("=" * 72)
    if failures:
        print("Fix the failed checks, then run: python analysis/run_full_pipeline.py")
        return 1

    print("All checks passed. Next: python analysis/run_full_pipeline.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
