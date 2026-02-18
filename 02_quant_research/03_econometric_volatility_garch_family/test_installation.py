"""
Installation and functionality test script.

Run this to verify all components are working correctly.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("  [OK] numpy")
    except ImportError as e:
        print(f"  [FAIL] numpy: {e}")
        return False
    
    try:
        import pandas as pd
        print("  [OK] pandas")
    except ImportError as e:
        print(f"  [FAIL] pandas: {e}")
        return False
    
    try:
        import scipy
        print("  [OK] scipy")
    except ImportError as e:
        print(f"  [FAIL] scipy: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("  [OK] matplotlib")
    except ImportError as e:
        print(f"  [FAIL] matplotlib: {e}")
        return False
    
    try:
        import seaborn as sns
        print("  [OK] seaborn")
    except ImportError as e:
        print(f"  [FAIL] seaborn: {e}")
        return False
    
    try:
        import statsmodels
        print("  [OK] statsmodels")
    except ImportError as e:
        print(f"  [FAIL] statsmodels: {e}")
        return False
    
    try:
        import yfinance as yf
        print("  [OK] yfinance")
    except ImportError as e:
        print(f"  [FAIL] yfinance: {e}")
        return False
    
    return True


def test_modules():
    """Test that project modules can be imported."""
    print("\nTesting project modules...")
    
    try:
        from src.data_loader import DataLoader
        print("  [OK] data_loader")
    except ImportError as e:
        print(f"  [FAIL] data_loader: {e}")
        return False
    
    try:
        from src.returns import StylizedFacts
        print("  [OK] returns")
    except ImportError as e:
        print(f"  [FAIL] returns: {e}")
        return False
    
    try:
        from src.diagnostics import ModelDiagnostics
        print("  [OK] diagnostics")
    except ImportError as e:
        print(f"  [FAIL] diagnostics: {e}")
        return False
    
    try:
        from src.models import GARCHModel, EGARCHModel, GJRGARCHModel, HARCHModel
        print("  [OK] models")
    except ImportError as e:
        print(f"  [FAIL] models: {e}")
        return False
    
    try:
        from src.forecasting import RollingForecast, ForecastMetrics
        print("  [OK] forecasting")
    except ImportError as e:
        print(f"  [FAIL] forecasting: {e}")
        return False
    
    try:
        from src.risk import VaRCalculator, ESCalculator, VaRBacktest
        print("  [OK] risk")
    except ImportError as e:
        print(f"  [FAIL] risk: {e}")
        return False
    
    try:
        from src.rough_vol import RoughBergomiModel, RoughVolBenchmark
        print("  [OK] rough_vol")
    except ImportError as e:
        print(f"  [FAIL] rough_vol: {e}")
        return False
    
    return True


def test_basic_functionality():
    """Test basic functionality of key components."""
    print("\nTesting basic functionality...")
    
    import numpy as np
    from src.models import GARCHModel
    from src.returns import StylizedFacts
    from src.risk import VaRCalculator
    
    # Generate synthetic data
    np.random.seed(42)
    returns = np.random.randn(500) * 0.01
    
    # Test stylized facts
    try:
        sf = StylizedFacts(returns)
        stats = sf.summary_statistics()
        assert 'mean' in stats
        assert 'std' in stats
        print("  [OK] Stylized facts analysis")
    except Exception as e:
        print(f"  [FAIL] Stylized facts: {e}")
        return False
    
    # Test GARCH model
    try:
        model = GARCHModel(p=1, q=1)
        model.fit(returns, verbose=False)
        assert model.params is not None
        assert model.log_likelihood is not None
        print("  [OK] GARCH model estimation")
    except Exception as e:
        print(f"  [FAIL] GARCH model: {e}")
        return False
    
    # Test forecasting
    try:
        forecast = model.forecast(returns, horizon=1)
        assert len(forecast) == 1
        assert forecast[0] > 0
        print("  [OK] Volatility forecasting")
    except Exception as e:
        print(f"  [FAIL] Forecasting: {e}")
        return False
    
    # Test VaR calculation
    try:
        var_calc = VaRCalculator(confidence_level=0.95)
        var = var_calc.parametric_var(0.01)
        assert var > 0
        print("  [OK] VaR calculation")
    except Exception as e:
        print(f"  [FAIL] VaR calculation: {e}")
        return False
    
    # Test rough volatility
    try:
        from src.rough_vol import FractionalBrownianMotion
        fbm = FractionalBrownianMotion(hurst=0.1, seed=42)
        path = fbm.simulate(100, dt=1.0)
        assert len(path) == 101
        print("  [OK] Rough volatility simulation")
    except Exception as e:
        print(f"  [FAIL] Rough volatility: {e}")
        return False
    
    return True


def test_data_download():
    """Test data download functionality (requires internet)."""
    print("\nTesting data download (requires internet)...")
    
    try:
        from src.data_loader import DataLoader
        loader = DataLoader()
        
        # Try to download a small amount of data
        returns, prices = loader.prepare_dataset(
            ticker="^GSPC",
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        
        assert len(returns) > 0
        print(f"  [OK] Downloaded {len(returns)} returns")
        return True
        
    except Exception as e:
        print(f"  [WARNING] Data download failed (may be offline): {e}")
        return None


def main():
    """Run all tests."""
    print("="*60)
    print("INSTALLATION TEST")
    print("="*60)
    
    results = []
    
    # Test imports
    results.append(("Package imports", test_imports()))
    
    # Test modules
    results.append(("Project modules", test_modules()))
    
    # Test functionality
    results.append(("Basic functionality", test_basic_functionality()))
    
    # Test data download (optional)
    download_result = test_data_download()
    if download_result is not None:
        results.append(("Data download", download_result))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{test_name:30s}: {status}")
        if not result:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n[OK] All tests passed! Installation is working correctly.")
        print("\nNext steps:")
        print("  1. Run 'python main.py' for full pipeline")
        print("  2. Open notebooks with 'jupyter notebook'")
        print("  3. Read INTERVIEW_DEFENSE.md for interview prep")
        return 0
    else:
        print("\n[FAIL] Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("  1. Install missing packages: pip install -r requirements.txt")
        print("  2. Check Python version (3.8+ recommended)")
        print("  3. Verify you're in the project root directory")
        return 1


if __name__ == "__main__":
    sys.exit(main())
