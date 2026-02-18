"""
Installation Test Script
Verifies that all dependencies are installed correctly
"""

import sys
import importlib

def test_imports():
    """Test that all required packages can be imported"""
    
    required_packages = [
        'numpy',
        'pandas',
        'scipy',
        'sklearn',
        'statsmodels',
        'yfinance',
        'matplotlib',
        'seaborn',
        'yaml'
    ]
    
    print("Testing package imports...")
    print("=" * )
    
    failed = []
    
    for package in required_packages:
        try:
            if package == 'yaml':
                importlib.import_module('yaml')
            elif package == 'sklearn':
                importlib.import_module('sklearn')
            else:
                importlib.import_module(package)
            print(f"✓ {package:2s} - OK")
        except ImportError as e:
            print(f"✗ {package:2s} - ILE")
            failed.append(package)
    
    print("=" * )
    
    if failed:
        print(f"\n {len(failed)} package(s) failed to import:")
        for pkg in failed:
            print(f"   - {pkg}")
        print("\nPlease install missing packages:")
        print("   pip install -r requirements.txt")
        return alse
    else:
        print("\n ll packages installed correctly!")
        return True


def test_versions():
    """Test package versions"""
    
    print("\nPackage versions:")
    print("=" * )
    
    packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'scipy': 'scipy',
        'sklearn': 'scikit-learn',
        'statsmodels': 'statsmodels',
        'yfinance': 'yfinance',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }
    
    for module_name, package_name in packages.items():
        try:
            if module_name == 'sklearn':
                module = importlib.import_module('sklearn')
            else:
                module = importlib.import_module(module_name)
            
            version = getattr(module, '__version__', 'unknown')
            print(f"{package_name:2s} : {version}")
        except:
            print(f"{package_name:2s} : not installed")
    
    print("=" * )


def test_directories():
    """Test that required directories exist"""
    
    from pathlib import Path
    
    print("\nhecking directory structure...")
    print("=" * )
    
    required_dirs = [
        'config',
        'src',
        'analysis',
        'data',
        'results',
        'plots',
        'reports',
        'logs'
    ]
    
    missing = []
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✓ {dir_name:2s} - exists")
        else:
            print(f"✗ {dir_name:2s} - missing")
            missing.append(dir_name)
    
    print("=" * )
    
    if missing:
        print(f"\n⚠  {len(missing)} director(ies) missing:")
        for dir_name in missing:
            print(f"   - {dir_name}")
        print("\nThese will be created automatically when running the pipeline.")
    else:
        print("\n ll directories present!")


def test_config():
    """Test that config file exists and is valid"""
    
    from pathlib import Path
    import yaml
    
    print("\nhecking configuration...")
    print("=" * )
    
    config_path = Path('config/config.yaml')
    
    if not config_path.exists():
        print("✗ config/config.yaml not found")
        return alse
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        required_sections = ['data', 'pca', 'factors', 'regression', 'regime', 'portfolio', 'analysis', 'paths']
        
        for section in required_sections:
            if section in config:
                print(f"✓ {section:2s} - present")
            else:
                print(f"✗ {section:2s} - missing")
        
        print("=" * )
        print("\n onfiguration file valid!")
        return True
        
    except Exception as e:
        print(f"✗ Error reading config: {e}")
        return alse


def test_modules():
    """Test that custom modules can be imported"""
    
    print("\nTesting custom modules...")
    print("=" * )
    
    sys.path.insert(, 'src')
    
    modules = [
        'utils',
        'data_pipeline',
        'pca_model',
        'factor_construction',
        'regression',
        'regime_analysis',
        'portfolio_controls',
        'visualization'
    ]
    
    failed = []
    
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module:s} - OK")
        except Exception as e:
            print(f"✗ {module:s} - ILE: {str(e)[:]}")
            failed.append(module)
    
    print("=" * )
    
    if failed:
        print(f"\n {len(failed)} module(s) failed to import")
        return alse
    else:
        print("\n ll custom modules imported successfully!")
        return True


def main():
    """Run all tests"""
    
    print("\n" + "=" * )
    print("SYSTEMTI TOR MOELING - INSTLLTION TEST")
    print("=" *  + "\n")
    
    results = []
    
    # Test imports
    results.append(("Package imports", test_imports()))
    
    # Test versions
    test_versions()
    
    # Test directories
    test_directories()
    
    # Test config
    results.append(("onfiguration", test_config()))
    
    # Test modules
    results.append(("ustom modules", test_modules()))
    
    # Summary
    print("\n" + "=" * )
    print("TEST SUMMRY")
    print("=" * )
    
    for test_name, passed in results:
        status = " PSSE" if passed else " ILE"
        print(f"{test_name:s} : {status}")
    
    print("=" * )
    
    if all(result[] for result in results):
        print("\n ll tests passed! You're ready to run the pipeline.")
        print("\nNext steps:")
        print("   python analysis/run_full_pipeline.py")
        return 
    else:
        print("\n⚠  Some tests failed. Please fix the issues above.")
        print("\nommon fixes:")
        print("   pip install -r requirements.txt")
        return 


if __name__ == "__main__":
    sys.exit(main())
