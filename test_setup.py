#!/usr/bin/env python3
"""
Quick test script to verify project setup.
Run this to check if everything is working.
"""

import sys
import traceback

def test_imports():
    """Test if all required packages can be imported."""
    print("=" * 60)
    print("Testing Package Imports")
    print("=" * 60)
    
    packages = {
        'pandas': 'pd',
        'numpy': 'np',
        'sklearn': 'sklearn',
        'xgboost': 'xgb',
        'lightgbm': 'lgb',
        'catboost': 'cb',
        'shap': 'shap',
        'matplotlib': 'plt',
        'seaborn': 'sns',
    }
    
    results = {}
    for package, alias in packages.items():
        try:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'unknown')
            results[package] = ('✅', version)
        except ImportError as e:
            results[package] = ('❌', str(e))
    
    for package, (status, info) in results.items():
        print(f"{status} {package:15s} {info}")
    
    return all(status == '✅' for status, _ in results.values())

def test_project_modules():
    """Test if project modules can be imported."""
    print("\n" + "=" * 60)
    print("Testing Project Modules")
    print("=" * 60)
    
    modules = [
        'src.data_loader',
        'src.preprocessing',
        'src.feature_engineering',
        'src.models',
        'src.evaluation',
        'src.utils',
    ]
    
    results = {}
    for module in modules:
        try:
            __import__(module)
            results[module] = ('✅', 'OK')
        except Exception as e:
            results[module] = ('❌', str(e))
    
    for module, (status, info) in results.items():
        print(f"{status} {module:30s} {info}")
    
    return all(status == '✅' for status, _ in results.values())

def test_data_loading():
    """Test if data can be loaded."""
    print("\n" + "=" * 60)
    print("Testing Data Loading")
    print("=" * 60)
    
    try:
        from src.data_loader import load_data
        train_df, test_df = load_data(data_dir='data', sample_size=100, use_identity=True)
        print(f"✅ Data loaded successfully!")
        print(f"   Training shape: {train_df.shape}")
        print(f"   Test shape: {test_df.shape}")
        print(f"   Fraud rate: {train_df['isFraud'].mean():.4f}")
        return True
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print(f"\nPython Version: {sys.version}")
    print(f"Python Executable: {sys.executable}\n")
    
    test1 = test_imports()
    test2 = test_project_modules()
    test3 = test_data_loading()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    if test1 and test2 and test3:
        print("✅ All tests passed! Project is ready to use.")
        print("\nYou can now:")
        print("  1. Run notebooks: jupyter notebook notebooks/01_eda.ipynb")
        print("  2. Start feature engineering")
        print("  3. Train models")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        if not test1:
            print("\n💡 Tip: Make sure you're using the correct Python environment.")
            print("   If using Anaconda, activate it or use: /opt/anaconda3/bin/python3")
        if not test2:
            print("\n💡 Tip: Make sure you're in the project root directory.")
        if not test3:
            print("\n💡 Tip: Make sure data files are in the 'data/' directory.")

if __name__ == "__main__":
    main()









