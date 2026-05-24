"""
Preprocessing Pipeline - Verification Script

This script verifies that all files are in place and the preprocessing
pipeline is ready to use.

Usage:
    python verify_preprocessing_setup.py
"""

import os
import sys
from pathlib import Path


def verify_files():
    """Verify all required files exist."""
    print("\n" + "="*80)
    print("PREPROCESSING PIPELINE - VERIFICATION")
    print("="*80 + "\n")
    
    required_files = {
        'Python Modules': {
            'preprocessing_pipeline.py': 'Main preprocessing module',
            'preprocessing_utils.py': 'Production utilities',
        },
        'Documentation': {
            'PREPROCESSING_README.md': 'Quick start guide',
            'PREPROCESSING_GUIDE.md': 'Detailed usage guide',
            'PREPROCESSING_DOCUMENTATION.md': 'Technical documentation',
            'PREPROCESSING_INDEX.md': 'Index and summary',
        },
        'Data Directories': {
            'data/': 'Data directory',
            'data/fraud_detection_elite_v3.csv': 'Input dataset',
            'data/raw/': 'Raw data directory',
            'data/processed/': 'Processed data directory',
        },
        'Model Directory': {
            'models/': 'Models directory',
        },
    }
    
    results = {}
    all_good = True
    
    for category, files in required_files.items():
        print(f"\n📂 {category}")
        print("-" * 80)
        
        results[category] = {}
        
        for file_path, description in files.items():
            exists = os.path.exists(file_path)
            status = "✅" if exists else "❌"
            results[category][file_path] = exists
            
            print(f"  {status} {file_path:40} ({description})")
            
            if not exists:
                all_good = False
    
    return all_good, results


def verify_dependencies():
    """Check if required Python packages are installed."""
    print("\n" + "="*80)
    print("PYTHON DEPENDENCIES")
    print("="*80 + "\n")
    
    required_packages = {
        'pandas': 'Data manipulation',
        'numpy': 'Numerical computing',
        'sklearn': 'Scikit-learn for ML',
        'imblearn': 'Imbalanced-learn for SMOTE',
        'joblib': 'Model persistence',
    }
    
    all_installed = True
    
    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"  ✅ {package:20} ({description})")
        except ImportError:
            print(f"  ❌ {package:20} ({description}) - NOT INSTALLED")
            all_installed = False
    
    return all_installed


def verify_data():
    """Check if input data file is valid."""
    print("\n" + "="*80)
    print("INPUT DATA VERIFICATION")
    print("="*80 + "\n")
    
    csv_file = 'data/fraud_detection_elite_v3.csv'
    
    if not os.path.exists(csv_file):
        print(f"  ❌ CSV file not found: {csv_file}")
        return False
    
    try:
        import pandas as pd
        df = pd.read_csv(csv_file)
        
        print(f"  ✅ File found: {csv_file}")
        print(f"  ✅ Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        # Check required columns
        required_cols = ['is_fraud', 'timestamp', 'amount', 'transaction_type']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"  ❌ Missing columns: {missing_cols}")
            return False
        else:
            print(f"  ✅ All required columns present")
            print(f"  ✅ Target variable (is_fraud) distribution:")
            print(f"     - Legitimate: {(df['is_fraud'] == 0).sum():,} ({(df['is_fraud'] == 0).mean()*100:.1f}%)")
            print(f"     - Fraud: {(df['is_fraud'] == 1).sum():,} ({(df['is_fraud'] == 1).mean()*100:.1f}%)")
            return True
    
    except Exception as e:
        print(f"  ❌ Error reading CSV: {e}")
        return False


def verify_code():
    """Check if main module can be imported."""
    print("\n" + "="*80)
    print("CODE VERIFICATION")
    print("="*80 + "\n")
    
    try:
        import preprocessing_pipeline
        print(f"  ✅ preprocessing_pipeline.py can be imported")
        
        # Check for key functions
        functions = [
            'load_data',
            'explore_data',
            'drop_id_columns',
            'extract_time_features',
            'encode_categorical_features',
            'scale_numerical_features',
            'split_train_test_data',
            'handle_class_imbalance',
            'run_preprocessing_pipeline'
        ]
        
        for func in functions:
            if hasattr(preprocessing_pipeline, func):
                print(f"    ✅ {func}()")
            else:
                print(f"    ❌ {func}() - NOT FOUND")
                return False
        
        return True
    
    except ImportError as e:
        print(f"  ❌ Cannot import preprocessing_pipeline: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    """Run all verifications."""
    
    # Check files
    files_ok, file_results = verify_files()
    
    # Check dependencies
    deps_ok = verify_dependencies()
    
    # Check data
    data_ok = verify_data()
    
    # Check code
    code_ok = verify_code()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80 + "\n")
    
    checks = {
        'Files present': files_ok,
        'Dependencies installed': deps_ok,
        'Input data valid': data_ok,
        'Code verified': code_ok,
    }
    
    all_ok = all(checks.values())
    
    for check, status in checks.items():
        symbol = "✅" if status else "❌"
        print(f"  {symbol} {check}")
    
    print("\n" + "="*80)
    
    if all_ok:
        print("\n🎉 ALL CHECKS PASSED! Ready to run preprocessing pipeline.\n")
        print("Next steps:")
        print("  1. python preprocessing_pipeline.py")
        print("  2. Check ./data/processed/ and ./models/ for outputs")
        print("  3. Review PREPROCESSING_README.md for usage\n")
        return 0
    else:
        print("\n❌ SOME CHECKS FAILED. Please fix the issues above.\n")
        print("Common fixes:")
        if not deps_ok:
            print("  • Install dependencies: pip install -r requirements.txt")
        if not data_ok:
            print("  • Ensure fraud_detection_elite_v3.csv exists in ./data/")
        if not files_ok:
            print("  • Ensure all preprocessing files are in the project root")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
