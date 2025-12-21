"""
Script to calculate feature types and categories for PRESENTATION.md
This script loads raw data and calculates exact feature types (int/string/etc) and categorical/numeric classification.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path
def get_project_root():
    """Get project root directory."""
    if '__file__' in globals():
        return Path(__file__).parent.parent.parent
    
    cwd = Path(os.getcwd())
    if (cwd / 'src').exists() and (cwd / 'data').exists():
        return cwd
    
    current = cwd
    for _ in range(5):
        if (current / 'src').exists() and (current / 'data').exists():
            return current
        current = current.parent
    
    return cwd

project_root = get_project_root()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils import get_data_dir


def calculate_feature_types():
    """Calculate exact feature types and categories."""
    print("=" * 70)
    print("CALCULATING FEATURE TYPES AND CATEGORIES")
    print("=" * 70)
    
    # Load raw data only (no preprocessing/engineering to avoid memory issues)
    data_path = get_data_dir()
    print(f"\nLoading raw data from: {data_path}")
    
    print("Loading transaction data...")
    train_trans = pd.read_csv(data_path / "train_transaction.csv", nrows=10000)  # Sample for speed
    print(f"Train transaction shape (sample): {train_trans.shape}")
    
    print("Loading identity data...")
    train_id = pd.read_csv(data_path / "train_identity.csv", nrows=10000)  # Sample for speed
    print(f"Train identity shape (sample): {train_id.shape}")
    
    # Merge
    print("Merging transaction and identity data...")
    train_df = train_trans.merge(train_id, on='TransactionID', how='left')
    print(f"Merged shape (sample): {train_df.shape}")
    
    print("\n" + "=" * 70)
    print("FEATURE TYPES AND CATEGORIES")
    print("=" * 70)
    
    # Define feature groups
    feature_groups = {
        'Card Features': ['card1', 'card2', 'card3', 'card4', 'card5', 'card6'],
        'Address Features': ['addr1', 'addr2'],
        'Email Features': ['P_emaildomain', 'R_emaildomain'],
        'IP/Distance Features': ['dist1', 'dist2'],
        'Card-Related Features': [f'C{i}' for i in range(1, 15)],
        'Device Features': ['DeviceType', 'DeviceInfo'],
        'Identity Features': ['id_28', 'id_29', 'id_30', 'id_31'],
        'Basic Features': ['TransactionAmt', 'TransactionDT', 'ProductCD']
    }
    
    results = {}
    
    for group_name, features in feature_groups.items():
        print(f"\n### {group_name}")
        print("-" * 70)
        
        for feat in features:
            # Check both original and with _isMissing suffix
            found = False
            for col in train_df.columns:
                if col == feat or col.startswith(feat + '_'):
                    if col == feat:  # Original feature
                        dtype = str(train_df[col].dtype)
                        nunique = train_df[col].nunique()
                        
                        # Improved classification logic
                        is_numeric_dtype = pd.api.types.is_numeric_dtype(train_df[col])
                        total_rows = len(train_df)
                        unique_ratio = nunique / total_rows if total_rows > 0 else 0
                        
                        # Determine classification
                        if train_df[col].dtype == 'object' or train_df[col].dtype.name == 'category':
                            is_categorical = True
                            classification = 'Categorical'
                        elif nunique == 2:
                            is_categorical = True
                            classification = 'Categorical (Boolean)'
                        elif is_numeric_dtype:
                            # Integer types: prioritize categorical (likely ID/code)
                            if 'int' in dtype:
                                # Integer with reasonable cardinality → Categorical (ID/code)
                                # Examples: card1 (ID), TransactionDT (timestamp - but check ratio)
                                if unique_ratio > 0.9:  # Very high ratio (>90%) → Likely timestamp/continuous
                                    is_categorical = False
                                    classification = 'Numeric'
                                elif nunique < 10000:  # IDs/codes typically have < 10k unique values
                                    is_categorical = True
                                    classification = 'Categorical'
                                else:
                                    is_categorical = False
                                    classification = 'Numeric'
                            
                            # Float types: check cardinality and ratio
                            elif 'float' in dtype:
                                # Very low cardinality → Categorical
                                if nunique < 50:
                                    is_categorical = True
                                    classification = 'Categorical'
                                # Very high cardinality with high ratio → Numeric
                                elif nunique > 1000 and unique_ratio > 0.5:
                                    is_categorical = False
                                    classification = 'Numeric'
                                # Medium cardinality: use unique ratio
                                else:
                                    if unique_ratio < 0.1:  # Less than 10% unique → Likely categorical
                                        is_categorical = True
                                        classification = 'Categorical'
                                    elif unique_ratio > 0.5:  # More than 50% unique → Likely numeric
                                        is_categorical = False
                                        classification = 'Numeric'
                                    else:  # 10-50% unique → Check if discrete
                                        if unique_ratio < 0.3:  # Low ratio → Categorical (discrete codes)
                                            is_categorical = True
                                            classification = 'Categorical'
                                        else:
                                            is_categorical = False
                                            classification = 'Numeric'
                            else:
                                is_categorical = False
                                classification = 'Numeric'
                        else:
                            is_categorical = False
                            classification = 'Other'
                        
                        results[feat] = {
                            'dtype': dtype,
                            'nunique': nunique,
                            'unique_ratio': unique_ratio,
                            'classification': classification
                        }
                        
                        print(f"{feat}:")
                        print(f"  Type: {dtype}")
                        print(f"  Classification: {classification}")
                        print(f"  Unique values: {nunique:,}")
                        print(f"  Unique ratio: {unique_ratio:.2%}")
                        found = True
                        break
            
            if not found:
                print(f"{feat}: NOT FOUND in data")
    
    # Also check some engineered features
    print("\n### Engineered Features (Sample)")
    print("-" * 70)
    
    engineered_features = [
        'card1_count', 'card2_freq', 'TransactionAmt_log', 
        'uid_1_fraud_rate', 'uid_2_fraud_rate', 'uid_3_fraud_rate',
        'hour', 'day_of_week', 'is_weekend', 'is_night'
    ]
    
    for feat in engineered_features:
        if feat in train_df.columns:
            dtype = str(train_df[feat].dtype)
            nunique = train_df[feat].nunique()
            
            is_numeric = pd.api.types.is_numeric_dtype(train_df[feat])
            is_categorical = False
            
            if train_df[feat].dtype == 'object' or train_df[feat].dtype.name == 'category':
                is_categorical = True
            elif is_numeric and nunique < 50:
                is_categorical = True
            
            classification = 'Categorical' if is_categorical else ('Numeric' if is_numeric else 'Other')
            
            print(f"{feat}:")
            print(f"  Type: {dtype}")
            print(f"  Classification: {classification}")
            print(f"  Unique values: {nunique:,}")
    
    print("\n" + "=" * 70)
    
    return results


if __name__ == "__main__":
    results = calculate_feature_types()
    
    # Save results to a file for easy reference
    # Handle both script execution and IPython/Jupyter
    try:
        script_dir = Path(__file__).parent
    except NameError:
        # Running in IPython/Jupyter - use current working directory
        script_dir = Path.cwd() / "src" / "eighth_view"
    
    output_file = script_dir / "feature_types_stats.txt"
    with open(output_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("FEATURE TYPES AND CATEGORIES\n")
        f.write("=" * 70 + "\n\n")
        
        for feat, info in results.items():
            f.write(f"{feat}:\n")
            f.write(f"  Type: {info['dtype']}\n")
            f.write(f"  Classification: {info['classification']}\n")
            f.write(f"  Unique values: {info['nunique']:,}\n\n")
    
    print(f"\nResults saved to: {output_file}")

