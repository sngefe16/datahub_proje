"""
Script to calculate exact missing data statistics for PRESENTATION.md
This script loads raw data and calculates exact missing value counts and percentages.
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


def calculate_missing_stats():
    """Calculate exact missing data statistics."""
    print("=" * 70)
    print("CALCULATING EXACT MISSING DATA STATISTICS")
    print("=" * 70)
    
    # Load data
    data_path = get_data_dir()
    print(f"\nLoading data from: {data_path}")
    
    print("\nLoading transaction data...")
    train_trans = pd.read_csv(data_path / "train_transaction.csv")
    print(f"Train transaction shape: {train_trans.shape}")
    
    print("Loading identity data...")
    train_id = pd.read_csv(data_path / "train_identity.csv")
    print(f"Train identity shape: {train_id.shape}")
    
    # Merge
    print("\nMerging transaction and identity data...")
    train_df = train_trans.merge(train_id, on='TransactionID', how='left')
    print(f"Merged shape: {train_df.shape}")
    
    total_rows = len(train_df)
    print(f"\nTotal rows: {total_rows:,}")
    
    print("\n" + "=" * 70)
    print("EXACT MISSING DATA STATISTICS")
    print("=" * 70)
    
    results = {}
    
    # === Card Features ===
    print("\n### Card Features (card1-card6)")
    print("-" * 70)
    card_features = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6']
    for card_col in card_features:
        if card_col in train_df.columns:
            missing_count = train_df[card_col].isna().sum()
            missing_pct = (missing_count / total_rows) * 100
            results[card_col] = {
                'missing_count': missing_count,
                'total_rows': total_rows,
                'missing_pct': missing_pct
            }
            print(f"{card_col}:")
            print(f"  Missing: {missing_count:,} / {total_rows:,} ({missing_pct:.2f}%)")
        else:
            print(f"{card_col}: NOT FOUND in data")
    
    # === Address Features ===
    print("\n### Address Features (addr1, addr2)")
    print("-" * 70)
    addr_features = ['addr1', 'addr2']
    for addr_col in addr_features:
        if addr_col in train_df.columns:
            missing_count = train_df[addr_col].isna().sum()
            missing_pct = (missing_count / total_rows) * 100
            results[addr_col] = {
                'missing_count': missing_count,
                'total_rows': total_rows,
                'missing_pct': missing_pct
            }
            print(f"{addr_col}:")
            print(f"  Missing: {missing_count:,} / {total_rows:,} ({missing_pct:.2f}%)")
        else:
            print(f"{addr_col}: NOT FOUND in data")
    
    # === Email Features ===
    print("\n### Email Features")
    print("-" * 70)
    email_features = ['P_emaildomain', 'R_emaildomain']
    for email_col in email_features:
        if email_col in train_df.columns:
            missing_count = train_df[email_col].isna().sum()
            missing_pct = (missing_count / total_rows) * 100
            results[email_col] = {
                'missing_count': missing_count,
                'total_rows': total_rows,
                'missing_pct': missing_pct
            }
            print(f"{email_col}:")
            print(f"  Missing: {missing_count:,} / {total_rows:,} ({missing_pct:.2f}%)")
        else:
            print(f"{email_col}: NOT FOUND in data")
    
    # === IP/Distance Features ===
    print("\n### IP/Distance Features (dist1, dist2)")
    print("-" * 70)
    dist_features = ['dist1', 'dist2']
    for dist_col in dist_features:
        if dist_col in train_df.columns:
            missing_count = train_df[dist_col].isna().sum()
            missing_pct = (missing_count / total_rows) * 100
            results[dist_col] = {
                'missing_count': missing_count,
                'total_rows': total_rows,
                'missing_pct': missing_pct
            }
            print(f"{dist_col}:")
            print(f"  Missing: {missing_count:,} / {total_rows:,} ({missing_pct:.2f}%)")
        else:
            print(f"{dist_col}: NOT FOUND in data")
    
    # === Card-Related Features (C1-C14) ===
    print("\n### Card-Related Features (C1-C14)")
    print("-" * 70)
    c_features = [f'C{i}' for i in range(1, 15)]
    for c_col in c_features:
        if c_col in train_df.columns:
            missing_count = train_df[c_col].isna().sum()
            missing_pct = (missing_count / total_rows) * 100
            results[c_col] = {
                'missing_count': missing_count,
                'total_rows': total_rows,
                'missing_pct': missing_pct
            }
            print(f"{c_col}:")
            print(f"  Missing: {missing_count:,} / {total_rows:,} ({missing_pct:.2f}%)")
        else:
            print(f"{c_col}: NOT FOUND in data")
    
    # === Device Features ===
    print("\n### Device Features")
    print("-" * 70)
    device_features = ['DeviceType', 'DeviceInfo']
    for device_col in device_features:
        if device_col in train_df.columns:
            missing_count = train_df[device_col].isna().sum()
            missing_pct = (missing_count / total_rows) * 100
            results[device_col] = {
                'missing_count': missing_count,
                'total_rows': total_rows,
                'missing_pct': missing_pct
            }
            print(f"{device_col}:")
            print(f"  Missing: {missing_count:,} / {total_rows:,} ({missing_pct:.2f}%)")
        else:
            print(f"{device_col}: NOT FOUND in data")
    
    # === Identity Features ===
    print("\n### Identity Features (id_28, id_29, id_30, id_31)")
    print("-" * 70)
    id_features = ['id_28', 'id_29', 'id_30', 'id_31']
    for id_col in id_features:
        # Check both formats (id_28 and id-28)
        col_name = id_col
        if id_col not in train_df.columns:
            # Try with hyphen
            col_name = id_col.replace('_', '-')
        
        if col_name in train_df.columns:
            missing_count = train_df[col_name].isna().sum()
            missing_pct = (missing_count / total_rows) * 100
            results[id_col] = {
                'missing_count': missing_count,
                'total_rows': total_rows,
                'missing_pct': missing_pct
            }
            print(f"{id_col} ({col_name}):")
            print(f"  Missing: {missing_count:,} / {total_rows:,} ({missing_pct:.2f}%)")
        else:
            print(f"{id_col}: NOT FOUND in data")
    
    # === Basic Features ===
    print("\n### Basic Features")
    print("-" * 70)
    basic_features = ['TransactionAmt', 'TransactionDT', 'ProductCD']
    for basic_col in basic_features:
        if basic_col in train_df.columns:
            missing_count = train_df[basic_col].isna().sum()
            missing_pct = (missing_count / total_rows) * 100
            results[basic_col] = {
                'missing_count': missing_count,
                'total_rows': total_rows,
                'missing_pct': missing_pct
            }
            print(f"{basic_col}:")
            print(f"  Missing: {missing_count:,} / {total_rows:,} ({missing_pct:.2f}%)")
        else:
            print(f"{basic_col}: NOT FOUND in data")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Group by missing percentage ranges
    no_missing = []
    low_missing = []  # < 20%
    medium_missing = []  # 20-50%
    high_missing = []  # > 50%
    
    for col, stats in results.items():
        pct = stats['missing_pct']
        if pct == 0:
            no_missing.append((col, pct))
        elif pct < 20:
            low_missing.append((col, pct))
        elif pct < 50:
            medium_missing.append((col, pct))
        else:
            high_missing.append((col, pct))
    
    print(f"\nNo Missing Data (0%): {len(no_missing)} features")
    for col, pct in no_missing:
        print(f"  - {col}: {pct:.2f}%")
    
    print(f"\nLow Missing Data (<20%): {len(low_missing)} features")
    for col, pct in sorted(low_missing, key=lambda x: x[1], reverse=True):
        print(f"  - {col}: {pct:.2f}%")
    
    print(f"\nMedium Missing Data (20-50%): {len(medium_missing)} features")
    for col, pct in sorted(medium_missing, key=lambda x: x[1], reverse=True):
        print(f"  - {col}: {pct:.2f}%")
    
    print(f"\nHigh Missing Data (>50%): {len(high_missing)} features")
    for col, pct in sorted(high_missing, key=lambda x: x[1], reverse=True):
        print(f"  - {col}: {pct:.2f}%")
    
    print("\n" + "=" * 70)
    
    return results


if __name__ == "__main__":
    results = calculate_missing_stats()
    
    # Save results to a file for easy reference
    # Handle both script execution and IPython/Jupyter
    try:
        script_dir = Path(__file__).parent
    except NameError:
        # Running in IPython/Jupyter - use current working directory
        script_dir = Path.cwd() / "src" / "eighth_view"
    
    output_file = script_dir / "missing_data_stats.txt"
    with open(output_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("EXACT MISSING DATA STATISTICS\n")
        f.write("=" * 70 + "\n\n")
        
        for col, stats in results.items():
            f.write(f"{col}:\n")
            f.write(f"  Missing: {stats['missing_count']:,} / {stats['total_rows']:,} ({stats['missing_pct']:.2f}%)\n\n")
    
    print(f"\nResults saved to: {output_file}")

