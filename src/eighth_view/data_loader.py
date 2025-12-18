"""
Data loader with dual-mode support (debug/full).
Loads transaction + identity data with all card features (card1-card6), IP/dist features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import sys
import os

# Cross-platform utilities
try:
    from ..utils import get_data_dir, get_project_root
except (ImportError, ValueError):
    if '__file__' in globals():
        current_dir = Path(__file__).parent.parent.parent
    else:
        current_dir = Path(os.getcwd())
    
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    from src.utils import get_data_dir, get_project_root


def load_sixth_view_data(data_dir: Optional[str] = None, 
                         use_full_data: bool = False,
                         debug_sample_size: int = 20000,
                         debug_fraud_rate: float = 0.035,
                         random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads transaction + identity data.
    
    Dual-mode:
    - DEBUG (use_full_data=False): Samples 20k rows, preserves fraud rate.
    - FULL (use_full_data=True): Uses all 590k rows.
    
    Features: card1-card6, addr1-addr2, TransactionAmt, TransactionDT, ProductCD,
    P_emaildomain, R_emaildomain, dist1/dist2, C1-C14, DeviceType, DeviceInfo, id_28-id_31.
    
    Args:
        data_dir: Data directory path.
        use_full_data: True for full dataset, False for debug sample.
        debug_sample_size: Sample size for debug mode.
        debug_fraud_rate: Target fraud rate for debug mode.
        random_state: Random seed.
    
    Returns:
        (train_df, test_df) with isFraud target in train_df.
    
    Raises:
        ValueError: If fraud rate deviates > 0.2% from target.
    """
    # Use default data directory if not specified
    if data_dir is None:
        data_path = get_data_dir()
    else:
        data_path = Path(data_dir)
        if not data_path.is_absolute():
            data_path = get_project_root() / data_dir
    
    print("Loading transaction data...")
    train_trans = pd.read_csv(data_path / "train_transaction.csv")
    test_trans = pd.read_csv(data_path / "test_transaction.csv")
    
    print("Loading identity data...")
    train_id = pd.read_csv(data_path / "train_identity.csv")
    test_id = pd.read_csv(data_path / "test_identity.csv")
    
    # Normalize test identity column names (id-28 -> id_28)
    test_id_rename_dict = {}
    for col in test_id.columns:
        if col.startswith('id-'):
            test_id_rename_dict[col] = col.replace('-', '_')
    if test_id_rename_dict:
        test_id = test_id.rename(columns=test_id_rename_dict)
    
    # Transaction features - ENHANCED with all cards and IP/dist
    trans_features = ['TransactionID', 'TransactionDT', 'TransactionAmt', 
                     'ProductCD', 'isFraud']
    
    # All card features (card1-card6)
    card_features = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6']
    
    # Address features (addr1, addr2)
    addr_features = ['addr1', 'addr2']
    
    # Email features
    email_features = ['P_emaildomain', 'R_emaildomain']
    
    # IP/Distance features
    ip_dist_features = ['dist1', 'dist2']
    
    # Card-related features (C1-C14)
    card_related_features = [f'C{i}' for i in range(1, 15)]
    
    # Identity features
    identity_features = ['TransactionID', 'DeviceType', 'DeviceInfo', 
                        'id_28', 'id_29', 'id_30', 'id_31']
    
    # Check which features exist in transaction data
    all_trans_features = (trans_features + card_features + addr_features + 
                         email_features + ip_dist_features + card_related_features)
    
    available_trans_features = []
    for feat in all_trans_features:
        if feat in train_trans.columns:
            available_trans_features.append(feat)
        else:
            if feat not in ['isFraud']:  # isFraud is expected to be missing in test
                print(f"Warning: {feat} not found in transaction data")
    
    # Check which features exist in identity data
    available_id_features = []
    for feat in identity_features:
        if feat in train_id.columns:
            available_id_features.append(feat)
        else:
            print(f"Warning: {feat} not found in train identity data")
    
    # Verify test identity has the features after renaming
    for feat in available_id_features:
        if feat not in test_id.columns:
            print(f"Warning: {feat} not found in test identity data (after renaming)")
    
    # Select features
    train_trans_selected = train_trans[available_trans_features].copy()
    test_trans_selected = test_trans[[f for f in available_trans_features if f != 'isFraud']].copy()
    
    train_id_selected = train_id[available_id_features].copy()
    test_id_selected = test_id[available_id_features].copy()
    
    # Merge transaction and identity data
    print("Merging transaction and identity data...")
    train_df = train_trans_selected.merge(train_id_selected, on='TransactionID', how='left')
    test_df = test_trans_selected.merge(test_id_selected, on='TransactionID', how='left')
    
    # ============================================================
    # DUAL-MODE PIPELINE: DEBUG vs FULL DATA
    # ============================================================
    # Why debug mode exists:
    # - Faster iteration during feature engineering and model development
    # - Reduces risk of bugs in production pipeline
    # - Allows testing on representative subset before full training
    # Why class ratio preservation matters:
    # - Maintains realistic fraud detection scenario
    # - Ensures validation metrics are meaningful
    # - Prevents overfitting to wrong class distribution
    
    original_train_size = len(train_df)
    original_fraud_rate = train_df['isFraud'].mean()
    
    if use_full_data:
        # FULL MODE: Use entire dataset
        print("\n" + "=" * 70)
        print("🔴 FULL DATA MODE")
        print("=" * 70)
        print(f"Using entire dataset: {original_train_size:,} rows")
        print(f"Original fraud rate: {original_fraud_rate:.4f} ({original_fraud_rate*100:.2f}%)")
        print("=" * 70)
        
        # Safety check: Verify fraud rate is reasonable
        expected_fraud_rate = 0.035  # ~3.5%
        fraud_rate_diff = abs(original_fraud_rate - expected_fraud_rate)
        if fraud_rate_diff > 0.002:  # 0.2% tolerance
            print(f"⚠️  WARNING: Fraud rate deviates from expected {expected_fraud_rate*100:.2f}%")
            print(f"   Actual: {original_fraud_rate*100:.4f}%, Difference: {fraud_rate_diff*100:.4f}%")
        else:
            print(f"✅ Fraud rate within tolerance: {fraud_rate_diff*100:.4f}% deviation")
        
        # No sampling - use full data
        train_df_sampled = train_df.copy()
        test_df_sampled = test_df.copy()
        
    else:
        # DEBUG MODE: Stratified sampling with preserved class ratio
        print("\n" + "=" * 70)
        print("🟢 DEBUG MODE")
        print("=" * 70)
        print(f"Sampling {debug_sample_size:,} rows with {debug_fraud_rate*100:.2f}% fraud rate...")
        print(f"Original dataset: {original_train_size:,} rows ({original_fraud_rate*100:.2f}% fraud)")
        print("=" * 70)
        
        # Use centralized stratified sampling
        train_df_sampled = sample_balanced_data(
            train_df, 
            target_size=debug_sample_size,
            fraud_rate=debug_fraud_rate,
            random_state=random_state,
            tolerance=0.002  # 0.2% tolerance for safety check
        )
        
        # Sample test data proportionally (test set has no isFraud, so just take first N rows)
        # This is safe because test data order doesn't affect training
        test_sample_size = min(debug_sample_size, len(test_df))
        test_df_sampled = test_df.head(test_sample_size).copy()
        print(f"\nTest data sampled: {test_sample_size:,} rows (for faster development)")
        
        # Safety check: Verify sampled fraud rate
        sampled_fraud_rate = train_df_sampled['isFraud'].mean()
        fraud_rate_diff = abs(sampled_fraud_rate - debug_fraud_rate)
        if fraud_rate_diff > 0.002:  # 0.2% tolerance
            raise ValueError(
                f"Sampled fraud rate deviates too much from target: "
                f"expected {debug_fraud_rate*100:.4f}%, got {sampled_fraud_rate*100:.4f}%, "
                f"difference: {fraud_rate_diff*100:.4f}% (tolerance: 0.2%)"
            )
        else:
            print(f"✅ Sampled fraud rate verified: {sampled_fraud_rate*100:.4f}% "
                  f"(target: {debug_fraud_rate*100:.4f}%, diff: {fraud_rate_diff*100:.4f}%)")
    
    # Final logging
    print(f"\n" + "=" * 70)
    print("DATA LOADING SUMMARY")
    print("=" * 70)
    print(f"Mode: {'FULL DATA' if use_full_data else 'DEBUG'}")
    print(f"Training data: {len(train_df_sampled):,} rows")
    print(f"Test data: {len(test_df_sampled):,} rows")
    print(f"Fraud rate: {train_df_sampled['isFraud'].mean():.4f} ({train_df_sampled['isFraud'].mean()*100:.2f}%)")
    
    all_features = [f for f in available_trans_features + available_id_features 
                    if f not in ['TransactionID', 'isFraud']]
    print(f"Features: {len(all_features)}")
    print("=" * 70)
    
    return train_df_sampled, test_df_sampled


def sample_balanced_data(train_df: pd.DataFrame, 
                        target_size: int = 10000,
                        fraud_rate: float = 0.035,
                        random_state: int = 42,
                        tolerance: float = 0.0001) -> pd.DataFrame:
    """
    Sample a balanced dataset with specified fraud rate.
    Ensures the actual fraud rate matches the target fraud rate exactly.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Full training dataframe with 'isFraud' column
    target_size : int
        Target total number of rows (default: 10000)
    fraud_rate : float
        Target fraud rate (default: 0.035 = 3.5%)
    random_state : int
        Random seed for reproducibility
    tolerance : float
        Maximum allowed difference between target and actual fraud rate (default: 0.0001 = 0.01%)
        
    Returns
    -------
    sampled_df : pd.DataFrame
        Sampled dataframe with exactly target_size rows and fraud_rate fraud percentage
        
    Raises
    ------
    ValueError
        If there are not enough fraud or normal transactions, or if actual fraud rate
        differs from target by more than tolerance
    """
    if 'isFraud' not in train_df.columns:
        raise ValueError("DataFrame must contain 'isFraud' column")
    
    if not (0 < fraud_rate < 1):
        raise ValueError(f"fraud_rate must be between 0 and 1, got {fraud_rate}")
    
    # Separate fraud and normal transactions
    fraud_df = train_df[train_df['isFraud'] == 1].copy()
    normal_df = train_df[train_df['isFraud'] == 0].copy()
    
    print(f"Original data: {len(train_df):,} rows")
    print(f"  Fraud: {len(fraud_df):,} ({len(fraud_df)/len(train_df)*100:.2f}%)")
    print(f"  Normal: {len(normal_df):,} ({len(normal_df)/len(train_df)*100:.2f}%)")
    
    # Calculate target fraud and normal counts with proper rounding
    # Use round() to ensure we get the closest integer to the target
    n_fraud_target = round(target_size * fraud_rate)
    n_normal_target = target_size - n_fraud_target
    
    # Verify the calculation is correct
    calculated_fraud_rate = n_fraud_target / target_size
    if abs(calculated_fraud_rate - fraud_rate) > tolerance:
        # If rounding causes too much deviation, adjust
        # Try rounding up or down to see which is closer
        n_fraud_floor = int(target_size * fraud_rate)
        n_fraud_ceil = n_fraud_floor + 1
        
        rate_floor = n_fraud_floor / target_size
        rate_ceil = n_fraud_ceil / target_size
        
        if abs(rate_floor - fraud_rate) < abs(rate_ceil - fraud_rate):
            n_fraud_target = n_fraud_floor
        else:
            n_fraud_target = n_fraud_ceil
        
        n_normal_target = target_size - n_fraud_target
        calculated_fraud_rate = n_fraud_target / target_size
    
    print(f"\nTarget sampling:")
    print(f"  Target total: {target_size:,} rows")
    print(f"  Target fraud: {n_fraud_target:,} ({calculated_fraud_rate*100:.4f}%)")
    print(f"  Target normal: {n_normal_target:,} ({(1-calculated_fraud_rate)*100:.4f}%)")
    
    # Check if we have enough fraud transactions
    if len(fraud_df) < n_fraud_target:
        raise ValueError(
            f"Insufficient fraud transactions: need {n_fraud_target:,}, "
            f"but only {len(fraud_df):,} available"
        )
    
    # Check if we have enough normal transactions
    if len(normal_df) < n_normal_target:
        raise ValueError(
            f"Insufficient normal transactions: need {n_normal_target:,}, "
            f"but only {len(normal_df):,} available"
        )
    
    # Sample fraud transactions
    fraud_sampled = fraud_df.sample(n=n_fraud_target, random_state=random_state).copy()
    
    # Sample normal transactions
    normal_sampled = normal_df.sample(n=n_normal_target, random_state=random_state).copy()
    
    # Combine and shuffle
    sampled_df = pd.concat([fraud_sampled, normal_sampled], ignore_index=True)
    sampled_df = sampled_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Verify the final result
    actual_fraud_count = int(sampled_df['isFraud'].sum())
    actual_normal_count = int((sampled_df['isFraud'] == 0).sum())
    actual_total = len(sampled_df)
    actual_fraud_rate = sampled_df['isFraud'].mean()
    
    # Verify total size
    if actual_total != target_size:
        raise ValueError(
            f"Sampled data size mismatch: expected {target_size:,}, got {actual_total:,}"
        )
    
    # Verify fraud count
    if actual_fraud_count != n_fraud_target:
        raise ValueError(
            f"Fraud count mismatch: expected {n_fraud_target:,}, got {actual_fraud_count:,}"
        )
    
    # Verify fraud rate
    fraud_rate_diff = abs(actual_fraud_rate - fraud_rate)
    if fraud_rate_diff > tolerance:
        raise ValueError(
            f"Fraud rate mismatch: expected {fraud_rate:.6f} ({fraud_rate*100:.4f}%), "
            f"got {actual_fraud_rate:.6f} ({actual_fraud_rate*100:.4f}%), "
            f"difference: {fraud_rate_diff:.6f} (tolerance: {tolerance:.6f})"
        )
    
    print(f"\n✅ Sampled data verification:")
    print(f"  Total rows: {actual_total:,} (target: {target_size:,}) ✓")
    print(f"  Fraud: {actual_fraud_count:,} ({actual_fraud_rate*100:.4f}%) (target: {n_fraud_target:,}, {fraud_rate*100:.4f}%) ✓")
    print(f"  Normal: {actual_normal_count:,} ({(1-actual_fraud_rate)*100:.4f}%) (target: {n_normal_target:,}, {(1-fraud_rate)*100:.4f}%) ✓")
    print(f"  Fraud rate difference: {fraud_rate_diff:.6f} (tolerance: {tolerance:.6f}) ✓")
    
    return sampled_df


def get_feature_info(df: pd.DataFrame) -> dict:
    """Get information about selected features."""
    info = {}
    
    for col in df.columns:
        if col in ['TransactionID', 'isFraud']:
            continue
        
        info[col] = {
            'dtype': str(df[col].dtype),
            'nunique': df[col].nunique(),
            'missing': df[col].isna().sum(),
            'missing_pct': (df[col].isna().sum() / len(df)) * 100
        }
    
    return info

