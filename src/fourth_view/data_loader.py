"""Data loader: same as third view."""

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


def load_fourth_view_data(data_dir: Optional[str] = None, 
                          sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and test data with enhanced features (same as third view).
    
    Parameters
    ----------
    data_dir : str, optional
        Directory containing the data files
    sample_size : int, optional
        If provided, sample this many rows for faster development
        
    Returns
    -------
    train_df : pd.DataFrame
        Training data with selected features and target
    test_df : pd.DataFrame
        Test data with selected features
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
    
    # Transaction features
    trans_features = ['TransactionID', 'TransactionDT', 'TransactionAmt', 
                     'ProductCD', 'card1', 'card2', 'addr1', 'isFraud']
    email_features = ['P_emaildomain', 'R_emaildomain']
    
    # Identity features
    identity_features = ['TransactionID', 'DeviceType', 'DeviceInfo', 
                        'id_28', 'id_29', 'id_30', 'id_31']
    
    # Check which features exist
    available_trans_features = []
    for feat in trans_features + email_features:
        if feat in train_trans.columns:
            available_trans_features.append(feat)
        else:
            print(f"Warning: {feat} not found in transaction data")
    
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
    
    if sample_size:
        print(f"Sampling {sample_size} rows for faster development...")
        train_df = train_df.sample(n=min(sample_size, len(train_df)), 
                                  random_state=42)
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    all_features = [f for f in available_trans_features + available_id_features 
                    if f not in ['TransactionID', 'isFraud']]
    print(f"Selected features: {all_features}")
    print(f"Fraud rate: {train_df['isFraud'].mean():.4f}")
    
    return train_df, test_df


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





