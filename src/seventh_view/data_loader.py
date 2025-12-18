"""Data loader: same as sixth view."""

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
                         sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and test data for sixth view with ALL card features and IP/dist features.
    
    Selected features:
    Transaction data:
    - card1-card6: All card identifiers (NEW!)
    - addr1-addr2: Address identifiers (NEW: addr2!)
    - TransactionAmt: Transaction amount
    - TransactionDT: Transaction datetime (seconds)
    - ProductCD: Product code
    - P_emaildomain: Purchaser email domain
    - R_emaildomain: Recipient email domain
    - dist1, dist2: IP distance features (NEW!)
    - C1-C14: Card-related features (NEW!)
    
    Identity data:
    - DeviceType: Device type (mobile/desktop)
    - DeviceInfo: Device information
    - id_28-id_31: Identity features
    
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
    
    if sample_size:
        print(f"Sampling {sample_size} rows for faster development...")
        train_df = train_df.sample(n=min(sample_size, len(train_df)), 
                                  random_state=42)
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    all_features = [f for f in available_trans_features + available_id_features 
                    if f not in ['TransactionID', 'isFraud']]
    print(f"Selected features ({len(all_features)}): {all_features[:20]}..." if len(all_features) > 20 else f"Selected features: {all_features}")
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

