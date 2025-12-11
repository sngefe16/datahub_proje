"""
Data loader for first view model.
Loads only selected features: card1, card2, addr1, TransactionAmt, TransactionDT, ProductCD, email_domain
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


def load_first_view_data(data_dir: Optional[str] = None, 
                         sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and test data with only selected features.
    
    Selected features:
    - card1: Primary card identifier
    - card2: Secondary card identifier
    - addr1: Address identifier
    - TransactionAmt: Transaction amount
    - TransactionDT: Transaction datetime (seconds)
    - ProductCD: Product code
    - P_emaildomain: Purchaser email domain
    - R_emaildomain: Recipient email domain
    
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
    
    # Selected features
    base_features = ['TransactionID', 'TransactionDT', 'TransactionAmt', 
                     'ProductCD', 'card1', 'card2', 'addr1', 'isFraud']
    email_features = ['P_emaildomain', 'R_emaildomain']
    
    # Check which email features exist
    available_features = []
    for feat in base_features + email_features:
        if feat in train_trans.columns:
            available_features.append(feat)
        else:
            print(f"Warning: {feat} not found in data")
    
    # Select only available features
    train_df = train_trans[available_features].copy()
    test_df = test_trans[[f for f in available_features if f != 'isFraud']].copy()
    
    if sample_size:
        print(f"Sampling {sample_size} rows for faster development...")
        train_df = train_df.sample(n=min(sample_size, len(train_df)), 
                                  random_state=42)
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"Selected features: {[f for f in available_features if f not in ['TransactionID', 'isFraud']]}")
    print(f"Fraud rate: {train_df['isFraud'].mean():.4f}")
    
    return train_df, test_df


def get_feature_info(df: pd.DataFrame) -> dict:
    """
    Get information about selected features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns
    -------
    info : dict
        Dictionary with feature information
    """
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


if __name__ == "__main__":
    # Example usage
    train_df, test_df = load_first_view_data(sample_size=10000)
    print("\nFeature information:")
    feature_info = get_feature_info(train_df)
    for feat, info in feature_info.items():
        print(f"{feat}: {info}")


