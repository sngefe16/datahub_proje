"""
Data loading utilities for IEEE Fraud Detection project.
Handles loading and merging transaction and identity data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import sys
import os

# Cross-platform utilities
# Handle both relative imports (package) and absolute imports (direct/IPython execution)
try:
    from .utils import get_data_dir, get_project_root
except (ImportError, ValueError):
    # Fallback for direct execution or IPython console
    # Get the directory containing this file
    if '__file__' in globals():
        # Normal Python execution
        current_dir = Path(__file__).parent.parent
    else:
        # IPython/Jupyter execution - use current working directory
        current_dir = Path(os.getcwd())
    
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    from src.utils import get_data_dir, get_project_root


def load_data(data_dir: Optional[str] = None, 
              sample_size: Optional[int] = None,
              use_identity: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and test data.
    
    Parameters
    ----------
    data_dir : str
        Directory containing the data files
    sample_size : int, optional
        If provided, sample this many rows for faster development
    use_identity : bool
        Whether to merge identity data
        
    Returns
    -------
    train_df : pd.DataFrame
        Training data with target
    test_df : pd.DataFrame
        Test data
    """
    # Use default data directory if not specified
    if data_dir is None:
        data_path = get_data_dir()
    else:
        # Handle both relative and absolute paths
        data_path = Path(data_dir)
        if not data_path.is_absolute():
            data_path = get_project_root() / data_dir
    
    print("Loading transaction data...")
    train_trans = pd.read_csv(data_path / "train_transaction.csv")
    test_trans = pd.read_csv(data_path / "test_transaction.csv")
    
    if sample_size:
        print(f"Sampling {sample_size} rows for faster development...")
        train_trans = train_trans.sample(n=min(sample_size, len(train_trans)), 
                                         random_state=42)
    
    if use_identity:
        print("Loading identity data...")
        train_id = pd.read_csv(data_path / "train_identity.csv")
        test_id = pd.read_csv(data_path / "test_identity.csv")
        
        print("Merging transaction and identity data...")
        train_df = train_trans.merge(train_id, on="TransactionID", how="left")
        test_df = test_trans.merge(test_id, on="TransactionID", how="left")
    else:
        train_df = train_trans
        test_df = test_trans
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"Fraud rate: {train_df['isFraud'].mean():.4f}")
    
    return train_df, test_df


def get_feature_types(df: pd.DataFrame) -> dict:
    """
    Identify feature types in the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns
    -------
    feature_types : dict
        Dictionary with 'numeric', 'categorical', 'binary' keys
    """
    feature_types = {
        'numeric': [],
        'categorical': [],
        'binary': []
    }
    
    for col in df.columns:
        if col in ['TransactionID', 'isFraud']:
            continue
            
        if df[col].dtype in ['float64', 'int64']:
            unique_vals = df[col].nunique()
            if unique_vals == 2:
                feature_types['binary'].append(col)
            else:
                feature_types['numeric'].append(col)
        elif df[col].dtype == 'object':
            feature_types['categorical'].append(col)
    
    return feature_types


def get_missing_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate missing value statistics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns
    -------
    missing_df : pd.DataFrame
        DataFrame with missing value statistics
    """
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'missing_count': missing_count,
        'missing_pct': missing_pct
    }).sort_values('missing_pct', ascending=False)
    
    return missing_df[missing_df['missing_count'] > 0]


if __name__ == "__main__":
    # Example usage
    train_df, test_df = load_data(sample_size=10000)
    print("\nFeature types:")
    feature_types = get_feature_types(train_df)
    for ftype, features in feature_types.items():
        print(f"{ftype}: {len(features)} features")
    
    print("\nMissing value statistics (top 20):")
    missing_stats = get_missing_stats(train_df)
    print(missing_stats.head(20))

