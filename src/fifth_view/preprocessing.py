"""
Preprocessing utilities for fifth view model.
Extends fourth view with SMOTE for class imbalance handling.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try to import SMOTE
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("Warning: imbalanced-learn not available. Install with: pip install imbalanced-learn")


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Reduce memory usage by optimizing data types."""
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        if col in ['TransactionID', 'isFraud']:
            continue
            
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
        else:
            if df[col].nunique() < 100:
                df[col] = df[col].astype('category')
    
    end_mem = df.memory_usage().sum() / 1024**2
    
    if verbose:
        print(f"Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB "
              f"({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")
    
    return df


def preprocess_fifth_view_data(df: pd.DataFrame, 
                               is_train: bool = True) -> pd.DataFrame:
    """
    Preprocess data for fifth view model (same as fourth view).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    is_train : bool
        Whether this is training data
        
    Returns
    -------
    df : pd.DataFrame
        Preprocessed dataframe
    """
    df = df.copy()
    
    print("Preprocessing data...")
    
    # === Transaction Features ===
    if 'card2' in df.columns:
        df['card2_isMissing'] = df['card2'].isna().astype(int)
        df['card2'] = df['card2'].fillna(-1)
    
    if 'addr1' in df.columns:
        df['addr1_isMissing'] = df['addr1'].isna().astype(int)
        df['addr1'] = df['addr1'].fillna(-1)
    
    for email_col in ['P_emaildomain', 'R_emaildomain']:
        if email_col in df.columns:
            df[f'{email_col}_isMissing'] = df[email_col].isna().astype(int)
            df[email_col] = df[email_col].fillna('missing')
    
    if 'TransactionAmt' in df.columns:
        if df['TransactionAmt'].isna().sum() > 0:
            df['TransactionAmt'] = df['TransactionAmt'].fillna(df['TransactionAmt'].median())
    
    if 'TransactionDT' in df.columns:
        if df['TransactionDT'].isna().sum() > 0:
            print("Warning: TransactionDT has missing values!")
    
    if 'ProductCD' in df.columns:
        if df['ProductCD'].isna().sum() > 0:
            df['ProductCD'] = df['ProductCD'].fillna('missing')
    
    if 'card1' in df.columns:
        if df['card1'].isna().sum() > 0:
            df['card1'] = df['card1'].fillna(-1)
    
    # === Identity Features ===
    if 'DeviceType' in df.columns:
        df['DeviceType_isMissing'] = df['DeviceType'].isna().astype(int)
        df['DeviceType'] = df['DeviceType'].fillna('missing')
    
    if 'DeviceInfo' in df.columns:
        df['DeviceInfo_isMissing'] = df['DeviceInfo'].isna().astype(int)
        df['DeviceInfo'] = df['DeviceInfo'].fillna('missing')
    
    for id_col in ['id_28', 'id_29', 'id_30', 'id_31']:
        if id_col in df.columns:
            df[f'{id_col}_isMissing'] = df[id_col].isna().astype(int)
            if df[id_col].dtype in ['float64', 'int64', 'float32', 'int32', 'int16', 'int8']:
                df[id_col] = df[id_col].fillna(-1)
            else:
                df[id_col] = df[id_col].fillna('missing')
    
    # Reduce memory usage
    df = reduce_memory_usage(df, verbose=False)
    
    print(f"Preprocessing complete. Shape: {df.shape}")
    
    return df


def apply_smote(X: pd.DataFrame, 
                y: np.ndarray,
                sampling_strategy: float = 0.1,
                k_neighbors: int = 5,
                random_state: int = 42,
                verbose: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance classes.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature dataframe
    y : np.ndarray
        Target array
    sampling_strategy : float
        Ratio of minority class to majority class after resampling (0.1 = 10% of majority)
    k_neighbors : int
        Number of nearest neighbors for SMOTE
    random_state : int
        Random seed
    verbose : bool
        Whether to print information
        
    Returns
    -------
    X_resampled : pd.DataFrame
        Resampled feature dataframe
    y_resampled : np.ndarray
        Resampled target array
    """
    if not SMOTE_AVAILABLE:
        if verbose:
            print("Warning: SMOTE not available. Skipping resampling.")
        return X, y
    
    # Separate numeric and categorical features
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]
    
    if len(categorical_cols) > 0:
        if verbose:
            print(f"Warning: {len(categorical_cols)} categorical features found. SMOTE works best with numeric features.")
            print("Converting categorical features to numeric for SMOTE...")
        
        # Convert categorical to numeric for SMOTE
        X_for_smote = X.copy()
        for col in categorical_cols:
            if X_for_smote[col].dtype.name == 'category':
                X_for_smote[col] = X_for_smote[col].cat.codes
            else:
                # Label encode
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                X_for_smote[col] = le.fit_transform(X_for_smote[col].astype(str))
    else:
        X_for_smote = X.copy()
    
    # Apply SMOTE
    if verbose:
        print(f"Applying SMOTE...")
        print(f"  Before: Fraud={y.sum()}, Non-fraud={len(y) - y.sum()}, Ratio={y.sum() / len(y):.4f}")
    
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=min(k_neighbors, y.sum() - 1),  # k_neighbors must be < n_minority
        random_state=random_state
    )
    
    try:
        X_resampled, y_resampled = smote.fit_resample(X_for_smote, y)
        
        # Convert back to DataFrame with original column names
        X_resampled = pd.DataFrame(X_resampled, columns=X_for_smote.columns)
        
        # Restore categorical features if needed
        if len(categorical_cols) > 0:
            # Note: SMOTE generates synthetic samples, so categorical values may not be exact
            # Round to nearest integer category
            for col in categorical_cols:
                X_resampled[col] = X_resampled[col].round().astype(int)
                # Clip to valid range
                if col in X.columns:
                    if X[col].dtype.name == 'category':
                        max_cat = len(X[col].cat.categories) - 1
                        X_resampled[col] = X_resampled[col].clip(0, max_cat)
        
        if verbose:
            print(f"  After: Fraud={y_resampled.sum()}, Non-fraud={len(y_resampled) - y_resampled.sum()}, Ratio={y_resampled.sum() / len(y_resampled):.4f}")
            print(f"  New samples: {len(y_resampled) - len(y)}")
        
        return X_resampled, y_resampled
    
    except Exception as e:
        if verbose:
            print(f"Error applying SMOTE: {e}")
            print("Returning original data without resampling.")
        return X, y

