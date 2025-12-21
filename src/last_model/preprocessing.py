"""Preprocessing: improved SMOTE sampling_strategy for better recall."""

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


def preprocess_sixth_view_data(df: pd.DataFrame, 
                               is_train: bool = True) -> pd.DataFrame:
    """
    Preprocesses data with proper NaN handling for SMOTE.
    
    Args:
        df: Input dataframe.
        is_train: Whether this is training data.
    
    Returns:
        Preprocessed dataframe with no NaN values.
    """
    df = df.copy()
    
    print("Preprocessing data...")
    
    # === Transaction Features ===
    # Card features (card1-card6) - fill with -1 for missing
    for card_col in ['card1', 'card2', 'card3', 'card4', 'card5', 'card6']:
        if card_col in df.columns:
            df[f'{card_col}_isMissing'] = df[card_col].isna().astype(int)
            df[card_col] = df[card_col].fillna(-1)
    
    # Address features (addr1, addr2)
    for addr_col in ['addr1', 'addr2']:
        if addr_col in df.columns:
            df[f'{addr_col}_isMissing'] = df[addr_col].isna().astype(int)
            df[addr_col] = df[addr_col].fillna(-1)
    
    # Email features
    for email_col in ['P_emaildomain', 'R_emaildomain']:
        if email_col in df.columns:
            df[f'{email_col}_isMissing'] = df[email_col].isna().astype(int)
            df[email_col] = df[email_col].fillna('missing')
    
    # Transaction amount
    if 'TransactionAmt' in df.columns:
        if df['TransactionAmt'].isna().sum() > 0:
            df['TransactionAmt'] = df['TransactionAmt'].fillna(df['TransactionAmt'].median())
    
    # Transaction datetime
    if 'TransactionDT' in df.columns:
        if df['TransactionDT'].isna().sum() > 0:
            print("Warning: TransactionDT has missing values!")
            df['TransactionDT'] = df['TransactionDT'].fillna(df['TransactionDT'].median())
    
    # Product code
    if 'ProductCD' in df.columns:
        if df['ProductCD'].isna().sum() > 0:
            df['ProductCD'] = df['ProductCD'].fillna('missing')
    
    # IP/Distance features (dist1, dist2)
    for dist_col in ['dist1', 'dist2']:
        if dist_col in df.columns:
            df[f'{dist_col}_isMissing'] = df[dist_col].isna().astype(int)
            # Fill with median for numeric, or -1 if all NaN
            if df[dist_col].dtype in ['float64', 'int64', 'float32', 'int32']:
                fill_value = df[dist_col].median() if df[dist_col].notna().sum() > 0 else -1
                df[dist_col] = df[dist_col].fillna(fill_value)
            else:
                df[dist_col] = df[dist_col].fillna(-1)
    
    # Card-related features (C1-C14)
    for c_col in [f'C{i}' for i in range(1, 15)]:
        if c_col in df.columns:
            df[f'{c_col}_isMissing'] = df[c_col].isna().astype(int)
            if df[c_col].dtype in ['float64', 'int64', 'float32', 'int32']:
                fill_value = df[c_col].median() if df[c_col].notna().sum() > 0 else -1
                df[c_col] = df[c_col].fillna(fill_value)
            else:
                df[c_col] = df[c_col].fillna(-1)
    
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
                fill_value = df[id_col].median() if df[id_col].notna().sum() > 0 else -1
                df[id_col] = df[id_col].fillna(fill_value)
            else:
                df[id_col] = df[id_col].fillna('missing')
    
    # CRITICAL: Fill ALL remaining NaN values to ensure SMOTE compatibility
    # This is important - SMOTE cannot handle NaN values
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            fill_value = df[col].median() if df[col].notna().sum() > 0 else 0
            df[col] = df[col].fillna(fill_value)
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna('missing')
    
    # Verify no NaN values remain
    remaining_nans = df.isna().sum().sum()
    if remaining_nans > 0:
        print(f"Warning: {remaining_nans} NaN values still remain after preprocessing!")
        # Force fill any remaining NaN
        df = df.fillna(0)
    
    # Reduce memory usage
    df = reduce_memory_usage(df, verbose=False)
    
    print(f"Preprocessing complete. Shape: {df.shape}")
    print(f"NaN values remaining: {df.isna().sum().sum()}")
    
    return df


def apply_smote(X: pd.DataFrame, 
                y: np.ndarray,
                sampling_strategy: float = 0.12,  # Increased from 0.1 to improve recall (fraud rate %9.09 → %12-15)
                k_neighbors: int = 5,
                random_state: int = 42,
                verbose: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance classes.
    ENHANCED: Properly handles NaN values by ensuring they're filled before SMOTE.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature dataframe (should have NO NaN values after preprocessing)
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
    
    # CRITICAL: Check for NaN values before SMOTE
    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        if verbose:
            print(f"Warning: {nan_count} NaN values found. Filling with median/0...")
        # Fill numeric columns with median
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if X[col].isna().sum() > 0:
                fill_value = X[col].median() if X[col].notna().sum() > 0 else 0
                X[col] = X[col].fillna(fill_value)
        
        # Fill categorical columns with 'missing'
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if X[col].isna().sum() > 0:
                X[col] = X[col].fillna('missing')
        
        # Final check
        if X.isna().sum().sum() > 0:
            X = X.fillna(0)
    
    # Separate numeric and categorical features
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]
    
    if len(categorical_cols) > 0:
        if verbose:
            print(f"Converting {len(categorical_cols)} categorical features to numeric for SMOTE...")
        
        # Convert categorical to numeric for SMOTE
        X_for_smote = X.copy()
        label_encoders = {}
        
        for col in categorical_cols:
            if X_for_smote[col].dtype.name == 'category':
                X_for_smote[col] = X_for_smote[col].cat.codes
            else:
                # Label encode
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                # Handle NaN values in string columns
                col_values = X_for_smote[col].astype(str)
                X_for_smote[col] = le.fit_transform(col_values)
                label_encoders[col] = le
    else:
        X_for_smote = X.copy()
    
    # Final NaN check before SMOTE
    if X_for_smote.isna().sum().sum() > 0:
        if verbose:
            print("Warning: NaN values still present after conversion. Filling with 0...")
        X_for_smote = X_for_smote.fillna(0)
    
    # Apply SMOTE
    if verbose:
        print(f"Applying SMOTE...")
        print(f"  Before: Fraud={y.sum()}, Non-fraud={len(y) - y.sum()}, Ratio={y.sum() / len(y):.4f}")
    
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=min(k_neighbors, y.sum() - 1),  # k_neighbors must be < n_minority
        random_state=random_state
        # Note: n_jobs parameter is not available in SMOTE
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
                    elif col in label_encoders:
                        # Clip to valid label encoder range
                        max_label = len(label_encoders[col].classes_) - 1
                        X_resampled[col] = X_resampled[col].clip(0, max_label)
        
        if verbose:
            print(f"  After: Fraud={y_resampled.sum()}, Non-fraud={len(y_resampled) - y_resampled.sum()}, Ratio={y_resampled.sum() / len(y_resampled):.4f}")
            print(f"  New samples: {len(y_resampled) - len(y)}")
        
        return X_resampled, y_resampled
    
    except Exception as e:
        if verbose:
            print(f"Error applying SMOTE: {e}")
            print("Returning original data without resampling.")
        return X, y

