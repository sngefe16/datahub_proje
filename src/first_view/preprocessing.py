"""Preprocessing: missing values, type optimization."""

import pandas as pd
import numpy as np
from typing import List


def preprocess_first_view_data(df: pd.DataFrame, 
                               is_train: bool = True) -> pd.DataFrame:
    """
    Preprocesses data: handles missing values, creates indicators, optimizes types.
    
    Args:
        df: Input dataframe.
        is_train: Whether this is training data.
    
    Returns:
        Preprocessed dataframe.
    """
    df = df.copy()
    
    print("Preprocessing data...")
    
    # Handle missing values for each feature
    # card2: Fill missing with -1 (new category)
    if 'card2' in df.columns:
        df['card2_isMissing'] = df['card2'].isna().astype(int)
        df['card2'] = df['card2'].fillna(-1)
    
    # addr1: Fill missing with -1 (new category)
    if 'addr1' in df.columns:
        df['addr1_isMissing'] = df['addr1'].isna().astype(int)
        df['addr1'] = df['addr1'].fillna(-1)
    
    # Email domains: Fill missing with 'missing'
    for email_col in ['P_emaildomain', 'R_emaildomain']:
        if email_col in df.columns:
            df[f'{email_col}_isMissing'] = df[email_col].isna().astype(int)
            df[email_col] = df[email_col].fillna('missing')
    
    # TransactionAmt: No missing values expected, but check
    if 'TransactionAmt' in df.columns:
        if df['TransactionAmt'].isna().sum() > 0:
            df['TransactionAmt'] = df['TransactionAmt'].fillna(df['TransactionAmt'].median())
    
    # TransactionDT: No missing values expected
    if 'TransactionDT' in df.columns:
        if df['TransactionDT'].isna().sum() > 0:
            print("Warning: TransactionDT has missing values!")
    
    # ProductCD: No missing values expected
    if 'ProductCD' in df.columns:
        if df['ProductCD'].isna().sum() > 0:
            df['ProductCD'] = df['ProductCD'].fillna('missing')
    
    # card1: No missing values expected
    if 'card1' in df.columns:
        if df['card1'].isna().sum() > 0:
            df['card1'] = df['card1'].fillna(-1)
    
    print(f"Preprocessing complete. Shape: {df.shape}")
    
    return df


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Reduce memory usage by optimizing data types.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    verbose : bool
        Whether to print memory reduction info
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with optimized data types
    """
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
            # Convert object to category if low cardinality
            if df[col].nunique() < 100:
                df[col] = df[col].astype('category')
    
    end_mem = df.memory_usage().sum() / 1024**2
    
    if verbose:
        print(f"Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB "
              f"({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")
    
    return df






