"""
Preprocessing utilities for fourth view model.
Same as third view - handles missing values and basic transformations.
"""

import pandas as pd
import numpy as np
from typing import List


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


def preprocess_fourth_view_data(df: pd.DataFrame, 
                               is_train: bool = True) -> pd.DataFrame:
    """
    Preprocess data for fourth view model (same as third view).
    
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

