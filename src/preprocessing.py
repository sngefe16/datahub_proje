"""
Data preprocessing utilities for IEEE Fraud Detection project.
Handles missing values, data types, and basic transformations.
"""

import pandas as pd
import numpy as np
from typing import List, Dict


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
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float32)
        else:
            df[col] = df[col].astype('category')
    
    end_mem = df.memory_usage().sum() / 1024**2
    
    if verbose:
        print(f"Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB "
              f"({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")
    
    return df


def handle_missing_values(df: pd.DataFrame, 
                         missing_threshold: float = 0.9,
                         create_indicators: bool = True) -> pd.DataFrame:
    """
    Handle missing values by dropping high-missing features and creating indicators.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    missing_threshold : float
        Features with > this % missing will be dropped
    create_indicators : bool
        Whether to create missing value indicators
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with handled missing values
    """
    df = df.copy()
    
    # Calculate missing percentages
    missing_pct = df.isnull().sum() / len(df)
    
    # Drop features with very high missing percentage
    cols_to_drop = missing_pct[missing_pct > missing_threshold].index.tolist()
    if cols_to_drop:
        print(f"Dropping {len(cols_to_drop)} features with >{missing_threshold*100}% missing values")
        df = df.drop(columns=cols_to_drop)
    
    # Create missing indicators for remaining features
    if create_indicators:
        missing_cols = df.columns[df.isnull().any()].tolist()
        for col in missing_cols:
            if col not in ['TransactionID', 'isFraud']:
                df[f'{col}_isMissing'] = df[col].isnull().astype(int)
    
    return df


def impute_missing_values(df: pd.DataFrame, 
                         strategy: str = 'median',
                         numeric_cols: List[str] = None,
                         categorical_cols: List[str] = None) -> pd.DataFrame:
    """
    Impute missing values using specified strategy.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    strategy : str
        'median', 'mean', 'mode', or 'zero'
    numeric_cols : list
        List of numeric columns to impute
    categorical_cols : list
        List of categorical columns to impute
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with imputed values
    """
    df = df.copy()
    
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ['TransactionID', 'isFraud']]
    
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Impute numeric columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            if strategy == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == 'zero':
                df[col].fillna(0, inplace=True)
    
    # Impute categorical columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna('missing', inplace=True)
    
    return df


def encode_categoricals(df: pd.DataFrame,
                       method: str = 'label',
                       categorical_cols: List[str] = None) -> pd.DataFrame:
    """
    Encode categorical variables.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    method : str
        'label' or 'onehot'
    categorical_cols : list
        List of categorical columns to encode
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with encoded categoricals
    """
    df = df.copy()
    
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if method == 'label':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        
        for col in categorical_cols:
            df[col] = le.fit_transform(df[col].astype(str))
    
    elif method == 'onehot':
        df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)
    
    return df









