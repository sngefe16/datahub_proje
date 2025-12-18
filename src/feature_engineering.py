"""
Feature engineering utilities for IEEE Fraud Detection project.
Creates time-based, aggregated, and interaction features.
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.preprocessing import LabelEncoder


def create_time_features(df: pd.DataFrame, 
                        time_col: str = 'TransactionDT') -> pd.DataFrame:
    """
    Create time-based features from TransactionDT.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    time_col : str
        Name of the time column
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with time features added
    """
    df = df.copy()
    
    # Convert TransactionDT to datetime-like features
    # TransactionDT appears to be seconds since some reference point
    
    # Day of week (0=Monday, 6=Sunday)
    df['day_of_week'] = (df[time_col] // (24 * 3600)) % 7
    
    # Hour of day
    df['hour'] = (df[time_col] // 3600) % 24
    
    # Day of month (approximate)
    df['day'] = (df[time_col] // (24 * 3600)) % 30
    
    # Week of year (approximate)
    df['week'] = (df[time_col] // (7 * 24 * 3600)) % 52
    
    # Is weekend
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Is night (22:00 - 06:00)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
    
    return df


def create_transaction_amount_features(df: pd.DataFrame,
                                      amount_col: str = 'TransactionAmt') -> pd.DataFrame:
    """
    Create features from transaction amount.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    amount_col : str
        Name of the amount column
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with amount features added
    """
    df = df.copy()
    
    # Log transformation (handle zeros)
    df[f'{amount_col}_log'] = np.log1p(df[amount_col])
    
    # Amount bins
    df[f'{amount_col}_bin'] = pd.qcut(df[amount_col], q=10, labels=False, duplicates='drop')
    
    # Amount percentiles
    df[f'{amount_col}_percentile'] = df[amount_col].rank(pct=True)
    
    return df


def create_aggregated_features(df: pd.DataFrame,
                              group_cols: List[str],
                              agg_cols: List[str],
                              agg_funcs: List[str] = ['mean', 'std', 'min', 'max', 'count']) -> pd.DataFrame:
    """
    Create aggregated features grouped by specified columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    group_cols : list
        Columns to group by
    agg_cols : list
        Columns to aggregate
    agg_funcs : list
        Aggregation functions to apply
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with aggregated features added
    """
    df = df.copy()
    
    for group_col in group_cols:
        if group_col not in df.columns:
            continue
            
        for agg_col in agg_cols:
            if agg_col not in df.columns:
                continue
            
            # Calculate aggregations
            grouped = df.groupby(group_col)[agg_col].agg(agg_funcs)
            
            # Create feature names
            for func in agg_funcs:
                if func in grouped.columns:
                    feature_name = f'{agg_col}_{func}_by_{group_col}'
                    df[feature_name] = df[group_col].map(grouped[func])
    
    return df


def create_frequency_features(df: pd.DataFrame,
                            categorical_cols: List[str]) -> pd.DataFrame:
    """
    Create frequency encoding for categorical features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    categorical_cols : list
        Categorical columns to encode
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with frequency features added
    """
    df = df.copy()
    
    for col in categorical_cols:
        if col in df.columns:
            freq_map = df[col].value_counts().to_dict()
            df[f'{col}_freq'] = df[col].map(freq_map)
    
    return df


def create_card_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create card-specific aggregated features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with card features added
    """
    df = df.copy()
    
    # Card transaction count
    if 'card1' in df.columns:
        card_counts = df.groupby('card1').size()
        df['card1_count'] = df['card1'].map(card_counts)
        
        # Card average transaction amount
        if 'TransactionAmt' in df.columns:
            card_avg_amt = df.groupby('card1')['TransactionAmt'].mean()
            df['card1_avg_amt'] = df['card1'].map(card_avg_amt)
            
            # Ratio of current amount to card average
            df['TransactionAmt_to_card1_avg'] = df['TransactionAmt'] / (df['card1_avg_amt'] + 1e-6)
    
    return df


def create_email_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create email domain features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with email features added
    """
    df = df.copy()
    
    email_cols = ['P_emaildomain', 'R_emaildomain']
    
    for col in email_cols:
        if col in df.columns:
            # Extract top-level domain
            df[f'{col}_tld'] = df[col].str.split('.').str[-1]
            
            # Is common email provider
            common_providers = ['gmail', 'yahoo', 'hotmail', 'outlook', 'aol']
            df[f'{col}_is_common'] = df[col].str.split('@').str[-1].str.split('.').str[0].isin(common_providers).astype(int)
    
    return df


def create_interaction_features(df: pd.DataFrame,
                               feature_pairs: List[tuple]) -> pd.DataFrame:
    """
    Create interaction features between pairs of features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    feature_pairs : list
        List of (col1, col2) tuples to create interactions
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with interaction features added
    """
    df = df.copy()
    
    for col1, col2 in feature_pairs:
        if col1 in df.columns and col2 in df.columns:
            # Multiplication
            if df[col1].dtype in [np.float64, np.int64] and df[col2].dtype in [np.float64, np.int64]:
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
            
            # Division (avoid division by zero)
            if df[col1].dtype in [np.float64, np.int64] and df[col2].dtype in [np.float64, np.int64]:
                df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-6)
    
    return df


def create_all_features(df: pd.DataFrame, 
                       is_train: bool = True) -> pd.DataFrame:
    """
    Create all engineered features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    is_train : bool
        Whether this is training data (for target encoding)
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with all features added
    """
    print("Creating time features...")
    df = create_time_features(df)
    
    print("Creating transaction amount features...")
    df = create_transaction_amount_features(df)
    
    print("Creating card features...")
    df = create_card_features(df)
    
    print("Creating email features...")
    df = create_email_features(df)
    
    # Frequency encoding for categoricals
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c not in ['TransactionID']]
    
    if categorical_cols:
        print(f"Creating frequency features for {len(categorical_cols)} categorical columns...")
        df = create_frequency_features(df, categorical_cols)
    
    # Interaction features
    interaction_pairs = [
        ('TransactionAmt', 'ProductCD'),
        ('card1', 'ProductCD'),
    ]
    
    print("Creating interaction features...")
    df = create_interaction_features(df, interaction_pairs)
    
    return df









