"""Feature engineering: time, amount, card, address, email, product features."""

import pandas as pd
import numpy as np
from typing import Optional


def create_time_features(df: pd.DataFrame, 
                        time_col: str = 'TransactionDT') -> pd.DataFrame:
    """
    Creates time features: day_of_week, hour, day, week, is_weekend, is_night.
    
    Args:
        df: Input dataframe.
        time_col: Time column name.
    
    Returns:
        DataFrame with time features added.
    """
    df = df.copy()
    
    if time_col not in df.columns:
        return df
    
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
    
    Features created:
    - TransactionAmt_log: Log transform of amount
    - TransactionAmt_bin: Amount bins (10 quantiles)
    - TransactionAmt_percentile: Amount percentile (0-1)
    
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
    
    if amount_col not in df.columns:
        return df
    
    # Log transformation (handle zeros)
    df[f'{amount_col}_log'] = np.log1p(df[amount_col])
    
    # Amount bins (10 quantiles)
    try:
        df[f'{amount_col}_bin'] = pd.qcut(df[amount_col], q=10, labels=False, duplicates='drop')
    except:
        df[f'{amount_col}_bin'] = 0
    
    # Amount percentiles
    df[f'{amount_col}_percentile'] = df[amount_col].rank(pct=True)
    
    return df


def create_card_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Create card-based aggregated features.
    
    Features created:
    - card1_count: Number of transactions per card1
    - card1_avg_amt: Average transaction amount per card1
    - TransactionAmt_to_card1_avg: Ratio of current amount to card1 average
    - card2_freq: Frequency of card2 value
    - card1_card2_combo_count: Count of (card1, card2) combinations
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    is_train : bool
        Whether this is training data
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with card features added
    """
    df = df.copy()
    
    # Card1 features
    if 'card1' in df.columns:
        # Card1 transaction count
        card1_counts = df.groupby('card1').size()
        df['card1_count'] = df['card1'].map(card1_counts)
        
        # Card1 average transaction amount
        if 'TransactionAmt' in df.columns:
            card1_avg_amt = df.groupby('card1')['TransactionAmt'].mean()
            df['card1_avg_amt'] = df['card1'].map(card1_avg_amt)
            df['TransactionAmt_to_card1_avg'] = df['TransactionAmt'] / (df['card1_avg_amt'] + 1e-6)
    
    # Card2 features
    if 'card2' in df.columns:
        # Card2 frequency
        card2_freq = df['card2'].value_counts()
        df['card2_freq'] = df['card2'].map(card2_freq)
    
    # Card1-Card2 combination features
    if 'card1' in df.columns and 'card2' in df.columns:
        # Combination count
        df['card1_card2_combo_count'] = df.groupby(['card1', 'card2']).transform('size')
        
        # Combination average amount
        if 'TransactionAmt' in df.columns:
            df['card1_card2_avg_amt'] = df.groupby(['card1', 'card2'])['TransactionAmt'].transform('mean')
            df['TransactionAmt_to_card1_card2_avg'] = df['TransactionAmt'] / (df['card1_card2_avg_amt'] + 1e-6)
    
    return df


def create_address_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create address-based features.
    
    Features created:
    - addr1_freq: Frequency of addr1 value
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with address features added
    """
    df = df.copy()
    
    if 'addr1' in df.columns:
        # Address frequency
        addr1_freq = df['addr1'].value_counts()
        df['addr1_freq'] = df['addr1'].map(addr1_freq)
    
    return df


def create_email_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create email domain-based features.
    
    Features created:
    - P_emaildomain_freq: Frequency of purchaser email domain
    - R_emaildomain_freq: Frequency of recipient email domain
    - email_match: Whether P and R email domains match
    
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
    
    # Purchaser email domain
    if 'P_emaildomain' in df.columns:
        p_email_freq = df['P_emaildomain'].value_counts()
        df['P_emaildomain_freq'] = df['P_emaildomain'].map(p_email_freq)
    
    # Recipient email domain
    if 'R_emaildomain' in df.columns:
        r_email_freq = df['R_emaildomain'].value_counts()
        df['R_emaildomain_freq'] = df['R_emaildomain'].map(r_email_freq)
    
    # Email domain match
    if 'P_emaildomain' in df.columns and 'R_emaildomain' in df.columns:
        df['email_match'] = (df['P_emaildomain'] == df['R_emaildomain']).astype(int)
    
    return df


def create_product_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create product-based features.
    
    Features created:
    - ProductCD_freq: Frequency of ProductCD value
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with product features added
    """
    df = df.copy()
    
    if 'ProductCD' in df.columns:
        product_freq = df['ProductCD'].value_counts()
        df['ProductCD_freq'] = df['ProductCD'].map(product_freq)
    
    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between selected features.
    
    Features created:
    - card1_ProductCD_count: Count of (card1, ProductCD) combinations
    - card2_ProductCD_count: Count of (card2, ProductCD) combinations
    - addr1_ProductCD_count: Count of (addr1, ProductCD) combinations
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with interaction features added
    """
    df = df.copy()
    
    if 'ProductCD' in df.columns:
        # Card1-ProductCD interaction
        if 'card1' in df.columns:
            df['card1_ProductCD_count'] = df.groupby(['card1', 'ProductCD']).transform('size')
        
        # Card2-ProductCD interaction
        if 'card2' in df.columns:
            df['card2_ProductCD_count'] = df.groupby(['card2', 'ProductCD']).transform('size')
        
        # Addr1-ProductCD interaction
        if 'addr1' in df.columns:
            df['addr1_ProductCD_count'] = df.groupby(['addr1', 'ProductCD']).transform('size')
    
    return df


def create_all_first_view_features(df: pd.DataFrame, 
                                   is_train: bool = True) -> pd.DataFrame:
    """
    Create all features for first view model.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    is_train : bool
        Whether this is training data
        
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
    df = create_card_features(df, is_train=is_train)
    
    print("Creating address features...")
    df = create_address_features(df)
    
    print("Creating email features...")
    df = create_email_features(df)
    
    print("Creating product features...")
    df = create_product_features(df)
    
    print("Creating interaction features...")
    df = create_interaction_features(df)
    
    print(f"Feature engineering complete. Final shape: {df.shape}")
    
    return df






