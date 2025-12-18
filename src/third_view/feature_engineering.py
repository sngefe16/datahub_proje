"""
Feature engineering for third view model.
Extends second view with:
- More triple/quadruple combinations
- Statistical features (mean, std, min, max)
- Lag features (time-based)
"""

import pandas as pd
import numpy as np
from typing import Optional, List


# ===== Second View Features (Reused) =====

def create_time_features(df: pd.DataFrame, 
                        time_col: str = 'TransactionDT') -> pd.DataFrame:
    """Create time-based features from TransactionDT."""
    df = df.copy()
    
    if time_col not in df.columns:
        return df
    
    df['day_of_week'] = (df[time_col] // (24 * 3600)) % 7
    df['hour'] = (df[time_col] // 3600) % 24
    df['day'] = (df[time_col] // (24 * 3600)) % 30
    df['week'] = (df[time_col] // (7 * 24 * 3600)) % 52
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
    
    return df


def create_transaction_amount_features(df: pd.DataFrame,
                                     amount_col: str = 'TransactionAmt') -> pd.DataFrame:
    """Create features from transaction amount."""
    df = df.copy()
    
    if amount_col not in df.columns:
        return df
    
    df[f'{amount_col}_log'] = np.log1p(df[amount_col])
    
    try:
        df[f'{amount_col}_bin'] = pd.qcut(df[amount_col], q=10, labels=False, duplicates='drop')
    except:
        df[f'{amount_col}_bin'] = 0
    
    df[f'{amount_col}_percentile'] = df[amount_col].rank(pct=True)
    
    return df


def create_card_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Create card-based aggregated features."""
    df = df.copy()
    
    if 'card1' in df.columns:
        card1_counts = df.groupby('card1').size()
        df['card1_count'] = df['card1'].map(card1_counts)
        
        if 'TransactionAmt' in df.columns:
            card1_avg_amt = df.groupby('card1')['TransactionAmt'].mean()
            df['card1_avg_amt'] = df['card1'].map(card1_avg_amt)
            df['TransactionAmt_to_card1_avg'] = df['TransactionAmt'] / (df['card1_avg_amt'] + 1e-6)
    
    if 'card2' in df.columns:
        card2_freq = df['card2'].value_counts()
        df['card2_freq'] = df['card2'].map(card2_freq)
    
    if 'card1' in df.columns and 'card2' in df.columns:
        df['card1_card2_combo_count'] = df.groupby(['card1', 'card2']).transform('size')
        
        if 'TransactionAmt' in df.columns:
            df['card1_card2_avg_amt'] = df.groupby(['card1', 'card2'])['TransactionAmt'].transform('mean')
            df['TransactionAmt_to_card1_card2_avg'] = df['TransactionAmt'] / (df['card1_card2_avg_amt'] + 1e-6)
    
    return df


def create_address_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create address-based features."""
    df = df.copy()
    
    if 'addr1' in df.columns:
        addr1_freq = df['addr1'].value_counts()
        df['addr1_freq'] = df['addr1'].map(addr1_freq)
    
    return df


def create_email_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create email domain-based features."""
    df = df.copy()
    
    if 'P_emaildomain' in df.columns:
        p_email_series = df['P_emaildomain'].astype(str) if df['P_emaildomain'].dtype.name == 'category' else df['P_emaildomain']
        p_email_freq = p_email_series.value_counts()
        df['P_emaildomain_freq'] = p_email_series.map(p_email_freq)
    
    if 'R_emaildomain' in df.columns:
        r_email_series = df['R_emaildomain'].astype(str) if df['R_emaildomain'].dtype.name == 'category' else df['R_emaildomain']
        r_email_freq = r_email_series.value_counts()
        df['R_emaildomain_freq'] = r_email_series.map(r_email_freq)
    
    if 'P_emaildomain' in df.columns and 'R_emaildomain' in df.columns:
        p_email = df['P_emaildomain'].astype(str) if df['P_emaildomain'].dtype.name == 'category' else df['P_emaildomain']
        r_email = df['R_emaildomain'].astype(str) if df['R_emaildomain'].dtype.name == 'category' else df['R_emaildomain']
        df['email_match'] = (p_email == r_email).astype(int)
    
    return df


def create_product_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create product-based features."""
    df = df.copy()
    
    if 'ProductCD' in df.columns:
        product_freq = df['ProductCD'].value_counts()
        df['ProductCD_freq'] = df['ProductCD'].map(product_freq)
    
    return df


def create_device_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create device-based features."""
    df = df.copy()
    
    if 'DeviceType' in df.columns:
        device_type_series = df['DeviceType'].astype(str) if df['DeviceType'].dtype.name == 'category' else df['DeviceType']
        device_type_freq = device_type_series.value_counts()
        df['DeviceType_freq'] = device_type_series.map(device_type_freq)
        df['is_mobile'] = (device_type_series == 'mobile').astype(int)
    
    if 'DeviceInfo' in df.columns:
        device_info_freq = df['DeviceInfo'].value_counts()
        df['DeviceInfo_freq'] = df['DeviceInfo'].map(device_info_freq)
    
    if 'DeviceType' in df.columns and 'DeviceInfo' in df.columns:
        df['DeviceType_DeviceInfo_count'] = df.groupby(['DeviceType', 'DeviceInfo']).transform('size')
    
    return df


def create_identity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create identity-based features from id_28, id_29, id_30, id_31."""
    df = df.copy()
    
    id_cols = ['id_28', 'id_29', 'id_30', 'id_31']
    
    for id_col in id_cols:
        if id_col in df.columns:
            id_freq = df[id_col].value_counts()
            df[f'{id_col}_freq'] = df[id_col].map(id_freq)
            
            if f'{id_col}_isMissing' not in df.columns:
                df[f'{id_col}_isMissing'] = df[id_col].isna().astype(int)
    
    if 'id_28' in df.columns and 'id_29' in df.columns:
        df['id_28_id_29_count'] = df.groupby(['id_28', 'id_29']).transform('size')
    
    if 'id_30' in df.columns and 'id_31' in df.columns:
        df['id_30_id_31_count'] = df.groupby(['id_30', 'id_31']).transform('size')
    
    if all(col in df.columns for col in ['id_28', 'id_29', 'id_30', 'id_31']):
        df['id_28_id_29_id_30_id_31_count'] = df.groupby(['id_28', 'id_29', 'id_30', 'id_31']).transform('size')
    
    return df


def create_advanced_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create advanced interaction features (from second view)."""
    df = df.copy()
    
    # === Card-Device Interactions ===
    if 'card1' in df.columns and 'DeviceType' in df.columns:
        device_type_for_groupby = df['DeviceType'].astype(str) if df['DeviceType'].dtype.name == 'category' else df['DeviceType']
        df['card1_DeviceType_count'] = df.groupby(['card1', device_type_for_groupby]).transform('size')
        if 'TransactionAmt' in df.columns:
            df['card1_DeviceType_avg_amt'] = df.groupby(['card1', device_type_for_groupby])['TransactionAmt'].transform('mean').astype(float)
            df['TransactionAmt_to_card1_DeviceType_avg'] = df['TransactionAmt'] / (df['card1_DeviceType_avg_amt'] + 1e-6)
    
    if 'card2' in df.columns and 'DeviceType' in df.columns:
        device_type_for_groupby = df['DeviceType'].astype(str) if df['DeviceType'].dtype.name == 'category' else df['DeviceType']
        df['card2_DeviceType_count'] = df.groupby(['card2', device_type_for_groupby]).transform('size')
    
    if 'card1' in df.columns and 'DeviceInfo' in df.columns:
        device_info_for_groupby = df['DeviceInfo'].astype(str) if df['DeviceInfo'].dtype.name == 'category' else df['DeviceInfo']
        df['card1_DeviceInfo_count'] = df.groupby(['card1', device_info_for_groupby]).transform('size')
    
    # === Card-Identity Interactions ===
    if 'card1' in df.columns and 'id_30' in df.columns:
        df['card1_id_30_count'] = df.groupby(['card1', 'id_30']).transform('size')
    
    if 'card1' in df.columns and 'id_31' in df.columns:
        df['card1_id_31_count'] = df.groupby(['card1', 'id_31']).transform('size')
    
    # === Device-Product Interactions ===
    if 'DeviceType' in df.columns and 'ProductCD' in df.columns:
        device_type_for_groupby = df['DeviceType'].astype(str) if df['DeviceType'].dtype.name == 'category' else df['DeviceType']
        product_for_groupby = df['ProductCD'].astype(str) if df['ProductCD'].dtype.name == 'category' else df['ProductCD']
        df['DeviceType_ProductCD_count'] = df.groupby([device_type_for_groupby, product_for_groupby]).transform('size')
        if 'TransactionAmt' in df.columns:
            df['DeviceType_ProductCD_avg_amt'] = df.groupby([device_type_for_groupby, product_for_groupby])['TransactionAmt'].transform('mean').astype(float)
    
    if 'DeviceInfo' in df.columns and 'ProductCD' in df.columns:
        device_info_for_groupby = df['DeviceInfo'].astype(str) if df['DeviceInfo'].dtype.name == 'category' else df['DeviceInfo']
        product_for_groupby = df['ProductCD'].astype(str) if df['ProductCD'].dtype.name == 'category' else df['ProductCD']
        df['DeviceInfo_ProductCD_count'] = df.groupby([device_info_for_groupby, product_for_groupby]).transform('size')
    
    # === Email-Device Interactions ===
    if 'P_emaildomain' in df.columns and 'DeviceType' in df.columns:
        p_email_for_groupby = df['P_emaildomain'].astype(str) if df['P_emaildomain'].dtype.name == 'category' else df['P_emaildomain']
        device_type_for_groupby = df['DeviceType'].astype(str) if df['DeviceType'].dtype.name == 'category' else df['DeviceType']
        df['P_emaildomain_DeviceType_count'] = df.groupby([p_email_for_groupby, device_type_for_groupby]).transform('size')
    
    if 'R_emaildomain' in df.columns and 'DeviceType' in df.columns:
        r_email_for_groupby = df['R_emaildomain'].astype(str) if df['R_emaildomain'].dtype.name == 'category' else df['R_emaildomain']
        device_type_for_groupby = df['DeviceType'].astype(str) if df['DeviceType'].dtype.name == 'category' else df['DeviceType']
        df['R_emaildomain_DeviceType_count'] = df.groupby([r_email_for_groupby, device_type_for_groupby]).transform('size')
    
    # === Time-Device Interactions ===
    if 'hour' in df.columns and 'DeviceType' in df.columns:
        device_type_for_groupby = df['DeviceType'].astype(str) if df['DeviceType'].dtype.name == 'category' else df['DeviceType']
        df['hour_DeviceType_count'] = df.groupby(['hour', device_type_for_groupby]).transform('size')
    
    if 'is_weekend' in df.columns and 'DeviceType' in df.columns:
        device_type_for_groupby = df['DeviceType'].astype(str) if df['DeviceType'].dtype.name == 'category' else df['DeviceType']
        df['is_weekend_DeviceType_count'] = df.groupby(['is_weekend', device_type_for_groupby]).transform('size')
    
    if 'is_night' in df.columns and 'DeviceType' in df.columns:
        device_type_for_groupby = df['DeviceType'].astype(str) if df['DeviceType'].dtype.name == 'category' else df['DeviceType']
        df['is_night_DeviceType_count'] = df.groupby(['is_night', device_type_for_groupby]).transform('size')
    
    # === Amount-Device Interactions ===
    if 'TransactionAmt' in df.columns and 'DeviceType' in df.columns:
        device_type_for_groupby = df['DeviceType'].astype(str) if df['DeviceType'].dtype.name == 'category' else df['DeviceType']
        device_avg_amt = df.groupby(device_type_for_groupby)['TransactionAmt'].mean()
        df['DeviceType_avg_amt'] = device_type_for_groupby.map(device_avg_amt).astype(float)
        df['TransactionAmt_to_DeviceType_avg'] = df['TransactionAmt'] / (df['DeviceType_avg_amt'] + 1e-6)
    
    # === Identity-Product Interactions ===
    if 'id_30' in df.columns and 'ProductCD' in df.columns:
        df['id_30_ProductCD_count'] = df.groupby(['id_30', 'ProductCD']).transform('size')
    
    if 'id_31' in df.columns and 'ProductCD' in df.columns:
        df['id_31_ProductCD_count'] = df.groupby(['id_31', 'ProductCD']).transform('size')
    
    # === Address-Device Interactions ===
    if 'addr1' in df.columns and 'DeviceType' in df.columns:
        device_type_for_groupby = df['DeviceType'].astype(str) if df['DeviceType'].dtype.name == 'category' else df['DeviceType']
        df['addr1_DeviceType_count'] = df.groupby(['addr1', device_type_for_groupby]).transform('size')
    
    # === Triple Combinations ===
    if all(col in df.columns for col in ['card1', 'DeviceType', 'ProductCD']):
        device_type_for_groupby = df['DeviceType'].astype(str) if df['DeviceType'].dtype.name == 'category' else df['DeviceType']
        product_for_groupby = df['ProductCD'].astype(str) if df['ProductCD'].dtype.name == 'category' else df['ProductCD']
        df['card1_DeviceType_ProductCD_count'] = df.groupby(['card1', device_type_for_groupby, product_for_groupby]).transform('size')
    
    if all(col in df.columns for col in ['card1', 'id_30', 'DeviceType']):
        device_type_for_groupby = df['DeviceType'].astype(str) if df['DeviceType'].dtype.name == 'category' else df['DeviceType']
        df['card1_id_30_DeviceType_count'] = df.groupby(['card1', 'id_30', device_type_for_groupby]).transform('size')
    
    return df


def create_basic_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create basic interaction features (from first view)."""
    df = df.copy()
    
    if 'ProductCD' in df.columns:
        if 'card1' in df.columns:
            df['card1_ProductCD_count'] = df.groupby(['card1', 'ProductCD']).transform('size')
        
        if 'card2' in df.columns:
            df['card2_ProductCD_count'] = df.groupby(['card2', 'ProductCD']).transform('size')
        
        if 'addr1' in df.columns:
            df['addr1_ProductCD_count'] = df.groupby(['addr1', 'ProductCD']).transform('size')
    
    return df


# ===== New Features for Third View =====

def create_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create statistical features (mean, std, min, max) for various groupings.
    
    Features created:
    - Card-based statistics (card1, card2, card1-card2)
    - Device-based statistics (DeviceType, DeviceInfo)
    - Identity-based statistics (id_28, id_29, id_30, id_31)
    - Product-based statistics (ProductCD)
    - Email-based statistics (P_emaildomain, R_emaildomain)
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with statistical features added
    """
    df = df.copy()
    
    if 'TransactionAmt' not in df.columns:
        return df
    
    # === Card-based Statistics ===
    if 'card1' in df.columns:
        card1_stats = df.groupby('card1')['TransactionAmt'].agg(['mean', 'std', 'min', 'max', 'median'])
        df['card1_amt_mean'] = df['card1'].map(card1_stats['mean'])
        df['card1_amt_std'] = df['card1'].map(card1_stats['std']).fillna(0)
        df['card1_amt_min'] = df['card1'].map(card1_stats['min'])
        df['card1_amt_max'] = df['card1'].map(card1_stats['max'])
        df['card1_amt_median'] = df['card1'].map(card1_stats['median'])
        
        # Anomaly features
        df['TransactionAmt_vs_card1_mean'] = (df['TransactionAmt'] - df['card1_amt_mean']) / (df['card1_amt_std'] + 1e-6)
        df['TransactionAmt_vs_card1_min'] = df['TransactionAmt'] - df['card1_amt_min']
        df['TransactionAmt_vs_card1_max'] = df['TransactionAmt'] - df['card1_amt_max']
    
    if 'card2' in df.columns:
        card2_stats = df.groupby('card2')['TransactionAmt'].agg(['mean', 'std', 'min', 'max'])
        df['card2_amt_mean'] = df['card2'].map(card2_stats['mean'])
        df['card2_amt_std'] = df['card2'].map(card2_stats['std']).fillna(0)
        df['card2_amt_min'] = df['card2'].map(card2_stats['min'])
        df['card2_amt_max'] = df['card2'].map(card2_stats['max'])
    
    if 'card1' in df.columns and 'card2' in df.columns:
        combo_stats = df.groupby(['card1', 'card2'])['TransactionAmt'].agg(['mean', 'std', 'min', 'max'])
        df['card1_card2_amt_mean'] = df.groupby(['card1', 'card2'])['TransactionAmt'].transform('mean')
        df['card1_card2_amt_std'] = df.groupby(['card1', 'card2'])['TransactionAmt'].transform('std').fillna(0)
        df['card1_card2_amt_min'] = df.groupby(['card1', 'card2'])['TransactionAmt'].transform('min')
        df['card1_card2_amt_max'] = df.groupby(['card1', 'card2'])['TransactionAmt'].transform('max')
    
    # === Device-based Statistics ===
    if 'DeviceType' in df.columns:
        device_type_for_groupby = df['DeviceType'].astype(str) if df['DeviceType'].dtype.name == 'category' else df['DeviceType']
        device_stats = df.groupby(device_type_for_groupby)['TransactionAmt'].agg(['mean', 'std', 'min', 'max'])
        df['DeviceType_amt_mean'] = device_type_for_groupby.map(device_stats['mean'])
        df['DeviceType_amt_std'] = device_type_for_groupby.map(device_stats['std']).fillna(0)
        df['DeviceType_amt_min'] = device_type_for_groupby.map(device_stats['min'])
        df['DeviceType_amt_max'] = device_type_for_groupby.map(device_stats['max'])
    
    if 'DeviceInfo' in df.columns:
        device_info_for_groupby = df['DeviceInfo'].astype(str) if df['DeviceInfo'].dtype.name == 'category' else df['DeviceInfo']
        device_info_stats = df.groupby(device_info_for_groupby)['TransactionAmt'].agg(['mean', 'std', 'min', 'max'])
        df['DeviceInfo_amt_mean'] = device_info_for_groupby.map(device_info_stats['mean'])
        df['DeviceInfo_amt_std'] = device_info_for_groupby.map(device_info_stats['std']).fillna(0)
        df['DeviceInfo_amt_min'] = device_info_for_groupby.map(device_info_stats['min'])
        df['DeviceInfo_amt_max'] = device_info_for_groupby.map(device_info_stats['max'])
    
    # === Identity-based Statistics ===
    for id_col in ['id_28', 'id_29', 'id_30', 'id_31']:
        if id_col in df.columns:
            # Only for numeric id columns
            if df[id_col].dtype in ['float64', 'int64', 'float32', 'int32', 'int16', 'int8']:
                id_stats = df.groupby(id_col)['TransactionAmt'].agg(['mean', 'std', 'min', 'max'])
                df[f'{id_col}_amt_mean'] = df[id_col].map(id_stats['mean'])
                df[f'{id_col}_amt_std'] = df[id_col].map(id_stats['std']).fillna(0)
                df[f'{id_col}_amt_min'] = df[id_col].map(id_stats['min'])
                df[f'{id_col}_amt_max'] = df[id_col].map(id_stats['max'])
    
    # === Product-based Statistics ===
    if 'ProductCD' in df.columns:
        product_for_groupby = df['ProductCD'].astype(str) if df['ProductCD'].dtype.name == 'category' else df['ProductCD']
        product_stats = df.groupby(product_for_groupby)['TransactionAmt'].agg(['mean', 'std', 'min', 'max'])
        df['ProductCD_amt_mean'] = product_for_groupby.map(product_stats['mean'])
        df['ProductCD_amt_std'] = product_for_groupby.map(product_stats['std']).fillna(0)
        df['ProductCD_amt_min'] = product_for_groupby.map(product_stats['min'])
        df['ProductCD_amt_max'] = product_for_groupby.map(product_stats['max'])
    
    # === Email-based Statistics ===
    if 'P_emaildomain' in df.columns:
        p_email_for_groupby = df['P_emaildomain'].astype(str) if df['P_emaildomain'].dtype.name == 'category' else df['P_emaildomain']
        p_email_stats = df.groupby(p_email_for_groupby)['TransactionAmt'].agg(['mean', 'std', 'min', 'max'])
        df['P_emaildomain_amt_mean'] = p_email_for_groupby.map(p_email_stats['mean'])
        df['P_emaildomain_amt_std'] = p_email_for_groupby.map(p_email_stats['std']).fillna(0)
        df['P_emaildomain_amt_min'] = p_email_for_groupby.map(p_email_stats['min'])
        df['P_emaildomain_amt_max'] = p_email_for_groupby.map(p_email_stats['max'])
    
    if 'R_emaildomain' in df.columns:
        r_email_for_groupby = df['R_emaildomain'].astype(str) if df['R_emaildomain'].dtype.name == 'category' else df['R_emaildomain']
        r_email_stats = df.groupby(r_email_for_groupby)['TransactionAmt'].agg(['mean', 'std', 'min', 'max'])
        df['R_emaildomain_amt_mean'] = r_email_for_groupby.map(r_email_stats['mean'])
        df['R_emaildomain_amt_std'] = r_email_for_groupby.map(r_email_stats['std']).fillna(0)
        df['R_emaildomain_amt_min'] = r_email_for_groupby.map(r_email_stats['min'])
        df['R_emaildomain_amt_max'] = r_email_for_groupby.map(r_email_stats['max'])
    
    return df


def create_lag_features(df: pd.DataFrame, 
                       time_col: str = 'TransactionDT',
                       group_cols: List[str] = None) -> pd.DataFrame:
    """
    Create time-based lag features.
    
    Features created:
    - Time since last transaction (by card1, card2, card1-card2, DeviceType, etc.)
    - Time since last transaction of same type (by ProductCD, etc.)
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    time_col : str
        Time column name
    group_cols : list
        List of columns to group by for lag features
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with lag features added
    """
    df = df.copy()
    
    if time_col not in df.columns:
        return df
    
    if group_cols is None:
        group_cols = ['card1', 'card2', 'DeviceType', 'ProductCD']
    
    # Sort by time for lag calculations
    df = df.sort_values(by=time_col).reset_index(drop=True)
    
    # === Card-based Lag Features ===
    if 'card1' in df.columns:
        df['time_since_last_card1'] = df.groupby('card1')[time_col].diff()
        df['time_since_last_card1'] = df['time_since_last_card1'].fillna(0)
        
        # Convert to hours for interpretability
        df['hours_since_last_card1'] = df['time_since_last_card1'] / 3600
        df['days_since_last_card1'] = df['time_since_last_card1'] / (24 * 3600)
    
    if 'card2' in df.columns:
        df['time_since_last_card2'] = df.groupby('card2')[time_col].diff()
        df['time_since_last_card2'] = df['time_since_last_card2'].fillna(0)
        df['hours_since_last_card2'] = df['time_since_last_card2'] / 3600
    
    if 'card1' in df.columns and 'card2' in df.columns:
        df['time_since_last_card1_card2'] = df.groupby(['card1', 'card2'])[time_col].diff()
        df['time_since_last_card1_card2'] = df['time_since_last_card1_card2'].fillna(0)
        df['hours_since_last_card1_card2'] = df['time_since_last_card1_card2'] / 3600
    
    # === Device-based Lag Features ===
    if 'DeviceType' in df.columns:
        device_type_for_groupby = df['DeviceType'].astype(str) if df['DeviceType'].dtype.name == 'category' else df['DeviceType']
        df['time_since_last_DeviceType'] = df.groupby(device_type_for_groupby)[time_col].diff()
        df['time_since_last_DeviceType'] = df['time_since_last_DeviceType'].fillna(0)
        df['hours_since_last_DeviceType'] = df['time_since_last_DeviceType'] / 3600
    
    # === Product-based Lag Features ===
    if 'ProductCD' in df.columns:
        product_for_groupby = df['ProductCD'].astype(str) if df['ProductCD'].dtype.name == 'category' else df['ProductCD']
        df['time_since_last_ProductCD'] = df.groupby(product_for_groupby)[time_col].diff()
        df['time_since_last_ProductCD'] = df['time_since_last_ProductCD'].fillna(0)
        df['hours_since_last_ProductCD'] = df['time_since_last_ProductCD'] / 3600
    
    # === Card-Device Lag Features ===
    if 'card1' in df.columns and 'DeviceType' in df.columns:
        device_type_for_groupby = df['DeviceType'].astype(str) if df['DeviceType'].dtype.name == 'category' else df['DeviceType']
        df['time_since_last_card1_DeviceType'] = df.groupby(['card1', device_type_for_groupby])[time_col].diff()
        df['time_since_last_card1_DeviceType'] = df['time_since_last_card1_DeviceType'].fillna(0)
        df['hours_since_last_card1_DeviceType'] = df['time_since_last_card1_DeviceType'] / 3600
    
    # === Card-Product Lag Features ===
    if 'card1' in df.columns and 'ProductCD' in df.columns:
        product_for_groupby = df['ProductCD'].astype(str) if df['ProductCD'].dtype.name == 'category' else df['ProductCD']
        df['time_since_last_card1_ProductCD'] = df.groupby(['card1', product_for_groupby])[time_col].diff()
        df['time_since_last_card1_ProductCD'] = df['time_since_last_card1_ProductCD'].fillna(0)
        df['hours_since_last_card1_ProductCD'] = df['time_since_last_card1_ProductCD'] / 3600
    
    return df


def create_extended_triple_combinations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create extended triple and quadruple combinations.
    
    Features created:
    - Triple combinations: Card-Device-Product, Card-Identity-Device, etc.
    - Quadruple combinations: Card-Device-Product-Email, etc.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with extended combinations added
    """
    df = df.copy()
    
    # === Triple Combinations ===
    # Card-Device-Product
    if all(col in df.columns for col in ['card1', 'DeviceType', 'ProductCD']):
        device_type_for_groupby = df['DeviceType'].astype(str) if df['DeviceType'].dtype.name == 'category' else df['DeviceType']
        product_for_groupby = df['ProductCD'].astype(str) if df['ProductCD'].dtype.name == 'category' else df['ProductCD']
        df['card1_DeviceType_ProductCD_count'] = df.groupby(['card1', device_type_for_groupby, product_for_groupby]).transform('size')
    
    # Card-Identity-Device
    if all(col in df.columns for col in ['card1', 'id_30', 'DeviceType']):
        device_type_for_groupby = df['DeviceType'].astype(str) if df['DeviceType'].dtype.name == 'category' else df['DeviceType']
        df['card1_id_30_DeviceType_count'] = df.groupby(['card1', 'id_30', device_type_for_groupby]).transform('size')
    
    if all(col in df.columns for col in ['card1', 'id_31', 'DeviceType']):
        device_type_for_groupby = df['DeviceType'].astype(str) if df['DeviceType'].dtype.name == 'category' else df['DeviceType']
        df['card1_id_31_DeviceType_count'] = df.groupby(['card1', 'id_31', device_type_for_groupby]).transform('size')
    
    # Card-Email-Device
    if all(col in df.columns for col in ['card1', 'P_emaildomain', 'DeviceType']):
        p_email_for_groupby = df['P_emaildomain'].astype(str) if df['P_emaildomain'].dtype.name == 'category' else df['P_emaildomain']
        device_type_for_groupby = df['DeviceType'].astype(str) if df['DeviceType'].dtype.name == 'category' else df['DeviceType']
        df['card1_P_emaildomain_DeviceType_count'] = df.groupby(['card1', p_email_for_groupby, device_type_for_groupby]).transform('size')
    
    # Card-Time-Device
    if all(col in df.columns for col in ['card1', 'hour', 'DeviceType']):
        device_type_for_groupby = df['DeviceType'].astype(str) if df['DeviceType'].dtype.name == 'category' else df['DeviceType']
        df['card1_hour_DeviceType_count'] = df.groupby(['card1', 'hour', device_type_for_groupby]).transform('size')
    
    if all(col in df.columns for col in ['card1', 'is_weekend', 'DeviceType']):
        device_type_for_groupby = df['DeviceType'].astype(str) if df['DeviceType'].dtype.name == 'category' else df['DeviceType']
        df['card1_is_weekend_DeviceType_count'] = df.groupby(['card1', 'is_weekend', device_type_for_groupby]).transform('size')
    
    # Device-Product-Email
    if all(col in df.columns for col in ['DeviceType', 'ProductCD', 'P_emaildomain']):
        device_type_for_groupby = df['DeviceType'].astype(str) if df['DeviceType'].dtype.name == 'category' else df['DeviceType']
        product_for_groupby = df['ProductCD'].astype(str) if df['ProductCD'].dtype.name == 'category' else df['ProductCD']
        p_email_for_groupby = df['P_emaildomain'].astype(str) if df['P_emaildomain'].dtype.name == 'category' else df['P_emaildomain']
        df['DeviceType_ProductCD_P_emaildomain_count'] = df.groupby([device_type_for_groupby, product_for_groupby, p_email_for_groupby]).transform('size')
    
    # === Quadruple Combinations ===
    # Card-Device-Product-Email
    if all(col in df.columns for col in ['card1', 'DeviceType', 'ProductCD', 'P_emaildomain']):
        device_type_for_groupby = df['DeviceType'].astype(str) if df['DeviceType'].dtype.name == 'category' else df['DeviceType']
        product_for_groupby = df['ProductCD'].astype(str) if df['ProductCD'].dtype.name == 'category' else df['ProductCD']
        p_email_for_groupby = df['P_emaildomain'].astype(str) if df['P_emaildomain'].dtype.name == 'category' else df['P_emaildomain']
        df['card1_DeviceType_ProductCD_P_emaildomain_count'] = df.groupby(['card1', device_type_for_groupby, product_for_groupby, p_email_for_groupby]).transform('size')
    
    # Card-Device-Identity-Product
    if all(col in df.columns for col in ['card1', 'DeviceType', 'id_30', 'ProductCD']):
        device_type_for_groupby = df['DeviceType'].astype(str) if df['DeviceType'].dtype.name == 'category' else df['DeviceType']
        product_for_groupby = df['ProductCD'].astype(str) if df['ProductCD'].dtype.name == 'category' else df['ProductCD']
        df['card1_DeviceType_id_30_ProductCD_count'] = df.groupby(['card1', device_type_for_groupby, 'id_30', product_for_groupby]).transform('size')
    
    # Card-Device-Time-Product
    if all(col in df.columns for col in ['card1', 'DeviceType', 'hour', 'ProductCD']):
        device_type_for_groupby = df['DeviceType'].astype(str) if df['DeviceType'].dtype.name == 'category' else df['DeviceType']
        product_for_groupby = df['ProductCD'].astype(str) if df['ProductCD'].dtype.name == 'category' else df['ProductCD']
        df['card1_DeviceType_hour_ProductCD_count'] = df.groupby(['card1', device_type_for_groupby, 'hour', product_for_groupby]).transform('size')
    
    return df


def create_all_third_view_features(df: pd.DataFrame, 
                                   is_train: bool = True) -> pd.DataFrame:
    """
    Create all features for third view model.
    
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
    
    print("Creating device features...")
    df = create_device_features(df)
    
    print("Creating identity features...")
    df = create_identity_features(df)
    
    print("Creating basic interaction features...")
    df = create_basic_interaction_features(df)
    
    print("Creating advanced interaction features...")
    df = create_advanced_interaction_features(df)
    
    print("Creating statistical features...")
    df = create_statistical_features(df)
    
    print("Creating lag features...")
    df = create_lag_features(df)
    
    print("Creating extended triple/quadruple combinations...")
    df = create_extended_triple_combinations(df)
    
    print(f"Feature engineering complete. Final shape: {df.shape}")
    
    return df

