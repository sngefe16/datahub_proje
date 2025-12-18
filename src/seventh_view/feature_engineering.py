"""
Feature engineering for seventh view model.
Extends sixth view with same features (UID, card, C, dist features).
Based on sixth view improvements from MODEL_COMPARISON2.md.
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
    """
    Create card-based aggregated features for ALL cards (card1-card6).
    Enhanced for sixth view with all card features.
    """
    df = df.copy()
    
    # Card1 features (existing)
    if 'card1' in df.columns:
        card1_counts = df.groupby('card1').size()
        df['card1_count'] = df['card1'].map(card1_counts)
        
        if 'TransactionAmt' in df.columns:
            card1_avg_amt = df.groupby('card1')['TransactionAmt'].mean()
            df['card1_avg_amt'] = df['card1'].map(card1_avg_amt)
            df['TransactionAmt_to_card1_avg'] = df['TransactionAmt'] / (df['card1_avg_amt'] + 1e-6)
    
    # Card2 features (existing)
    if 'card2' in df.columns:
        card2_freq = df['card2'].value_counts()
        df['card2_freq'] = df['card2'].map(card2_freq)
    
    # Card3 features (NEW!)
    if 'card3' in df.columns:
        card3_freq = df['card3'].value_counts()
        df['card3_freq'] = df['card3'].map(card3_freq)
        
        if 'TransactionAmt' in df.columns:
            card3_avg_amt = df.groupby('card3')['TransactionAmt'].mean()
            df['card3_avg_amt'] = df['card3'].map(card3_avg_amt)
            df['TransactionAmt_to_card3_avg'] = df['TransactionAmt'] / (df['card3_avg_amt'] + 1e-6)
    
    # Card4 features (NEW!)
    if 'card4' in df.columns:
        card4_freq = df['card4'].value_counts()
        df['card4_freq'] = df['card4'].map(card4_freq)
    
    # Card5 features (NEW!)
    if 'card5' in df.columns:
        card5_freq = df['card5'].value_counts()
        df['card5_freq'] = df['card5'].map(card5_freq)
        
        if 'TransactionAmt' in df.columns:
            card5_avg_amt = df.groupby('card5')['TransactionAmt'].mean()
            df['card5_avg_amt'] = df['card5'].map(card5_avg_amt)
            df['TransactionAmt_to_card5_avg'] = df['TransactionAmt'] / (df['card5_avg_amt'] + 1e-6)
    
    # Card6 features (NEW!)
    if 'card6' in df.columns:
        card6_freq = df['card6'].value_counts()
        df['card6_freq'] = df['card6'].map(card6_freq)
    
    # Card1-Card2 combo (existing)
    if 'card1' in df.columns and 'card2' in df.columns:
        df['card1_card2_combo_count'] = df.groupby(['card1', 'card2']).transform('size')
        
        if 'TransactionAmt' in df.columns:
            df['card1_card2_avg_amt'] = df.groupby(['card1', 'card2'])['TransactionAmt'].transform('mean')
            df['TransactionAmt_to_card1_card2_avg'] = df['TransactionAmt'] / (df['card1_card2_avg_amt'] + 1e-6)
    
    # Card1-Card3 combo (NEW!)
    if 'card1' in df.columns and 'card3' in df.columns:
        df['card1_card3_combo_count'] = df.groupby(['card1', 'card3']).transform('size')
    
    # Card1-Card5 combo (NEW!)
    if 'card1' in df.columns and 'card5' in df.columns:
        df['card1_card5_combo_count'] = df.groupby(['card1', 'card5']).transform('size')
    
    # Card2-Card3 combo (NEW!)
    if 'card2' in df.columns and 'card3' in df.columns:
        df['card2_card3_combo_count'] = df.groupby(['card2', 'card3']).transform('size')
    
    # Triple combinations (NEW!)
    if all(c in df.columns for c in ['card1', 'card2', 'card3']):
        df['card1_card2_card3_count'] = df.groupby(['card1', 'card2', 'card3']).transform('size')
    
    if all(c in df.columns for c in ['card1', 'card3', 'card5']):
        df['card1_card3_card5_count'] = df.groupby(['card1', 'card3', 'card5']).transform('size')
    
    return df


def create_address_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create address-based features for addr1 and addr2.
    Enhanced for sixth view with addr2.
    """
    df = df.copy()
    
    if 'addr1' in df.columns:
        addr1_freq = df['addr1'].value_counts()
        df['addr1_freq'] = df['addr1'].map(addr1_freq)
    
    # Addr2 features (NEW!)
    if 'addr2' in df.columns:
        addr2_freq = df['addr2'].value_counts()
        df['addr2_freq'] = df['addr2'].map(addr2_freq)
    
    # Addr1-Addr2 combo (NEW!)
    if 'addr1' in df.columns and 'addr2' in df.columns:
        df['addr1_addr2_combo_count'] = df.groupby(['addr1', 'addr2']).transform('size')
        
        if 'TransactionAmt' in df.columns:
            df['addr1_addr2_avg_amt'] = df.groupby(['addr1', 'addr2'])['TransactionAmt'].transform('mean')
            df['TransactionAmt_to_addr1_addr2_avg'] = df['TransactionAmt'] / (df['addr1_addr2_avg_amt'] + 1e-6)
    
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


def create_enhanced_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create enhanced statistical features (percentile, quantile, skew, kurtosis).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with enhanced statistical features added
    """
    df = df.copy()
    
    if 'TransactionAmt' not in df.columns:
        return df
    
    # === Card-based Enhanced Statistics ===
    if 'card1' in df.columns:
        card1_grouped = df.groupby('card1')['TransactionAmt']
        
        # Percentiles
        df['card1_amt_p25'] = card1_grouped.transform(lambda x: x.quantile(0.25))
        df['card1_amt_p75'] = card1_grouped.transform(lambda x: x.quantile(0.75))
        df['card1_amt_p90'] = card1_grouped.transform(lambda x: x.quantile(0.90))
        df['card1_amt_p95'] = card1_grouped.transform(lambda x: x.quantile(0.95))
        
        # IQR
        df['card1_amt_iqr'] = df['card1_amt_p75'] - df['card1_amt_p25']
        
        # Relative position
        df['TransactionAmt_vs_card1_p25'] = df['TransactionAmt'] - df['card1_amt_p25']
        df['TransactionAmt_vs_card1_p75'] = df['TransactionAmt'] - df['card1_amt_p75']
        df['TransactionAmt_vs_card1_p95'] = df['TransactionAmt'] - df['card1_amt_p95']
    
    if 'card2' in df.columns:
        card2_grouped = df.groupby('card2')['TransactionAmt']
        df['card2_amt_p25'] = card2_grouped.transform(lambda x: x.quantile(0.25))
        df['card2_amt_p75'] = card2_grouped.transform(lambda x: x.quantile(0.75))
        df['card2_amt_p90'] = card2_grouped.transform(lambda x: x.quantile(0.90))
    
    # === Device-based Enhanced Statistics ===
    if 'DeviceType' in df.columns:
        device_type_for_groupby = df['DeviceType'].astype(str) if df['DeviceType'].dtype.name == 'category' else df['DeviceType']
        device_grouped = df.groupby(device_type_for_groupby)['TransactionAmt']
        df['DeviceType_amt_p25'] = device_grouped.transform(lambda x: x.quantile(0.25))
        df['DeviceType_amt_p75'] = device_grouped.transform(lambda x: x.quantile(0.75))
        df['DeviceType_amt_p90'] = device_grouped.transform(lambda x: x.quantile(0.90))
    
    # === Product-based Enhanced Statistics ===
    if 'ProductCD' in df.columns:
        product_for_groupby = df['ProductCD'].astype(str) if df['ProductCD'].dtype.name == 'category' else df['ProductCD']
        product_grouped = df.groupby(product_for_groupby)['TransactionAmt']
        df['ProductCD_amt_p25'] = product_grouped.transform(lambda x: x.quantile(0.25))
        df['ProductCD_amt_p75'] = product_grouped.transform(lambda x: x.quantile(0.75))
        df['ProductCD_amt_p90'] = product_grouped.transform(lambda x: x.quantile(0.90))
    
    return df


def create_enhanced_lag_features(df: pd.DataFrame,
                                 time_col: str = 'TransactionDT') -> pd.DataFrame:
    """
    Create enhanced lag features with rolling windows and multiple time windows.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    time_col : str
        Time column name
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with enhanced lag features added
    """
    df = df.copy()
    
    if time_col not in df.columns:
        return df
    
    # Sort by time for lag calculations
    df = df.sort_values(by=time_col).reset_index(drop=True)
    
    # === Rolling Window Features (Last N transactions) - HIGHLY OPTIMIZED ===
    if 'card1' in df.columns:
        # Optimized: Use cumcount instead of rolling (much faster - vectorized)
        # Count of transactions in last N transactions using cumcount
        card1_groups = df.groupby('card1')
        card1_cumcount = card1_groups.cumcount()  # 0-indexed position within each card1 group
        
        for window_hours, n_transactions in [(1, 5), (6, 20), (24, 50)]:
            # For each row, count how many transactions occurred in the last N transactions
            # Using cumcount: if position >= N, count is N; otherwise position+1
            # Shift by 1 to exclude current transaction
            shifted_pos = card1_groups.cumcount().shift(1).fillna(-1)
            # Count = min(N, shifted_pos + 1) but only if shifted_pos >= 0
            df[f'card1_count_last_{window_hours}h'] = (shifted_pos + 1).clip(upper=n_transactions)
            df[f'card1_count_last_{window_hours}h'] = df[f'card1_count_last_{window_hours}h'].where(shifted_pos >= 0, 0)
        
        # Average amount in last N transactions - use expanding mean (already fast)
        if 'TransactionAmt' in df.columns:
            # Use expanding mean of previous transactions (shifted by 1 to exclude current)
            # This is already optimized and fast
            df['card1_avg_amt_expanding'] = card1_groups['TransactionAmt'].transform(
                lambda x: x.shift(1).expanding(min_periods=1).mean()
            )
            # Fill NaN (first transaction of each card) with current transaction amount
            df['card1_avg_amt_expanding'] = df['card1_avg_amt_expanding'].fillna(df['TransactionAmt'])
    
    # === Multiple Time Window Lag Features ===
    if 'card1' in df.columns:
        # Time since last transaction (already exists, but add more windows)
        df['time_since_last_card1'] = df.groupby('card1')[time_col].diff().fillna(0)
        
        # Time since second-to-last transaction
        df['time_since_second_last_card1'] = df.groupby('card1')[time_col].diff().shift(1).fillna(0)
        
        # Time between last two transactions
        df['time_between_last_two_card1'] = df['time_since_last_card1'] - df['time_since_second_last_card1']
    
    # === Device-based Enhanced Lag - HIGHLY OPTIMIZED ===
    if 'DeviceType' in df.columns:
        device_type_for_groupby = df['DeviceType'].astype(str) if df['DeviceType'].dtype.name == 'category' else df['DeviceType']
        df['time_since_last_DeviceType'] = df.groupby(device_type_for_groupby)[time_col].diff().fillna(0)
        
        # Count in last N transactions - use cumcount instead of rolling (much faster)
        device_groups = df.groupby(device_type_for_groupby)
        
        for window_hours, n_transactions in [(6, 20), (24, 50)]:
            # Use cumcount with shift (vectorized, much faster than rolling)
            shifted_pos = device_groups.cumcount().shift(1).fillna(-1)
            df[f'DeviceType_count_last_{window_hours}h'] = (shifted_pos + 1).clip(upper=n_transactions)
            df[f'DeviceType_count_last_{window_hours}h'] = df[f'DeviceType_count_last_{window_hours}h'].where(shifted_pos >= 0, 0)
    
    return df


def create_advanced_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create advanced statistical features (skew, kurtosis, coefficient of variation).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with advanced statistical features added
    """
    df = df.copy()
    
    if 'TransactionAmt' not in df.columns:
        return df
    
    # === Card-based Advanced Statistics ===
    if 'card1' in df.columns:
        card1_grouped = df.groupby('card1')['TransactionAmt']
        
        # Coefficient of Variation (CV = std/mean)
        df['card1_amt_cv'] = (card1_grouped.transform('std') / (card1_grouped.transform('mean') + 1e-6)).fillna(0)
        
        # Range (max - min)
        df['card1_amt_range'] = card1_grouped.transform('max') - card1_grouped.transform('min')
        
        # Relative amount (current / mean)
        df['TransactionAmt_to_card1_mean_ratio'] = df['TransactionAmt'] / (card1_grouped.transform('mean') + 1e-6)
    
    # === Device-based Advanced Statistics ===
    if 'DeviceType' in df.columns:
        device_type_for_groupby = df['DeviceType'].astype(str) if df['DeviceType'].dtype.name == 'category' else df['DeviceType']
        device_grouped = df.groupby(device_type_for_groupby)['TransactionAmt']
        df['DeviceType_amt_cv'] = (device_grouped.transform('std') / (device_grouped.transform('mean') + 1e-6)).fillna(0)
        df['DeviceType_amt_range'] = device_grouped.transform('max') - device_grouped.transform('min')
        df['DeviceType_amt_median'] = device_grouped.transform('median')
        df['TransactionAmt_to_DeviceType_median_ratio'] = df['TransactionAmt'] / (df['DeviceType_amt_median'] + 1e-6)
    
    # === Product-based Advanced Statistics ===
    if 'ProductCD' in df.columns:
        product_for_groupby = df['ProductCD'].astype(str) if df['ProductCD'].dtype.name == 'category' else df['ProductCD']
        product_grouped = df.groupby(product_for_groupby)['TransactionAmt']
        df['ProductCD_amt_cv'] = (product_grouped.transform('std') / (product_grouped.transform('mean') + 1e-6)).fillna(0)
        df['ProductCD_amt_range'] = product_grouped.transform('max') - product_grouped.transform('min')
        df['ProductCD_amt_median'] = product_grouped.transform('median')
        df['TransactionAmt_to_ProductCD_median_ratio'] = df['TransactionAmt'] / (df['ProductCD_amt_median'] + 1e-6)
    
    # === Email-based Advanced Statistics ===
    if 'P_emaildomain' in df.columns:
        p_email_series = df['P_emaildomain'].astype(str) if df['P_emaildomain'].dtype.name == 'category' else df['P_emaildomain']
        email_grouped = df.groupby(p_email_series)['TransactionAmt']
        df['P_emaildomain_amt_cv'] = (email_grouped.transform('std') / (email_grouped.transform('mean') + 1e-6)).fillna(0)
        df['P_emaildomain_amt_range'] = email_grouped.transform('max') - email_grouped.transform('min')
        df['P_emaildomain_amt_median'] = email_grouped.transform('median')
        df['TransactionAmt_to_P_emaildomain_median_ratio'] = df['TransactionAmt'] / (df['P_emaildomain_amt_median'] + 1e-6)
    
    if 'R_emaildomain' in df.columns:
        r_email_series = df['R_emaildomain'].astype(str) if df['R_emaildomain'].dtype.name == 'category' else df['R_emaildomain']
        email_grouped = df.groupby(r_email_series)['TransactionAmt']
        df['R_emaildomain_amt_cv'] = (email_grouped.transform('std') / (email_grouped.transform('mean') + 1e-6)).fillna(0)
        df['R_emaildomain_amt_range'] = email_grouped.transform('max') - email_grouped.transform('min')
        df['R_emaildomain_amt_median'] = email_grouped.transform('median')
        df['TransactionAmt_to_R_emaildomain_median_ratio'] = df['TransactionAmt'] / (df['R_emaildomain_amt_median'] + 1e-6)
    
    # === Card2 Advanced Statistics ===
    if 'card2' in df.columns:
        card2_grouped = df.groupby('card2')['TransactionAmt']
        df['card2_amt_cv'] = (card2_grouped.transform('std') / (card2_grouped.transform('mean') + 1e-6)).fillna(0)
        df['card2_amt_range'] = card2_grouped.transform('max') - card2_grouped.transform('min')
        df['TransactionAmt_to_card2_mean_ratio'] = df['TransactionAmt'] / (card2_grouped.transform('mean') + 1e-6)
        df['TransactionAmt_to_card2_median_ratio'] = df['TransactionAmt'] / (card2_grouped.transform('median') + 1e-6)
    
    return df


def create_advanced_lag_features_v2(df: pd.DataFrame,
                                     time_col: str = 'TransactionDT') -> pd.DataFrame:
    """
    Create additional advanced lag features (velocity, acceleration, trend).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    time_col : str
        Time column name
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with advanced lag features added
    """
    df = df.copy()
    
    if time_col not in df.columns:
        return df
    
    # Sort by time
    df = df.sort_values(by=time_col).reset_index(drop=True)
    
    # === Velocity Features (rate of change) ===
    if 'card1' in df.columns and 'TransactionAmt' in df.columns:
        # Amount velocity (change in amount per unit time)
        df['card1_amt_velocity'] = df.groupby('card1')['TransactionAmt'].diff() / (df.groupby('card1')[time_col].diff() + 1e-6)
        df['card1_amt_velocity'] = df['card1_amt_velocity'].fillna(0)
        
        # Time velocity (inverse of time since last transaction)
        df['card1_time_velocity'] = 1.0 / (df.groupby('card1')[time_col].diff() + 1e-6)
        df['card1_time_velocity'] = df['card1_time_velocity'].fillna(0)
    
    # === Acceleration Features (rate of change of velocity) ===
    if 'card1' in df.columns and 'card1_amt_velocity' in df.columns:
        df['card1_amt_acceleration'] = df.groupby('card1')['card1_amt_velocity'].diff().fillna(0)
    
    # === Trend Features ===
    if 'card1' in df.columns and 'TransactionAmt' in df.columns:
        # Rolling mean trend (increasing/decreasing)
        card1_groups = df.groupby('card1')
        rolling_mean = card1_groups['TransactionAmt'].transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
        df['card1_amt_trend'] = (df['TransactionAmt'] - rolling_mean).fillna(0)
        df['card1_amt_trend_positive'] = (df['card1_amt_trend'] > 0).astype(int)
        
        # Additional trend features
        # Trend strength (absolute change)
        df['card1_amt_trend_strength'] = df['card1_amt_trend'].abs()
        
        # Trend direction change (acceleration)
        df['card1_amt_trend_change'] = card1_groups['card1_amt_trend'].diff().fillna(0)
    
    # === Additional Velocity Features ===
    if 'card2' in df.columns and 'TransactionAmt' in df.columns and time_col in df.columns:
        df['card2_amt_velocity'] = df.groupby('card2')['TransactionAmt'].diff() / (df.groupby('card2')[time_col].diff() + 1e-6)
        df['card2_amt_velocity'] = df['card2_amt_velocity'].fillna(0)
        df['card2_time_velocity'] = 1.0 / (df.groupby('card2')[time_col].diff() + 1e-6)
        df['card2_time_velocity'] = df['card2_time_velocity'].fillna(0)
    
    # === Additional Acceleration Features ===
    if 'card1_amt_velocity' in df.columns:
        # Second derivative (acceleration of acceleration)
        df['card1_amt_jerk'] = df.groupby('card1')['card1_amt_acceleration'].diff().fillna(0)
    
    # === Time-based Velocity Features ===
    if 'card1' in df.columns and time_col in df.columns:
        # Time between transactions (inverse velocity)
        time_diff = df.groupby('card1')[time_col].diff()
        df['card1_time_between'] = time_diff.fillna(0)
        df['card1_time_between_normalized'] = time_diff / (time_diff.mean() + 1e-6)
    
    return df


def create_domain_specific_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-specific features for fraud detection.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with domain-specific features added
    """
    df = df.copy()
    
    # === Risk Score Features ===
    if 'TransactionAmt' in df.columns:
        # High amount transactions (potential risk)
        df['is_high_amount'] = (df['TransactionAmt'] > df['TransactionAmt'].quantile(0.95)).astype(int)
        df['is_low_amount'] = (df['TransactionAmt'] < df['TransactionAmt'].quantile(0.05)).astype(int)
    
    # === Time-based Risk Features ===
    if 'hour' in df.columns:
        # Unusual hours (potential risk)
        df['is_unusual_hour'] = ((df['hour'] >= 2) & (df['hour'] <= 5)).astype(int)
    
    if 'day_of_week' in df.columns:
        # Weekend transactions
        df['is_weekend_transaction'] = (df['day_of_week'] >= 5).astype(int)
    
    # === Combination Risk Features ===
    if 'is_high_amount' in df.columns and 'is_unusual_hour' in df.columns:
        df['high_amount_unusual_hour'] = (df['is_high_amount'] & df['is_unusual_hour']).astype(int)
    
    if 'is_high_amount' in df.columns and 'is_weekend_transaction' in df.columns:
        df['high_amount_weekend'] = (df['is_high_amount'] & df['is_weekend_transaction']).astype(int)
    
    # === Frequency-based Risk Features ===
    if 'card1_count' in df.columns:
        # New card (low count) or very active card (high count)
        df['card1_is_new'] = (df['card1_count'] <= 1).astype(int)
        df['card1_is_very_active'] = (df['card1_count'] > df['card1_count'].quantile(0.95)).astype(int)
    
    # === Email Domain Risk Features ===
    if 'P_emaildomain' in df.columns:
        p_email_series = df['P_emaildomain'].astype(str) if df['P_emaildomain'].dtype.name == 'category' else df['P_emaildomain']
        # Rare email domains (potential risk)
        email_counts = p_email_series.value_counts()
        df['P_emaildomain_is_rare'] = (p_email_series.map(email_counts) <= 10).astype(int)
        df['P_emaildomain_freq_rank'] = p_email_series.map(email_counts).rank(pct=True)
    
    if 'R_emaildomain' in df.columns:
        r_email_series = df['R_emaildomain'].astype(str) if df['R_emaildomain'].dtype.name == 'category' else df['R_emaildomain']
        email_counts = r_email_series.value_counts()
        df['R_emaildomain_is_rare'] = (r_email_series.map(email_counts) <= 10).astype(int)
        df['R_emaildomain_freq_rank'] = r_email_series.map(email_counts).rank(pct=True)
    
    # === Device Risk Features ===
    if 'DeviceType' in df.columns:
        device_type_for_groupby = df['DeviceType'].astype(str) if df['DeviceType'].dtype.name == 'category' else df['DeviceType']
        device_counts = device_type_for_groupby.value_counts()
        df['DeviceType_is_rare'] = (device_type_for_groupby.map(device_counts) <= 50).astype(int)
    
    if 'DeviceInfo' in df.columns:
        device_info_for_groupby = df['DeviceInfo'].astype(str) if df['DeviceInfo'].dtype.name == 'category' else df['DeviceInfo']
        device_info_counts = device_info_for_groupby.value_counts()
        df['DeviceInfo_is_rare'] = (device_info_for_groupby.map(device_info_counts) <= 20).astype(int)
    
    # === Card Risk Features ===
    if 'card1' in df.columns and 'card1_count' in df.columns:
        # Card activity risk
        df['card1_activity_risk'] = (
            (df['card1_count'] <= 1).astype(int) * 2 +  # New card (high risk)
            (df['card1_count'] > df['card1_count'].quantile(0.99)).astype(int)  # Very active (potential risk)
        )
    
    if 'card2' in df.columns and 'card2_freq' in df.columns:
        df['card2_activity_risk'] = (
            (df['card2_freq'] <= 1).astype(int) * 2 +
            (df['card2_freq'] > df['card2_freq'].quantile(0.99)).astype(int)
        )
    
    # === Time-based Risk Combinations ===
    if 'is_unusual_hour' in df.columns and 'is_high_amount' in df.columns:
        df['unusual_hour_high_amount_risk'] = (
            df['is_unusual_hour'] * df['is_high_amount']
        )
    
    if 'is_weekend_transaction' in df.columns and 'is_high_amount' in df.columns:
        df['weekend_high_amount_risk'] = (
            df['is_weekend_transaction'] * df['is_high_amount']
        )
    
    # === Identity Risk Features ===
    if 'id_28' in df.columns:
        id_28_series = df['id_28'].astype(str) if df['id_28'].dtype.name == 'category' else df['id_28']
        id_28_counts = id_28_series.value_counts()
        df['id_28_is_rare'] = (id_28_series.map(id_28_counts) <= 10).astype(int)
    
    if 'id_30' in df.columns:
        id_30_series = df['id_30'].astype(str) if df['id_30'].dtype.name == 'category' else df['id_30']
        id_30_counts = id_30_series.value_counts()
        df['id_30_is_rare'] = (id_30_series.map(id_30_counts) <= 10).astype(int)
    
    if 'id_31' in df.columns:
        id_31_series = df['id_31'].astype(str) if df['id_31'].dtype.name == 'category' else df['id_31']
        id_31_counts = id_31_series.value_counts()
        df['id_31_is_rare'] = (id_31_series.map(id_31_counts) <= 10).astype(int)
    
    return df


def create_ip_dist_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create IP/Distance-based features (dist1, dist2).
    These features likely represent distance from cardholder's typical location.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with IP/Distance features added
    """
    df = df.copy()
    
    # Dist1 features
    if 'dist1' in df.columns:
        dist1_freq = df['dist1'].value_counts()
        df['dist1_freq'] = df['dist1'].map(dist1_freq)
        
        if 'TransactionAmt' in df.columns:
            dist1_avg_amt = df.groupby('dist1')['TransactionAmt'].mean()
            df['dist1_avg_amt'] = df['dist1'].map(dist1_avg_amt)
            df['TransactionAmt_to_dist1_avg'] = df['TransactionAmt'] / (df['dist1_avg_amt'] + 1e-6)
        
        # Distance risk: very high or very low distance might indicate fraud
        if df['dist1'].dtype in ['float64', 'int64', 'float32', 'int32']:
            dist1_q99 = df['dist1'].quantile(0.99)
            dist1_q01 = df['dist1'].quantile(0.01)
            df['dist1_is_extreme'] = ((df['dist1'] >= dist1_q99) | (df['dist1'] <= dist1_q01)).astype(int)
    
    # Dist2 features
    if 'dist2' in df.columns:
        dist2_freq = df['dist2'].value_counts()
        df['dist2_freq'] = df['dist2'].map(dist2_freq)
        
        if 'TransactionAmt' in df.columns:
            dist2_avg_amt = df.groupby('dist2')['TransactionAmt'].mean()
            df['dist2_avg_amt'] = df['dist2'].map(dist2_avg_amt)
            df['TransactionAmt_to_dist2_avg'] = df['TransactionAmt'] / (df['dist2_avg_amt'] + 1e-6)
        
        # Distance risk
        if df['dist2'].dtype in ['float64', 'int64', 'float32', 'int32']:
            dist2_q99 = df['dist2'].quantile(0.99)
            dist2_q01 = df['dist2'].quantile(0.01)
            df['dist2_is_extreme'] = ((df['dist2'] >= dist2_q99) | (df['dist2'] <= dist2_q01)).astype(int)
    
    # Dist1-Dist2 combo
    if 'dist1' in df.columns and 'dist2' in df.columns:
        df['dist1_dist2_combo_count'] = df.groupby(['dist1', 'dist2']).transform('size')
        
        if 'TransactionAmt' in df.columns:
            df['dist1_dist2_avg_amt'] = df.groupby(['dist1', 'dist2'])['TransactionAmt'].transform('mean')
            df['TransactionAmt_to_dist1_dist2_avg'] = df['TransactionAmt'] / (df['dist1_dist2_avg_amt'] + 1e-6)
        
        # Distance difference
        if df['dist1'].dtype in ['float64', 'int64', 'float32', 'int32'] and df['dist2'].dtype in ['float64', 'int64', 'float32', 'int32']:
            df['dist_diff'] = (df['dist1'] - df['dist2']).abs()
            df['dist_sum'] = df['dist1'] + df['dist2']
    
    # Card-Distance interactions
    if 'card1' in df.columns and 'dist1' in df.columns:
        df['card1_dist1_combo_count'] = df.groupby(['card1', 'dist1']).transform('size')
    
    if 'card1' in df.columns and 'dist2' in df.columns:
        df['card1_dist2_combo_count'] = df.groupby(['card1', 'dist2']).transform('size')
    
    return df


def create_card_related_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features from card-related columns (C1-C14).
    These are likely encoded card features or card metadata.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with card-related features added
    """
    df = df.copy()
    
    # Process each C column
    c_cols = [f'C{i}' for i in range(1, 15)]
    available_c_cols = [col for col in c_cols if col in df.columns]
    
    if not available_c_cols:
        return df
    
    # Frequency features for each C column
    for c_col in available_c_cols:
        if df[c_col].dtype in ['float64', 'int64', 'float32', 'int32', 'int16', 'int8']:
            c_freq = df[c_col].value_counts()
            df[f'{c_col}_freq'] = df[c_col].map(c_freq)
            
            if 'TransactionAmt' in df.columns:
                c_avg_amt = df.groupby(c_col)['TransactionAmt'].mean()
                df[f'{c_col}_avg_amt'] = df[c_col].map(c_avg_amt)
                df[f'TransactionAmt_to_{c_col}_avg'] = df['TransactionAmt'] / (df[f'{c_col}_avg_amt'] + 1e-6)
    
    # Statistical features across C columns
    if len(available_c_cols) > 1:
        # Sum of C columns
        numeric_c_cols = [col for col in available_c_cols if df[col].dtype in ['float64', 'int64', 'float32', 'int32']]
        if numeric_c_cols:
            df['C_sum'] = df[numeric_c_cols].sum(axis=1)
            df['C_mean'] = df[numeric_c_cols].mean(axis=1)
            df['C_std'] = df[numeric_c_cols].std(axis=1).fillna(0)
            df['C_max'] = df[numeric_c_cols].max(axis=1)
            df['C_min'] = df[numeric_c_cols].min(axis=1)
    
    # C column interactions with cards
    if 'card1' in df.columns:
        for c_col in available_c_cols[:5]:  # Limit to first 5 to avoid too many features
            df[f'card1_{c_col}_combo_count'] = df.groupby(['card1', c_col]).transform('size')
    
    # C column interactions with ProductCD
    if 'ProductCD' in df.columns:
        for c_col in available_c_cols[:3]:  # Limit to first 3
            product_for_groupby = df['ProductCD'].astype(str) if df['ProductCD'].dtype.name == 'category' else df['ProductCD']
            df[f'ProductCD_{c_col}_combo_count'] = df.groupby([product_for_groupby, c_col]).transform('size')
    
    return df


def create_uid_features(df: pd.DataFrame, 
                        is_train: bool = True,
                        time_col: str = 'TransactionDT',
                        target_col: str = 'isFraud') -> pd.DataFrame:
    """
    Create UID (Unique Identifier) based features.
    Based on FraudSquad approach: combining card, addr, email, device features.
    
    UID combinations:
    - uid_1 = card1 + addr1 + P_emaildomain
    - uid_2 = card1 + addr1 + R_emaildomain
    - uid_3 = card1 + addr1 + DeviceInfo
    - uid_4 = card1 + addr1 + DeviceType
    - uid_5 = card1 + addr2 + P_emaildomain
    
    For each UID, create aggregate features:
    - count: Transaction count
    - avg_amt: Average transaction amount
    - std_amt: Standard deviation of amounts
    - fraud_rate: Fraud rate (if is_train=True)
    - time_since_last: Time since last transaction
    - velocity: Transaction velocity (amount/time)
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    is_train : bool
        Whether this is training data
    time_col : str
        Time column name
    target_col : str
        Target column name (for fraud_rate calculation)
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with UID features added
    """
    df = df.copy()
    
    # Sort by time for lag calculations
    if time_col in df.columns:
        df = df.sort_values(by=time_col).reset_index(drop=True)
    
    # UID 1: card1 + addr1 + P_emaildomain
    if all(col in df.columns for col in ['card1', 'addr1', 'P_emaildomain']):
        p_email_for_uid = df['P_emaildomain'].astype(str) if df['P_emaildomain'].dtype.name == 'category' else df['P_emaildomain']
        df['uid_1'] = df['card1'].astype(str) + '_' + df['addr1'].astype(str) + '_' + p_email_for_uid
        
        # Aggregate features
        uid1_counts = df.groupby('uid_1').size()
        df['uid_1_count'] = df['uid_1'].map(uid1_counts)
        
        if 'TransactionAmt' in df.columns:
            uid1_avg_amt = df.groupby('uid_1')['TransactionAmt'].mean()
            df['uid_1_avg_amt'] = df['uid_1'].map(uid1_avg_amt)
            df['uid_1_std_amt'] = df.groupby('uid_1')['TransactionAmt'].transform('std').fillna(0)
            df['TransactionAmt_to_uid_1_avg'] = df['TransactionAmt'] / (df['uid_1_avg_amt'] + 1e-6)
        
        if is_train and target_col in df.columns:
            uid1_fraud_rate = df.groupby('uid_1')[target_col].mean()
            df['uid_1_fraud_rate'] = df['uid_1'].map(uid1_fraud_rate).fillna(0)
        else:
            # For test set, fill with 0 (no target available)
            df['uid_1_fraud_rate'] = 0
        
        if time_col in df.columns:
            df['uid_1_time_since_last'] = df.groupby('uid_1')[time_col].diff().fillna(0)
            if 'TransactionAmt' in df.columns:
                df['uid_1_velocity'] = df.groupby('uid_1')['TransactionAmt'].diff() / (df['uid_1_time_since_last'] + 1e-6)
                df['uid_1_velocity'] = df['uid_1_velocity'].fillna(0)
    
    # UID 2: card1 + addr1 + R_emaildomain
    if all(col in df.columns for col in ['card1', 'addr1', 'R_emaildomain']):
        r_email_for_uid = df['R_emaildomain'].astype(str) if df['R_emaildomain'].dtype.name == 'category' else df['R_emaildomain']
        df['uid_2'] = df['card1'].astype(str) + '_' + df['addr1'].astype(str) + '_' + r_email_for_uid
        
        uid2_counts = df.groupby('uid_2').size()
        df['uid_2_count'] = df['uid_2'].map(uid2_counts)
        
        if 'TransactionAmt' in df.columns:
            uid2_avg_amt = df.groupby('uid_2')['TransactionAmt'].mean()
            df['uid_2_avg_amt'] = df['uid_2'].map(uid2_avg_amt)
            df['uid_2_std_amt'] = df.groupby('uid_2')['TransactionAmt'].transform('std').fillna(0)
            df['TransactionAmt_to_uid_2_avg'] = df['TransactionAmt'] / (df['uid_2_avg_amt'] + 1e-6)
        
        if is_train and target_col in df.columns:
            uid2_fraud_rate = df.groupby('uid_2')[target_col].mean()
            df['uid_2_fraud_rate'] = df['uid_2'].map(uid2_fraud_rate).fillna(0)
        else:
            # For test set, fill with 0 (no target available)
            df['uid_2_fraud_rate'] = 0
        
        if time_col in df.columns:
            df['uid_2_time_since_last'] = df.groupby('uid_2')[time_col].diff().fillna(0)
    
    # UID 3: card1 + addr1 + DeviceInfo
    if all(col in df.columns for col in ['card1', 'addr1', 'DeviceInfo']):
        device_info_for_uid = df['DeviceInfo'].astype(str) if df['DeviceInfo'].dtype.name == 'category' else df['DeviceInfo']
        df['uid_3'] = df['card1'].astype(str) + '_' + df['addr1'].astype(str) + '_' + device_info_for_uid
        
        uid3_counts = df.groupby('uid_3').size()
        df['uid_3_count'] = df['uid_3'].map(uid3_counts)
        
        if 'TransactionAmt' in df.columns:
            uid3_avg_amt = df.groupby('uid_3')['TransactionAmt'].mean()
            df['uid_3_avg_amt'] = df['uid_3'].map(uid3_avg_amt)
            df['uid_3_std_amt'] = df.groupby('uid_3')['TransactionAmt'].transform('std').fillna(0)
        
        if is_train and target_col in df.columns:
            uid3_fraud_rate = df.groupby('uid_3')[target_col].mean()
            df['uid_3_fraud_rate'] = df['uid_3'].map(uid3_fraud_rate).fillna(0)
        else:
            # For test set, fill with 0 (no target available)
            df['uid_3_fraud_rate'] = 0
    
    # UID 4: card1 + addr1 + DeviceType
    if all(col in df.columns for col in ['card1', 'addr1', 'DeviceType']):
        device_type_for_uid = df['DeviceType'].astype(str) if df['DeviceType'].dtype.name == 'category' else df['DeviceType']
        df['uid_4'] = df['card1'].astype(str) + '_' + df['addr1'].astype(str) + '_' + device_type_for_uid
        
        uid4_counts = df.groupby('uid_4').size()
        df['uid_4_count'] = df['uid_4'].map(uid4_counts)
        
        if 'TransactionAmt' in df.columns:
            uid4_avg_amt = df.groupby('uid_4')['TransactionAmt'].mean()
            df['uid_4_avg_amt'] = df['uid_4'].map(uid4_avg_amt)
        
        if is_train and target_col in df.columns:
            uid4_fraud_rate = df.groupby('uid_4')[target_col].mean()
            df['uid_4_fraud_rate'] = df['uid_4'].map(uid4_fraud_rate).fillna(0)
        else:
            # For test set, fill with 0 (no target available)
            df['uid_4_fraud_rate'] = 0
    
    # UID 5: card1 + addr2 + P_emaildomain
    if all(col in df.columns for col in ['card1', 'addr2', 'P_emaildomain']):
        p_email_for_uid = df['P_emaildomain'].astype(str) if df['P_emaildomain'].dtype.name == 'category' else df['P_emaildomain']
        df['uid_5'] = df['card1'].astype(str) + '_' + df['addr2'].astype(str) + '_' + p_email_for_uid
        
        uid5_counts = df.groupby('uid_5').size()
        df['uid_5_count'] = df['uid_5'].map(uid5_counts)
        
        if 'TransactionAmt' in df.columns:
            uid5_avg_amt = df.groupby('uid_5')['TransactionAmt'].mean()
            df['uid_5_avg_amt'] = df['uid_5'].map(uid5_avg_amt)
        
        if is_train and target_col in df.columns:
            uid5_fraud_rate = df.groupby('uid_5')[target_col].mean()
            df['uid_5_fraud_rate'] = df['uid_5'].map(uid5_fraud_rate).fillna(0)
        else:
            # For test set, fill with 0 (no target available)
            df['uid_5_fraud_rate'] = 0
    
    # Drop UID columns (we only need the aggregate features)
    uid_cols = [col for col in df.columns if col.startswith('uid_') and col not in 
                [f'uid_{i}_count' for i in range(1, 6)] + 
                [f'uid_{i}_avg_amt' for i in range(1, 6)] + 
                [f'uid_{i}_std_amt' for i in range(1, 4)] +
                [f'uid_{i}_fraud_rate' for i in range(1, 6)] +
                [f'uid_{i}_time_since_last' for i in range(1, 3)] +
                [f'uid_{i}_velocity' for i in range(1, 2)] +
                [f'TransactionAmt_to_uid_{i}_avg' for i in range(1, 3)]]
    df = df.drop(columns=[col for col in uid_cols if col in df.columns], errors='ignore')
    
    return df


def create_all_sixth_view_features(df: pd.DataFrame, 
                                   is_train: bool = True) -> pd.DataFrame:
    """
    Create all features for sixth view model.
    Extends fifth view with:
    - All card features (card1-card6)
    - IP/Distance features (dist1, dist2)
    - Card-related features (C1-C14)
    - Enhanced address features (addr1, addr2)
    
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
    
    print("Creating card features (card1-card6)...")
    df = create_card_features(df, is_train=is_train)
    
    print("Creating address features (addr1, addr2)...")
    df = create_address_features(df)
    
    print("Creating email features...")
    df = create_email_features(df)
    
    print("Creating product features...")
    df = create_product_features(df)
    
    print("Creating device features...")
    df = create_device_features(df)
    
    print("Creating identity features...")
    df = create_identity_features(df)
    
    print("Creating IP/Distance features (dist1, dist2)...")
    df = create_ip_dist_features(df)
    
    print("Creating card-related features (C1-C14)...")
    df = create_card_related_features(df)
    
    print("Creating UID features (FraudSquad approach)...")
    df = create_uid_features(df, is_train=is_train)
    
    print("Creating basic interaction features...")
    df = create_basic_interaction_features(df)
    
    print("Creating advanced interaction features...")
    df = create_advanced_interaction_features(df)
    
    print("Creating statistical features...")
    df = create_statistical_features(df)
    
    print("Creating enhanced statistical features...")
    df = create_enhanced_statistical_features(df)
    
    print("Creating advanced statistical features...")
    df = create_advanced_statistical_features(df)
    
    print("Creating lag features...")
    df = create_lag_features(df)
    
    print("Creating enhanced lag features...")
    df = create_enhanced_lag_features(df)
    
    print("Creating advanced lag features v2...")
    df = create_advanced_lag_features_v2(df)
    
    print("Creating extended triple/quadruple combinations...")
    df = create_extended_triple_combinations(df)
    
    print("Creating domain-specific features...")
    df = create_domain_specific_features(df)
    
    print(f"Feature engineering complete. Final shape: {df.shape}")
    
    return df

