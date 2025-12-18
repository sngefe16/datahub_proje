"""Feature engineering: adds DeviceType, DeviceInfo, id_28-id_31, interaction features."""

import pandas as pd
import numpy as np
from typing import Optional, List


# ===== First View Features (Reused) =====

def create_time_features(df: pd.DataFrame, 
                        time_col: str = 'TransactionDT') -> pd.DataFrame:
    """Creates time features: day_of_week, hour, day, week, is_weekend, is_night."""
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
        # Convert to string if categorical to avoid comparison issues
        p_email_series = df['P_emaildomain'].astype(str) if df['P_emaildomain'].dtype.name == 'category' else df['P_emaildomain']
        p_email_freq = p_email_series.value_counts()
        df['P_emaildomain_freq'] = p_email_series.map(p_email_freq)
    
    if 'R_emaildomain' in df.columns:
        # Convert to string if categorical to avoid comparison issues
        r_email_series = df['R_emaildomain'].astype(str) if df['R_emaildomain'].dtype.name == 'category' else df['R_emaildomain']
        r_email_freq = r_email_series.value_counts()
        df['R_emaildomain_freq'] = r_email_series.map(r_email_freq)
    
    if 'P_emaildomain' in df.columns and 'R_emaildomain' in df.columns:
        # Convert to string for comparison if categorical
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


# ===== New Features for Second View =====

def create_device_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create device-based features.
    
    Features created:
    - DeviceType_freq: Frequency of DeviceType value
    - DeviceInfo_freq: Frequency of DeviceInfo value
    - DeviceType_DeviceInfo_count: Count of (DeviceType, DeviceInfo) combinations
    - is_mobile: Is mobile device (1) or not (0)
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with device features added
    """
    df = df.copy()
    
    # DeviceType features
    if 'DeviceType' in df.columns:
        # Convert to string if categorical to avoid comparison issues
        device_type_series = df['DeviceType'].astype(str) if df['DeviceType'].dtype.name == 'category' else df['DeviceType']
        device_type_freq = device_type_series.value_counts()
        df['DeviceType_freq'] = device_type_series.map(device_type_freq)
        
        # Is mobile device
        df['is_mobile'] = (device_type_series == 'mobile').astype(int)
    
    # DeviceInfo features
    if 'DeviceInfo' in df.columns:
        device_info_freq = df['DeviceInfo'].value_counts()
        df['DeviceInfo_freq'] = df['DeviceInfo'].map(device_info_freq)
    
    # DeviceType-DeviceInfo combination
    if 'DeviceType' in df.columns and 'DeviceInfo' in df.columns:
        df['DeviceType_DeviceInfo_count'] = df.groupby(['DeviceType', 'DeviceInfo']).transform('size')
    
    return df


def create_identity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create identity-based features from id_28, id_29, id_30, id_31.
    
    Features created:
    - id_XX_freq: Frequency of each id value
    - id_XX_isMissing: Missing indicator
    - id_combinations: Various combinations of id features
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with identity features added
    """
    df = df.copy()
    
    id_cols = ['id_28', 'id_29', 'id_30', 'id_31']
    
    for id_col in id_cols:
        if id_col in df.columns:
            # Frequency encoding
            id_freq = df[id_col].value_counts()
            df[f'{id_col}_freq'] = df[id_col].map(id_freq)
            
            # Missing indicator (if not already created in preprocessing)
            if f'{id_col}_isMissing' not in df.columns:
                df[f'{id_col}_isMissing'] = df[id_col].isna().astype(int)
    
    # Identity feature combinations
    if 'id_28' in df.columns and 'id_29' in df.columns:
        df['id_28_id_29_count'] = df.groupby(['id_28', 'id_29']).transform('size')
    
    if 'id_30' in df.columns and 'id_31' in df.columns:
        df['id_30_id_31_count'] = df.groupby(['id_30', 'id_31']).transform('size')
    
    if all(col in df.columns for col in ['id_28', 'id_29', 'id_30', 'id_31']):
        df['id_28_id_29_id_30_id_31_count'] = df.groupby(['id_28', 'id_29', 'id_30', 'id_31']).transform('size')
    
    return df


# ===== Advanced Interaction Features =====

def create_advanced_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create advanced interaction features between different feature groups.
    
    Features created:
    - Card-Device interactions
    - Card-Identity interactions
    - Device-Product interactions
    - Email-Device interactions
    - Time-Device interactions
    - Amount-Device interactions
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with advanced interaction features added
    """
    df = df.copy()
    
    # === Card-Device Interactions ===
    if 'card1' in df.columns and 'DeviceType' in df.columns:
        # Convert DeviceType to string if categorical for groupby
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
        # Convert DeviceType to string if categorical for groupby
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
    
    # === Triple Combinations (Most Powerful) ===
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


def create_all_second_view_features(df: pd.DataFrame, 
                                   is_train: bool = True) -> pd.DataFrame:
    """
    Create all features for second view model.
    
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
    
    print(f"Feature engineering complete. Final shape: {df.shape}")
    
    return df

