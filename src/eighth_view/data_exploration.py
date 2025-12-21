"""
Data exploration utilities for logging detailed analysis results.
These functions will be called during training to generate exploration logs.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


def log_missing_data_statistics(df: pd.DataFrame, 
                                is_train: bool = True,
                                verbose: bool = True) -> Dict:
    """
    Log detailed missing data statistics for all features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    is_train : bool
        Whether this is training data
    verbose : bool
        Whether to print results
        
    Returns
    -------
    stats : dict
        Dictionary with missing data statistics
    """
    stats = {
        'total_rows': len(df),
        'features': {},
        'summary': {
            'no_missing': [],
            'low_missing': [],  # < 20%
            'medium_missing': [],  # 20-50%
            'high_missing': []  # > 50%
        }
    }
    
    for col in df.columns:
        if col in ['TransactionID', 'isFraud']:
            continue
            
        missing_count = df[col].isna().sum()
        missing_pct = (missing_count / len(df)) * 100
        
        stats['features'][col] = {
            'missing_count': int(missing_count),
            'missing_pct': float(missing_pct),
            'dtype': str(df[col].dtype)
        }
        
        if missing_pct == 0:
            stats['summary']['no_missing'].append(col)
        elif missing_pct < 20:
            stats['summary']['low_missing'].append(col)
        elif missing_pct < 50:
            stats['summary']['medium_missing'].append(col)
        else:
            stats['summary']['high_missing'].append(col)
    
    if verbose:
        print("\n" + "=" * 70)
        print("MISSING DATA STATISTICS")
        print("=" * 70)
        print(f"Total rows: {stats['total_rows']:,}")
        print(f"\nFeatures with NO missing data: {len(stats['summary']['no_missing'])}")
        print(f"Features with LOW missing data (<20%): {len(stats['summary']['low_missing'])}")
        print(f"Features with MEDIUM missing data (20-50%): {len(stats['summary']['medium_missing'])}")
        print(f"Features with HIGH missing data (>50%): {len(stats['summary']['high_missing'])}")
        
        # Show top features with missing data
        missing_features = [(col, stats['features'][col]['missing_pct']) 
                           for col in stats['features'] 
                           if stats['features'][col]['missing_pct'] > 0]
        missing_features.sort(key=lambda x: x[1], reverse=True)
        
        if missing_features:
            print(f"\nTop 20 features with missing data:")
            for col, pct in missing_features[:20]:
                print(f"  {col}: {pct:.2f}%")
        
        print("=" * 70)
    
    return stats


def log_feature_distribution_statistics(df: pd.DataFrame,
                                       target_col: str = 'isFraud',
                                       is_train: bool = True,
                                       verbose: bool = True) -> Dict:
    """
    Log detailed distribution statistics for numeric and categorical features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Target column name
    is_train : bool
        Whether this is training data
    verbose : bool
        Whether to print results
        
    Returns
    -------
    stats : dict
        Dictionary with distribution statistics
    """
    stats = {
        'numeric_features': {},
        'categorical_features': {},
        'transaction_amt': {}
    }
    
    # TransactionAmt detailed statistics
    if 'TransactionAmt' in df.columns:
        amt = df['TransactionAmt']
        stats['transaction_amt'] = {
            'mean': float(amt.mean()),
            'median': float(amt.median()),
            'std': float(amt.std()),
            'min': float(amt.min()),
            'max': float(amt.max()),
            'q25': float(amt.quantile(0.25)),
            'q75': float(amt.quantile(0.75)),
            'q95': float(amt.quantile(0.95)),
            'q99': float(amt.quantile(0.99))
        }
        
        if is_train and target_col in df.columns:
            fraud_amt = df[df[target_col] == 1]['TransactionAmt']
            normal_amt = df[df[target_col] == 0]['TransactionAmt']
            
            stats['transaction_amt']['fraud_mean'] = float(fraud_amt.mean())
            stats['transaction_amt']['fraud_median'] = float(fraud_amt.median())
            stats['transaction_amt']['normal_mean'] = float(normal_amt.mean())
            stats['transaction_amt']['normal_median'] = float(normal_amt.median())
    
    # Numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols[:20]:  # Limit to first 20 for performance
        if col in ['TransactionID', 'isFraud']:
            continue
        try:
            stats['numeric_features'][col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max())
            }
        except:
            pass
    
    # Categorical features
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_cols[:10]:  # Limit to first 10
        try:
            value_counts = df[col].value_counts()
            stats['categorical_features'][col] = {
                'nunique': int(df[col].nunique()),
                'top_5_values': value_counts.head(5).to_dict()
            }
        except:
            pass
    
    if verbose:
        print("\n" + "=" * 70)
        print("FEATURE DISTRIBUTION STATISTICS")
        print("=" * 70)
        
        if stats['transaction_amt']:
            amt_stats = stats['transaction_amt']
            print(f"\nTransactionAmt Statistics:")
            print(f"  Mean: ${amt_stats['mean']:.2f}")
            print(f"  Median: ${amt_stats['median']:.2f}")
            print(f"  Std: ${amt_stats['std']:.2f}")
            print(f"  Min: ${amt_stats['min']:.2f}, Max: ${amt_stats['max']:.2f}")
            print(f"  Q25: ${amt_stats['q25']:.2f}, Q75: ${amt_stats['q75']:.2f}")
            print(f"  Q95: ${amt_stats['q95']:.2f}, Q99: ${amt_stats['q99']:.2f}")
            
            if 'fraud_mean' in amt_stats:
                print(f"\n  Fraud Transactions:")
                print(f"    Mean: ${amt_stats['fraud_mean']:.2f}")
                print(f"    Median: ${amt_stats['fraud_median']:.2f}")
                print(f"\n  Normal Transactions:")
                print(f"    Mean: ${amt_stats['normal_mean']:.2f}")
                print(f"    Median: ${amt_stats['normal_median']:.2f}")
        
        if stats['categorical_features']:
            print(f"\nCategorical Features (Top 10):")
            for col, info in list(stats['categorical_features'].items())[:10]:
                print(f"  {col}: {info['nunique']} unique values")
                print(f"    Top values: {list(info['top_5_values'].keys())[:3]}")
        
        print("=" * 70)
    
    return stats


def log_correlation_analysis(df: pd.DataFrame,
                            target_col: str = 'isFraud',
                            is_train: bool = True,
                            verbose: bool = True,
                            top_n: int = 20) -> Dict:
    """
    Log correlation analysis including feature-feature and feature-target correlations.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Target column name
    is_train : bool
        Whether this is training data
    verbose : bool
        Whether to print results
    top_n : int
        Number of top correlations to show
        
    Returns
    -------
    stats : dict
        Dictionary with correlation statistics
    """
    stats = {
        'high_correlation_pairs': [],
        'target_correlations': {}
    }
    
    # Select numeric features only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['TransactionID', 'isFraud']]
    
    if len(numeric_cols) == 0:
        return stats
    
    # Feature-feature correlations (sample for performance)
    sample_cols = numeric_cols[:50]  # Limit to 50 features for performance
    if len(sample_cols) > 1:
        corr_matrix = df[sample_cols].corr().abs()
        
        # Find high correlation pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if corr_val > 0.95:  # Very high correlation
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': float(corr_val)
                    })
        
        high_corr_pairs.sort(key=lambda x: x['correlation'], reverse=True)
        stats['high_correlation_pairs'] = high_corr_pairs[:top_n]
    
    # Target correlations (if training data)
    if is_train and target_col in df.columns:
        target_corrs = {}
        for col in numeric_cols[:100]:  # Limit to 100 features
            try:
                corr = df[col].corr(df[target_col])
                if not np.isnan(corr):
                    target_corrs[col] = float(abs(corr))
            except:
                pass
        
        # Sort by absolute correlation
        sorted_corrs = sorted(target_corrs.items(), key=lambda x: x[1], reverse=True)
        stats['target_correlations'] = dict(sorted_corrs[:top_n])
    
    if verbose:
        print("\n" + "=" * 70)
        print("CORRELATION ANALYSIS")
        print("=" * 70)
        
        if stats['high_correlation_pairs']:
            print(f"\nHigh Correlation Pairs (>0.95): {len(stats['high_correlation_pairs'])}")
            for pair in stats['high_correlation_pairs'][:10]:
                print(f"  {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.4f}")
        
        if stats['target_correlations']:
            print(f"\nTop {top_n} Features Correlated with Target:")
            for i, (col, corr) in enumerate(list(stats['target_correlations'].items())[:top_n], 1):
                print(f"  {i}. {col}: {corr:.4f}")
        
        print("=" * 70)
    
    return stats


def log_temporal_pattern_analysis(df: pd.DataFrame,
                                 target_col: str = 'isFraud',
                                 time_col: str = 'TransactionDT',
                                 is_train: bool = True,
                                 verbose: bool = True) -> Dict:
    """
    Log temporal pattern analysis including fraud rates by time periods.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Target column name
    time_col : str
        Time column name
    is_train : bool
        Whether this is training data
    verbose : bool
        Whether to print results
        
    Returns
    -------
    stats : dict
        Dictionary with temporal pattern statistics
    """
    stats = {
        'time_range': {},
        'fraud_by_hour': {},
        'fraud_by_day': {},
        'fraud_by_weekend': {}
    }
    
    if time_col not in df.columns:
        return stats
    
    # Time range
    stats['time_range'] = {
        'min': float(df[time_col].min()),
        'max': float(df[time_col].max()),
        'span_days': float((df[time_col].max() - df[time_col].min()) / (24 * 3600))
    }
    
    if is_train and target_col in df.columns:
        # Create time features if not exist
        if 'hour' not in df.columns:
            df['hour'] = (df[time_col] // 3600) % 24
        if 'day_of_week' not in df.columns:
            df['day_of_week'] = (df[time_col] // (24 * 3600)) % 7
        if 'is_weekend' not in df.columns:
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Fraud by hour
        fraud_by_hour = df.groupby('hour')[target_col].agg(['mean', 'count'])
        stats['fraud_by_hour'] = {
            'hourly_fraud_rates': fraud_by_hour['mean'].to_dict(),
            'hourly_counts': fraud_by_hour['count'].to_dict()
        }
        
        # Fraud by day of week
        fraud_by_day = df.groupby('day_of_week')[target_col].agg(['mean', 'count'])
        stats['fraud_by_day'] = {
            'daily_fraud_rates': fraud_by_day['mean'].to_dict(),
            'daily_counts': fraud_by_day['count'].to_dict()
        }
        
        # Fraud by weekend
        fraud_by_weekend = df.groupby('is_weekend')[target_col].agg(['mean', 'count'])
        stats['fraud_by_weekend'] = {
            'weekend_fraud_rate': float(fraud_by_weekend.loc[1, 'mean']) if 1 in fraud_by_weekend.index else 0,
            'weekday_fraud_rate': float(fraud_by_weekend.loc[0, 'mean']) if 0 in fraud_by_weekend.index else 0
        }
    
    if verbose:
        print("\n" + "=" * 70)
        print("TEMPORAL PATTERN ANALYSIS")
        print("=" * 70)
        
        if stats['time_range']:
            print(f"\nTime Range:")
            print(f"  Span: {stats['time_range']['span_days']:.1f} days")
        
        if stats['fraud_by_hour']:
            print(f"\nFraud Rate by Hour (Top 5):")
            hourly_rates = sorted(stats['fraud_by_hour']['hourly_fraud_rates'].items(), 
                                 key=lambda x: x[1], reverse=True)
            for hour, rate in hourly_rates[:5]:
                print(f"  Hour {hour}: {rate:.4f}")
        
        if stats['fraud_by_weekend']:
            print(f"\nFraud Rate by Weekend:")
            print(f"  Weekend: {stats['fraud_by_weekend']['weekend_fraud_rate']:.4f}")
            print(f"  Weekday: {stats['fraud_by_weekend']['weekday_fraud_rate']:.4f}")
        
        print("=" * 70)
    
    return stats


def log_fraud_pattern_analysis(df: pd.DataFrame,
                              target_col: str = 'isFraud',
                              is_train: bool = True,
                              verbose: bool = True,
                              top_n: int = 10) -> Dict:
    """
    Log fraud pattern analysis including fraud rates by feature groups.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Target column name
    is_train : bool
        Whether this is training data
    verbose : bool
        Whether to print results
    top_n : int
        Number of top patterns to show
        
    Returns
    -------
    stats : dict
        Dictionary with fraud pattern statistics
    """
    stats = {
        'fraud_by_product': {},
        'fraud_by_device': {},
        'fraud_by_card_count': {},
        'uid_fraud_rates': {}
    }
    
    if not is_train or target_col not in df.columns:
        return stats
    
    # Fraud by ProductCD
    if 'ProductCD' in df.columns:
        fraud_by_product = df.groupby('ProductCD')[target_col].agg(['mean', 'count'])
        stats['fraud_by_product'] = fraud_by_product['mean'].to_dict()
    
    # Fraud by DeviceType
    if 'DeviceType' in df.columns:
        fraud_by_device = df.groupby('DeviceType')[target_col].agg(['mean', 'count'])
        stats['fraud_by_device'] = fraud_by_device['mean'].to_dict()
    
    # Fraud by card1_count (new vs active cards)
    if 'card1_count' in df.columns:
        df['card1_count_bin'] = pd.cut(df['card1_count'], 
                                       bins=[0, 1, 5, 20, 100, float('inf')],
                                       labels=['1', '2-5', '6-20', '21-100', '100+'])
        fraud_by_card_count = df.groupby('card1_count_bin')[target_col].agg(['mean', 'count'])
        stats['fraud_by_card_count'] = fraud_by_card_count['mean'].to_dict()
    
    # UID fraud rates (if available)
    uid_cols = [col for col in df.columns if col.startswith('uid_') and col.endswith('_fraud_rate')]
    if uid_cols:
        for col in uid_cols[:5]:
            stats['uid_fraud_rates'][col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'max': float(df[col].max())
            }
    
    if verbose:
        print("\n" + "=" * 70)
        print("FRAUD PATTERN ANALYSIS")
        print("=" * 70)
        
        if stats['fraud_by_product']:
            print(f"\nFraud Rate by ProductCD:")
            for product, rate in sorted(stats['fraud_by_product'].items(), 
                                       key=lambda x: x[1], reverse=True)[:top_n]:
                print(f"  {product}: {rate:.4f}")
        
        if stats['fraud_by_device']:
            print(f"\nFraud Rate by DeviceType:")
            for device, rate in sorted(stats['fraud_by_device'].items(), 
                                     key=lambda x: x[1], reverse=True)[:top_n]:
                print(f"  {device}: {rate:.4f}")
        
        if stats['fraud_by_card_count']:
            print(f"\nFraud Rate by Card Activity:")
            for bin_label, rate in sorted(stats['fraud_by_card_count'].items(), 
                                         key=lambda x: x[1], reverse=True)[:top_n]:
                print(f"  {bin_label}: {rate:.4f}")
        
        if stats['uid_fraud_rates']:
            print(f"\nUID Fraud Rate Statistics:")
            for col, info in stats['uid_fraud_rates'].items():
                print(f"  {col}: mean={info['mean']:.4f}, median={info['median']:.4f}, max={info['max']:.4f}")
        
        print("=" * 70)
    
    return stats


def log_feature_types_and_categories(df: pd.DataFrame,
                                    verbose: bool = True) -> Dict:
    """
    Log feature data types and categorical/numeric classification.
    Uses improved logic to identify categorical features that appear numeric.
    
    Classification Logic:
    1. Object/string types → Always Categorical
    2. Boolean (2 unique values) → Boolean/Categorical
    3. Numeric types:
       - If unique_ratio < 0.1 (unique values < 10% of total rows) → Categorical
       - If unique_ratio > 0.5 (unique values > 50% of total rows) → Numeric
       - If 0.1 <= unique_ratio <= 0.5 → Context-based (check if it's an ID/code)
       - Also check: if nunique < 50 → Likely Categorical
       - Also check: if nunique > 1000 → Likely Numeric
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    verbose : bool
        Whether to print results
        
    Returns
    -------
    stats : dict
        Dictionary with feature type information
    """
    stats = {
        'feature_types': {},
        'summary': {
            'numeric_features': [],
            'categorical_features': [],
            'boolean_features': [],
            'datetime_features': []
        }
    }
    
    total_rows = len(df)
    
    for col in df.columns:
        if col in ['TransactionID', 'isFraud']:
            continue
        
        dtype = str(df[col].dtype)
        nunique = df[col].nunique()
        unique_ratio = nunique / total_rows if total_rows > 0 else 0
        
        # Determine base type
        is_numeric_dtype = pd.api.types.is_numeric_dtype(df[col])
        is_boolean = False
        is_datetime = pd.api.types.is_datetime64_any_dtype(df[col])
        
        # Step 1: Check if object/string (always categorical)
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            is_categorical = True
            classification = 'categorical'
        
        # Step 2: Check if boolean
        elif dtype in ['bool', 'bool_'] or nunique == 2:
            is_boolean = True
            is_categorical = True
            classification = 'boolean'
        
        # Step 3: For numeric types, use improved logic
        elif is_numeric_dtype:
            # Integer types: prioritize categorical (likely ID/code)
            if 'int' in dtype:
                # Integer with reasonable cardinality → Categorical (ID/code)
                # Examples: card1 (ID), TransactionDT (timestamp - but check ratio)
                if unique_ratio > 0.9:  # Very high ratio (>90%) → Likely timestamp/continuous
                    is_categorical = False
                    classification = 'numeric'
                elif nunique < 10000:  # IDs/codes typically have < 10k unique values
                    is_categorical = True
                    classification = 'categorical'
                else:
                    is_categorical = False
                    classification = 'numeric'
            
            # Float types: check cardinality and ratio
            elif 'float' in dtype:
                # Very low cardinality → Categorical
                if nunique < 50:
                    is_categorical = True
                    classification = 'categorical'
                # Very high cardinality with high ratio → Numeric
                elif nunique > 1000 and unique_ratio > 0.5:
                    is_categorical = False
                    classification = 'numeric'
                # Medium cardinality: use unique ratio
                else:
                    if unique_ratio < 0.1:  # Less than 10% unique → Likely categorical
                        is_categorical = True
                        classification = 'categorical'
                    elif unique_ratio > 0.5:  # More than 50% unique → Likely numeric
                        is_categorical = False
                        classification = 'numeric'
                    else:  # 10-50% unique → Check if discrete
                        if unique_ratio < 0.3:  # Low ratio → Categorical (discrete codes)
                            is_categorical = True
                            classification = 'categorical'
                        else:
                            is_categorical = False
                            classification = 'numeric'
            else:
                is_categorical = False
                classification = 'numeric'
        else:
            # Other types (datetime, etc.)
            is_categorical = False
            classification = 'other'
        
        stats['feature_types'][col] = {
            'dtype': dtype,
            'nunique': int(nunique),
            'unique_ratio': float(unique_ratio),
            'is_numeric_dtype': is_numeric_dtype,
            'is_categorical': is_categorical,
            'is_boolean': is_boolean,
            'is_datetime': is_datetime,
            'classification': classification
        }
        
        # Add to summary
        if is_datetime:
            stats['summary']['datetime_features'].append(col)
        elif is_boolean:
            stats['summary']['boolean_features'].append(col)
        elif is_categorical:
            stats['summary']['categorical_features'].append(col)
        elif classification == 'numeric':
            stats['summary']['numeric_features'].append(col)
    
    if verbose:
        print("\n" + "=" * 70)
        print("FEATURE TYPES AND CATEGORIES")
        print("=" * 70)
        
        print(f"\nSummary:")
        print(f"  Numeric features: {len(stats['summary']['numeric_features'])}")
        print(f"  Categorical features: {len(stats['summary']['categorical_features'])}")
        print(f"  Boolean features: {len(stats['summary']['boolean_features'])}")
        print(f"  Datetime features: {len(stats['summary']['datetime_features'])}")
        
        # Show feature types by category with improved classification
        print(f"\nNumeric Features (sample):")
        for col in stats['summary']['numeric_features'][:10]:
            info = stats['feature_types'][col]
            print(f"  {col}: {info['dtype']} | unique: {info['nunique']:,} | ratio: {info['unique_ratio']:.2%}")
        
        print(f"\nCategorical Features (sample):")
        for col in stats['summary']['categorical_features'][:10]:
            info = stats['feature_types'][col]
            print(f"  {col}: {info['dtype']} | unique: {info['nunique']:,} | ratio: {info['unique_ratio']:.2%}")
        
        # Show potentially misclassified features (numeric dtype but categorical)
        print(f"\nPotentially Categorical (Numeric Dtype but Low Cardinality):")
        misclassified = []
        for col, info in stats['feature_types'].items():
            if info['is_numeric_dtype'] and info['classification'] == 'categorical':
                misclassified.append((col, info))
        misclassified.sort(key=lambda x: x[1]['nunique'])
        for col, info in misclassified[:10]:
            print(f"  {col}: {info['dtype']} | unique: {info['nunique']:,} | ratio: {info['unique_ratio']:.2%} → Categorical")
        
        print("=" * 70)
    
    return stats


def run_all_exploration_analyses(df: pd.DataFrame,
                                target_col: str = 'isFraud',
                                time_col: str = 'TransactionDT',
                                is_train: bool = True,
                                verbose: bool = True) -> Dict:
    """
    Run all exploration analyses and return combined results.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Target column name
    time_col : str
        Time column name
    is_train : bool
        Whether this is training data
    verbose : bool
        Whether to print results
        
    Returns
    -------
    all_stats : dict
        Dictionary with all exploration statistics
    """
    all_stats = {}
    
    # Missing data statistics
    all_stats['missing_data'] = log_missing_data_statistics(df, is_train=is_train, verbose=verbose)
    
    # Feature distribution statistics
    all_stats['distributions'] = log_feature_distribution_statistics(
        df, target_col=target_col, is_train=is_train, verbose=verbose
    )
    
    # Correlation analysis
    all_stats['correlations'] = log_correlation_analysis(
        df, target_col=target_col, is_train=is_train, verbose=verbose
    )
    
    # Temporal pattern analysis
    all_stats['temporal_patterns'] = log_temporal_pattern_analysis(
        df, target_col=target_col, time_col=time_col, is_train=is_train, verbose=verbose
    )
    
    # Fraud pattern analysis
    all_stats['fraud_patterns'] = log_fraud_pattern_analysis(
        df, target_col=target_col, is_train=is_train, verbose=verbose
    )
    
    # Feature types and categories
    all_stats['feature_types'] = log_feature_types_and_categories(
        df, verbose=verbose
    )
    
    return all_stats

