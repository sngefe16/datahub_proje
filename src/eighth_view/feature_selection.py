"""
Feature selection: removes leakage, low variance, high correlation, selects by importance.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    VarianceThreshold, SelectFromModel
)
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


def remove_leakage_features(X: pd.DataFrame, verbose: bool = True) -> tuple:
    """
    Removes leakage features: fraud_rate, time_since_last, future velocity, expanding windows.
    Keeps past-based rolling/statistical features.
    
    Args:
        X: Feature dataframe.
        verbose: Print removed features.
    
    Returns:
        (X_safe, removed_features)
    """
    X_safe = X.copy()
    removed_features = []
    
    # Pattern 1: Target leakage features (fraud_rate)
    # These directly encode the target variable and cause severe overfitting
    fraud_rate_features = [col for col in X_safe.columns if col.endswith('_fraud_rate')]
    removed_features.extend(fraud_rate_features)
    
    # ✅ Pattern 2: Sadece gerçek time-based leakage (geleceğe bakan)
    # CRITICAL FIX: Geçmişe dayalı rolling/statistical features KALMALI
    # Sadece geleceğe bakan features kaldırılmalı
    time_leakage_patterns = [
        'time_since_last',  # ✅ KALDIR - geleceğe bakıyor
        'hours_since_last',  # ✅ KALDIR - geleceğe bakıyor
        'days_since_last',  # ✅ KALDIR - geleceğe bakıyor (yanlış direction)
        'time_between_last_two',  # ✅ KALDIR - geleceğe bakıyor
        'time_since_second_last',  # ✅ KALDIR - geleceğe bakıyor
    ]
    for pattern in time_leakage_patterns:
        matching_features = [col for col in X_safe.columns if pattern in col.lower()]
        removed_features.extend(matching_features)
    
    # ✅ Pattern 2.5: Velocity features - sadece geleceğe bakan velocity'ler
    # NOT: Geçmişe dayalı velocity (örn: card1_amt_velocity_last_24h) KALMALI
    velocity_features = [col for col in X_safe.columns if '_velocity' in col.lower()]
    for feat in velocity_features:
        # Sadece geleceğe bakan velocity'leri kaldır (örn: uid_1_velocity - geleceğe bakıyor)
        # Geçmişe dayalı velocity'leri KAL (örn: card1_amt_velocity_last_24h - geçmişe bakıyor)
        if 'last_' not in feat.lower() and 'rolling' not in feat.lower():
            removed_features.append(feat)
    
    # ✅ Pattern 3: Expanding window features - sadece geleceğe bakan expanding
    # CRITICAL FIX: Expanding max/min/mean KALDIR (geleceğe bakıyor)
    # Expanding features that look forward
    expanding_leakage_patterns = [
        'expanding_max',
        'expanding_min',
        'expanding_mean',
        'expanding_std',
        'expanding_sum',
    ]
    for pattern in expanding_leakage_patterns:
        matching_features = [col for col in X_safe.columns if pattern in col.lower()]
        removed_features.extend(matching_features)
    
    # ✅ Pattern 4: Rolling features - sadece geleceğe bakan rolling
    # CRITICAL FIX: Geçmişe dayalı rolling features (last_Nh, last_Nd) KALMALI
    # Sadece geleceğe bakan rolling features kaldırılmalı
    # NOT: card1_amt_rolling_mean_20 gibi features - eğer shift(1) kullanıyorsa KALMALI
    # Ancak güvenlik için, "last_" içermeyen rolling features'ları kaldır
    rolling_leakage_patterns = [
        '_rolling_mean_',  # Sadece "last_" içermeyenler
        '_rolling_std_',
        '_rolling_min_',
        '_rolling_max_',
        '_rolling_sum_',
    ]
    for pattern in rolling_leakage_patterns:
        matching_features = [col for col in X_safe.columns if pattern in col.lower()]
        for feat in matching_features:
            # "last_" içeren rolling features KAL (geçmişe bakıyor, safe)
            # "last_" içermeyen rolling features KALDIR (geleceğe bakıyor olabilir)
            if 'last_' not in feat.lower():
                removed_features.append(feat)
    
    # Pattern 5: High-cardinality combination features that may overfit
    # These create very specific patterns that may not generalize
    # We'll be conservative and keep them, but note them for monitoring
    high_card_combo_patterns = [
        '_combo_count',  # Multi-column combinations
    ]
    # Keep combo_count features but note them (they're useful but need monitoring)
    
    # Remove duplicates
    removed_features = list(set(removed_features))
    
    # Remove features that don't exist (safety check)
    removed_features = [f for f in removed_features if f in X_safe.columns]
    
    # Remove the features
    if removed_features:
        X_safe = X_safe.drop(columns=removed_features, errors='ignore')
        if verbose:
            print(f"  Removed {len(removed_features)} leakage-prone features:")
            for feat in sorted(removed_features)[:20]:  # Show first 20
                print(f"    - {feat}")
            if len(removed_features) > 20:
                print(f"    ... and {len(removed_features) - 20} more")
    
    return X_safe, removed_features


def remove_low_variance_features(X: pd.DataFrame, 
                                 threshold: float = 0.01,
                                 verbose: bool = True) -> tuple:
    """
    Remove features with low variance.
    
    Why this reduces overfitting:
    - Features with no variance provide no information
    - Low variance features are likely noise and contribute to overfitting
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature dataframe
    threshold : float
        Variance threshold (features with variance < threshold will be removed)
    verbose : bool
        Whether to print information
        
    Returns
    -------
    X_selected : pd.DataFrame
        DataFrame with low variance features removed
    selected_features : list
        List of selected feature names
    """
    # Separate numeric and non-numeric features
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [col for col in X.columns if col not in numeric_cols]
    
    # Apply variance threshold only to numeric features
    if len(numeric_cols) > 0:
        X_numeric = X[numeric_cols]
        selector = VarianceThreshold(threshold=threshold)
        X_numeric_selected = selector.fit_transform(X_numeric)
        
        selected_numeric = X_numeric.columns[selector.get_support()].tolist()
        removed_numeric = [f for f in numeric_cols if f not in selected_numeric]
    else:
        selected_numeric = []
        removed_numeric = []
        X_numeric_selected = np.array([]).reshape(len(X), 0)
    
    # Keep all non-numeric features (categorical features)
    selected_features = non_numeric_cols + selected_numeric
    removed_features = removed_numeric
    
    # Combine numeric and non-numeric
    if len(non_numeric_cols) > 0:
        X_non_numeric = X[non_numeric_cols]
        if len(selected_numeric) > 0:
            X_selected = pd.concat([
                pd.DataFrame(X_numeric_selected, columns=selected_numeric, index=X.index),
                X_non_numeric
            ], axis=1)
        else:
            X_selected = X_non_numeric
    else:
        if len(selected_numeric) > 0:
            X_selected = pd.DataFrame(X_numeric_selected, columns=selected_numeric, index=X.index)
        else:
            X_selected = pd.DataFrame(index=X.index)
    
    # Reorder columns to match original order
    X_selected = X_selected[selected_features]
    
    if verbose:
        print(f"Removed {len(removed_features)} low variance features (numeric only)")
        print(f"Remaining features: {len(selected_features)} ({len(non_numeric_cols)} categorical, {len(selected_numeric)} numeric)")
    
    return X_selected, selected_features


def remove_correlated_features(X: pd.DataFrame,
                              threshold: float = 0.95,
                              verbose: bool = True) -> tuple:
    """
    Remove highly correlated features.
    
    Why this reduces overfitting:
    - Highly correlated features provide redundant information
    - Redundancy increases model complexity without adding signal
    - Removing one of a correlated pair reduces overfitting risk
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature dataframe (numeric only)
    threshold : float
        Correlation threshold (features with correlation > threshold will be removed)
    verbose : bool
        Whether to print information
        
    Returns
    -------
    X_selected : pd.DataFrame
        DataFrame with correlated features removed
    selected_features : list
        List of selected feature names
    """
    # Separate numeric and non-numeric features
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [col for col in X.columns if col not in numeric_cols]
    
    if len(numeric_cols) == 0:
        return X, list(X.columns)
    
    # Calculate correlation matrix for numeric features only
    X_numeric = X[numeric_cols]
    corr_matrix = X_numeric.corr().abs()
    
    # Find highly correlated pairs
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find features to remove (one from each highly correlated pair)
    to_remove = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    
    # Keep non-numeric features and non-correlated numeric features
    selected_numeric = [f for f in numeric_cols if f not in to_remove]
    selected_features = non_numeric_cols + selected_numeric
    
    # Create selected dataframe
    if len(non_numeric_cols) > 0:
        X_non_numeric = X[non_numeric_cols]
        if len(selected_numeric) > 0:
            X_selected = pd.concat([X[selected_numeric], X_non_numeric], axis=1)
        else:
            X_selected = X_non_numeric
    else:
        X_selected = X[selected_numeric]
    
    # Reorder columns
    X_selected = X_selected[selected_features]
    
    if verbose:
        print(f"Removed {len(to_remove)} highly correlated features")
        print(f"Remaining features: {len(selected_features)} ({len(non_numeric_cols)} categorical, {len(selected_numeric)} numeric)")
    
    return X_selected, selected_features


def select_features_by_importance(X: pd.DataFrame,
                                  y: pd.Series,
                                  model: Optional[lgb.Booster] = None,
                                  threshold: float = 0.001,
                                  target_count: Optional[int] = None,
                                  n_estimators: int = 100,
                                  early_stopping_rounds: int = 10,
                                  verbose: bool = True) -> tuple:
    """
    Select features based on LightGBM importance.
    
    Why this reduces overfitting:
    - Removes features that don't contribute to prediction
    - Focuses model on most informative features
    - Reduces model complexity
    
    Impact on recall vs precision:
    - May slightly reduce recall if important features are removed
    - Improves precision by removing noise features
    - Better generalization overall
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature dataframe
    y : pd.Series
        Target variable
    model : Optional[lgb.Booster]
        Pre-trained model (if None, will train a new one)
    threshold : float
        Importance threshold (features with importance < threshold will be removed)
    target_count : Optional[int]
        Target number of features (if set, will select top N features)
    n_estimators : int
        Number of estimators for feature selection model
    early_stopping_rounds : int
        Early stopping rounds for feature selection model
    verbose : bool
        Whether to print information
        
    Returns
    -------
    X_selected : pd.DataFrame
        DataFrame with selected features
    selected_features : list
        List of selected feature names
    """
    # Prepare X for LightGBM - convert object/category columns
    X_prepared = X.copy()
    categorical_features_list = []
    
    for col in X_prepared.columns:
        if X_prepared[col].dtype == 'object' or X_prepared[col].dtype.name == 'category':
            X_prepared[col] = X_prepared[col].astype('category')
            categorical_features_list.append(col)
        elif X_prepared[col].dtype in ['int8', 'int16', 'int32', 'int64']:
            if X_prepared[col].nunique() < 50:
                X_prepared[col] = X_prepared[col].astype('category')
                categorical_features_list.append(col)
    
    # Get categorical feature indices
    if categorical_features_list:
        cat_indices = [X_prepared.columns.get_loc(cat) for cat in categorical_features_list]
    else:
        cat_indices = None
    
    # Train model if not provided
    if model is None:
        train_data = lgb.Dataset(X_prepared, label=y, categorical_feature=cat_indices)
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'verbose': -1
        }
        # Use fewer rounds for faster feature selection
        model = lgb.train(params, train_data, num_boost_round=50)
    
    # Get feature importance - model might have been trained on different features
    # Get all feature names from model
    model_feature_names = model.feature_name()
    model_importance_raw = model.feature_importance(importance_type='gain')
    
    # Check if model features match X features exactly
    X_feature_set = set(X.columns.tolist())
    model_feature_set = set(model_feature_names)
    
    # Calculate overlap
    overlap = len(X_feature_set & model_feature_set)
    overlap_ratio = overlap / len(X_feature_set) if len(X_feature_set) > 0 else 0
    
    # If feature counts don't match OR less than 95% overlap, train a new model on current X
    # This is more strict to avoid dimension mismatches
    if len(model_feature_names) != len(X.columns) or overlap_ratio < 0.95:
        if verbose:
            print(f"Model features don't match X (model: {len(model_feature_names)}, X: {len(X.columns)}, overlap: {overlap_ratio:.1%}). Training new model on current features...")
        
        # Prepare X for LightGBM - convert object/category columns
        X_prepared = X.copy()
        categorical_features_list = []
        
        for col in X_prepared.columns:
            if X_prepared[col].dtype == 'object' or X_prepared[col].dtype.name == 'category':
                X_prepared[col] = X_prepared[col].astype('category')
                categorical_features_list.append(col)
            elif X_prepared[col].dtype in ['int8', 'int16', 'int32', 'int64']:
                if X_prepared[col].nunique() < 50:
                    X_prepared[col] = X_prepared[col].astype('category')
                    categorical_features_list.append(col)
        
        # Get categorical feature indices
        if categorical_features_list:
            cat_indices = [X_prepared.columns.get_loc(cat) for cat in categorical_features_list]
        else:
            cat_indices = None
        
        train_data = lgb.Dataset(X_prepared, label=y, categorical_feature=cat_indices)
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'verbose': -1
        }
        # Use fewer rounds for faster feature selection
        model = lgb.train(params, train_data, num_boost_round=50)
        importance = pd.Series(
            model.feature_importance(importance_type='gain'),
            index=X.columns
        ).sort_values(ascending=False)
    else:
        # Create importance series with model's feature names
        model_importance_series = pd.Series(
            model_importance_raw,
            index=model_feature_names
        )
        
        # Filter to only features that exist in X (should be all if overlap >= 95%)
        available_features = [f for f in model_feature_names if f in X.columns]
        if len(available_features) != len(X.columns):
            # Safety check: if not all features match, train new model
            if verbose:
                print(f"Warning: Feature mismatch detected. Training new model...")
            
            # Prepare X for LightGBM - convert object/category columns
            X_prepared = X.copy()
            categorical_features_list = []
            
            for col in X_prepared.columns:
                if X_prepared[col].dtype == 'object' or X_prepared[col].dtype.name == 'category':
                    X_prepared[col] = X_prepared[col].astype('category')
                    categorical_features_list.append(col)
                elif X_prepared[col].dtype in ['int8', 'int16', 'int32', 'int64']:
                    if X_prepared[col].nunique() < 50:
                        X_prepared[col] = X_prepared[col].astype('category')
                        categorical_features_list.append(col)
            
            # Get categorical feature indices
            if categorical_features_list:
                cat_indices = [X_prepared.columns.get_loc(cat) for cat in categorical_features_list]
            else:
                cat_indices = None
            
            train_data = lgb.Dataset(X_prepared, label=y, categorical_feature=cat_indices)
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'verbose': -1
            }
            model = lgb.train(params, train_data, num_boost_round=50)
            importance = pd.Series(
                model.feature_importance(importance_type='gain'),
                index=X.columns
            ).sort_values(ascending=False)
        else:
            # Use model importance, but only for features in X
            importance = model_importance_series[available_features].sort_values(ascending=False)
    
    # Select features based on threshold or target count
    if target_count is not None:
        # Select top N features
        selected_features = importance.head(target_count).index.tolist()
        if verbose:
            print(f"Selected top {target_count} features based on importance")
    else:
        # Select features above threshold
        selected_features = importance[importance >= threshold].index.tolist()
        if verbose:
            print(f"Selected {len(selected_features)} features based on importance (threshold={threshold})")
    
    # Ensure we have at least some features
    if len(selected_features) == 0:
        if verbose:
            print("Warning: No features selected. Keeping top 20 features.")
        selected_features = importance.head(20).index.tolist()
    
    # Select features
    X_selected = X[selected_features]
    
    if verbose:
        print(f"Top 10 features: {selected_features[:10]}")
    
    return X_selected, selected_features


def comprehensive_feature_selection(X: pd.DataFrame,
                                   y: pd.Series,
                                   variance_threshold: float = 0.01,
                                   correlation_threshold: float = 0.95,
                                   importance_threshold: float = 0.001,
                                   target_feature_count: Optional[int] = None,
                                   remove_leakage_features: bool = True,
                                   model_params_config: Optional[dict] = None,
                                   verbose: bool = True) -> tuple:
    """
    Comprehensive feature selection pipeline.
    
    PRODUCTION-SAFE: This function applies multiple feature selection steps:
    1. Remove leakage features (time-based, target leakage)
    2. Remove low variance features
    3. Remove highly correlated features
    4. Select features by importance
    
    Why this reduces overfitting:
    - Removes features that leak future/target information
    - Reduces model complexity by removing redundant/noisy features
    - Focuses model on most informative features
    
    Impact on recall vs precision:
    - May slightly reduce recall (fewer features)
    - Significantly improves precision (no leakage, less noise)
    - Better generalization (model learns real patterns)
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature dataframe
    y : pd.Series
        Target variable
    variance_threshold : float
        Variance threshold for low variance removal
    correlation_threshold : float
        Correlation threshold for correlated feature removal
    importance_threshold : float
        Importance threshold for importance-based selection
    target_feature_count : Optional[int]
        Target number of features (if set, will select top N)
    remove_leakage_features : bool
        Whether to remove leakage-prone features
    model_params_config : Optional[dict]
        Model parameters configuration (for dynamic thresholds)
    verbose : bool
        Whether to print information
        
    Returns
    -------
    X_selected : pd.DataFrame
        DataFrame with selected features
    selected_features : list
        List of selected feature names
    """
    if verbose:
        print("=" * 70)
        print("Comprehensive Feature Selection")
        print("=" * 70)
        print(f"Initial features: {len(X.columns)}")
    
    X_selected = X.copy()
    
    # Step 0: Remove leakage features (CRITICAL for production safety)
    # Note: Parameter name 'remove_leakage_features' (bool) conflicts with function name
    # Solution: Store function reference at module level before parameter shadows it
    # The function reference is stored above, now use the parameter value
    should_remove_leakage = remove_leakage_features  # This is the bool parameter
    if should_remove_leakage:
        if verbose:
            print("\n[Step 0] Removing leakage features (target leakage prevention)...")
        # Call the function using the stored reference (avoids name collision)
        # We need to get the function from the module namespace
        import sys
        current_module = sys.modules[__name__]
        remove_leakage_func = getattr(current_module, 'remove_leakage_features')
        X_selected, removed_leakage = remove_leakage_func(X_selected, verbose=verbose)
        if verbose:
            print(f"  Removed {len(removed_leakage)} leakage features")
            print(f"  Remaining features: {len(X_selected.columns)}")
    
    # Step 1: Remove low variance features
    if verbose:
        print("\n[Step 1] Removing low variance features...")
    X_selected, _ = remove_low_variance_features(
        X_selected, 
        threshold=variance_threshold,
        verbose=verbose
    )
    if verbose:
        print(f"Remaining features: {len(X_selected.columns)}")
    
    # Step 2: Remove highly correlated features
    if verbose:
        print("\n[Step 2] Removing highly correlated features...")
    X_selected, _ = remove_correlated_features(
        X_selected,
        threshold=correlation_threshold,
        verbose=verbose
    )
    if verbose:
        print(f"Remaining features: {len(X_selected.columns)}")
    
    # Step 3: Select features by importance
    if verbose:
        print("\n[Step 3] Selecting features by importance...")
    
    # Adjust thresholds based on model_params_config if provided
    if model_params_config:
        importance_threshold = model_params_config.get('feature_selection_importance_threshold', importance_threshold)
        if target_feature_count is None:
            target_feature_count = model_params_config.get('feature_selection_target_count', None)
    
    X_selected, selected_features = select_features_by_importance(
        X_selected,
        y,
        threshold=importance_threshold,
        target_count=target_feature_count,
        verbose=verbose
    )
    
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Final features: {len(selected_features)} ({len(selected_features)/len(X.columns)*100:.1f}% of original)")
        if should_remove_leakage:
            print("⚠️  Note: Leakage features (fraud_rate, time_since_last, velocity) were removed to prevent unrealistic performance")
        print("=" * 70)
    
    return X_selected, selected_features
