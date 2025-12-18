"""
Feature selection utilities for fourth view model.
Removes unnecessary features to improve model performance and reduce overfitting.
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


def remove_low_variance_features(X: pd.DataFrame, 
                                 threshold: float = 0.01,
                                 verbose: bool = True) -> tuple:
    """
    Remove features with low variance.
    
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
    # Select only numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_numeric = X[numeric_cols].copy()
    
    # Calculate correlation matrix
    corr_matrix = X_numeric.corr().abs()
    
    # Find pairs of highly correlated features
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find features to remove
    to_remove = [column for column in upper_triangle.columns 
                if any(upper_triangle[column] > threshold)]
    
    # Keep non-numeric columns
    non_numeric_cols = [col for col in X.columns if col not in numeric_cols]
    
    # Remove correlated features
    selected_numeric = [col for col in numeric_cols if col not in to_remove]
    selected_features = non_numeric_cols + selected_numeric
    
    if verbose:
        print(f"Removed {len(to_remove)} highly correlated features")
        print(f"Remaining features: {len(selected_features)}")
    
    return X[selected_features], selected_features


def select_features_by_importance(X: pd.DataFrame,
                                  y: pd.Series,
                                  model: Optional[lgb.Booster] = None,
                                  top_k: Optional[int] = None,
                                  importance_threshold: Optional[float] = None,
                                  verbose: bool = True) -> tuple:
    """
    Select features based on LightGBM feature importance.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature dataframe
    y : pd.Series
        Target series
    model : lgb.Booster, optional
        Trained LightGBM model. If None, will train a simple model
    top_k : int, optional
        Select top K features
    importance_threshold : float, optional
        Select features with importance > threshold
    verbose : bool
        Whether to print information
        
    Returns
    -------
    X_selected : pd.DataFrame
        DataFrame with selected features
    selected_features : list
        List of selected feature names
    """
    if model is None:
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
        
        # Train a simple model for feature importance (faster for feature selection)
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
            # Use importance from model, filtered to available features
            importance = model_importance_series[available_features].sort_values(ascending=False)
    
    # Select features
    if top_k is not None:
        selected_features = importance.head(top_k).index.tolist()
    elif importance_threshold is not None:
        max_importance = importance.max()
        threshold_value = max_importance * importance_threshold
        selected_features = importance[importance >= threshold_value].index.tolist()
    else:
        # Default: select top 80% of features
        top_pct = int(len(importance) * 0.8)
        selected_features = importance.head(top_pct).index.tolist()
    
    if verbose:
        print(f"Selected {len(selected_features)} features based on importance")
        print(f"Top 10 features: {selected_features[:10]}")
    
    return X[selected_features], selected_features


def select_features_statistical(X: pd.DataFrame,
                               y: pd.Series,
                               k: int = 100,
                               score_func: str = 'f_classif',
                               verbose: bool = True) -> tuple:
    """
    Select features using statistical tests (SelectKBest).
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature dataframe
    y : pd.Series
        Target series
    k : int
        Number of top features to select
    score_func : str
        'f_classif' or 'mutual_info'
    verbose : bool
        Whether to print information
        
    Returns
    -------
    X_selected : pd.DataFrame
        DataFrame with selected features
    selected_features : list
        List of selected feature names
    """
    # Select only numeric columns for statistical tests
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_numeric = X[numeric_cols].copy()
    
    # Choose score function
    if score_func == 'f_classif':
        score_func_obj = f_classif
    elif score_func == 'mutual_info':
        score_func_obj = mutual_info_classif
    else:
        raise ValueError(f"Unknown score_func: {score_func}")
    
    # Select K best features
    k_actual = min(k, len(numeric_cols))
    selector = SelectKBest(score_func=score_func_obj, k=k_actual)
    X_selected_numeric = selector.fit_transform(X_numeric, y)
    
    selected_numeric = X_numeric.columns[selector.get_support()].tolist()
    
    # Keep non-numeric columns (categorical features)
    non_numeric_cols = [col for col in X.columns if col not in numeric_cols]
    selected_features = non_numeric_cols + selected_numeric
    
    if verbose:
        print(f"Selected {len(selected_features)} features using {score_func}")
        print(f"  - Numeric: {len(selected_numeric)}")
        print(f"  - Categorical: {len(non_numeric_cols)}")
    
    return X[selected_features], selected_features


def comprehensive_feature_selection(X: pd.DataFrame,
                                   y: pd.Series,
                                   model: Optional[lgb.Booster] = None,
                                   variance_threshold: float = 0.01,
                                   correlation_threshold: float = 0.95,
                                   importance_top_k: Optional[int] = None,
                                   importance_threshold: float = 0.01,
                                   verbose: bool = True) -> tuple:
    """
    Comprehensive feature selection pipeline.
    
    Steps:
    1. Remove low variance features
    2. Remove highly correlated features
    3. Select features by importance
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature dataframe
    y : pd.Series
        Target series
    model : lgb.Booster, optional
        Trained LightGBM model
    variance_threshold : float
        Variance threshold for step 1
    correlation_threshold : float
        Correlation threshold for step 2
    importance_top_k : int, optional
        Top K features to select in step 3
    importance_threshold : float
        Importance threshold (relative to max) for step 3
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
        print(f"Initial features: {X.shape[1]}")
    
    # Step 1: Remove low variance features
    if verbose:
        print("\n[Step 1] Removing low variance features...")
    X_step1, features_step1 = remove_low_variance_features(
        X, threshold=variance_threshold, verbose=verbose
    )
    
    # Step 2: Remove highly correlated features
    if verbose:
        print("\n[Step 2] Removing highly correlated features...")
    X_step2, features_step2 = remove_correlated_features(
        X_step1, threshold=correlation_threshold, verbose=verbose
    )
    
    # Step 3: Select by importance
    if verbose:
        print("\n[Step 3] Selecting features by importance...")
    X_final, features_final = select_features_by_importance(
        X_step2, y, model=model,
        top_k=importance_top_k,
        importance_threshold=importance_threshold,
        verbose=verbose
    )
    
    if verbose:
        print("\n" + "=" * 70)
        print(f"Final features: {X_final.shape[1]} ({100 * X_final.shape[1] / X.shape[1]:.1f}% of original)")
        print("=" * 70)
    
    return X_final, features_final

