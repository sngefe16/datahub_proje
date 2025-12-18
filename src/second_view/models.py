"""
Machine learning models for second view fraud detection.
Uses LightGBM with threshold tuning (0.6-0.7) for better precision.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')


class SecondViewLightGBM:
    """
    LightGBM model optimized for second view features.
    Includes threshold tuning for better precision-recall balance.
    """
    
    def __init__(self, 
                 n_estimators: int = 2000,
                 max_depth: int = 7,
                 learning_rate: float = 0.01,
                 num_leaves: int = 31,
                 min_child_samples: int = 20,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 0.1,
                 scale_pos_weight: Optional[float] = None,
                 random_state: int = 42,
                 threshold: float = 0.65,  # Default threshold (between 0.6-0.7)
                 **kwargs):
        """
        Initialize LightGBM model with threshold tuning.
        
        Parameters
        ----------
        n_estimators : int
            Number of boosting rounds
        max_depth : int
            Maximum tree depth
        learning_rate : float
            Learning rate
        num_leaves : int
            Number of leaves in one tree
        min_child_samples : int
            Minimum number of data needed in a child
        subsample : float
            Subsample ratio of the training instance
        colsample_bytree : float
            Subsample ratio of columns when constructing each tree
        reg_alpha : float
            L1 regularization term
        reg_lambda : float
            L2 regularization term
        scale_pos_weight : float, optional
            Weight of positive class. If None, will be calculated from data
        random_state : int
            Random seed
        threshold : float
            Classification threshold (default: 0.65, range: 0.6-0.7)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.threshold = threshold
        self.model = None
        self.feature_importance_ = None
        self.categorical_features = None
        
    def fit(self, X_train, y_train, 
            categorical_features: Optional[List[str]] = None,
            eval_set: Optional[tuple] = None,
            early_stopping_rounds: int = 100,
            verbose: int = 100):
        """
        Train the model.
        
        Parameters
        ----------
        X_train : pd.DataFrame or np.ndarray
            Training features
        y_train : np.ndarray
            Training target
        categorical_features : list, optional
            List of categorical feature names
        eval_set : tuple, optional
            (X_val, y_val) for early stopping
        early_stopping_rounds : int
            Early stopping rounds
        verbose : int
            Verbosity level
        """
        # Calculate scale_pos_weight if not provided
        if self.scale_pos_weight is None:
            fraud_count = y_train.sum()
            non_fraud_count = len(y_train) - fraud_count
            self.scale_pos_weight = non_fraud_count / fraud_count if fraud_count > 0 else 1.0
        
        # Identify categorical features
        if categorical_features is None and hasattr(X_train, 'columns'):
            categorical_features = []
            for col in X_train.columns:
                if X_train[col].dtype == 'object' or X_train[col].dtype.name == 'category':
                    categorical_features.append(col)
                elif X_train[col].dtype in ['int8', 'int16', 'int32', 'int64']:
                    if X_train[col].nunique() < 50:
                        categorical_features.append(col)
        
        self.categorical_features = categorical_features
        
        # Convert object dtype to category for LightGBM
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.copy()
            for col in X_train.columns:
                if X_train[col].dtype == 'object':
                    X_train[col] = X_train[col].astype('category')
            
            if categorical_features:
                cat_indices = [X_train.columns.get_loc(cat) for cat in categorical_features if cat in X_train.columns]
            else:
                cat_indices = 'auto'
        else:
            cat_indices = 'auto'
        
        # Model parameters
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': self.num_leaves,
            'learning_rate': self.learning_rate,
            'feature_fraction': self.colsample_bytree,
            'bagging_fraction': self.subsample,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': self.random_state,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'max_depth': self.max_depth,
            'min_child_samples': self.min_child_samples,
            'scale_pos_weight': self.scale_pos_weight
        }
        
        # Convert validation set object dtypes to category if needed
        if eval_set is not None:
            X_val, y_val = eval_set
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.copy()
                for col in X_val.columns:
                    if X_val[col].dtype == 'object':
                        X_val[col] = X_val[col].astype('category')
        
        # Create LightGBM dataset
        train_data = lgb.Dataset(X_train, label=y_train, 
                                categorical_feature=cat_indices if isinstance(cat_indices, list) else None,
                                free_raw_data=False)
        
        # Validation set
        valid_sets = [train_data]
        valid_names = ['train']
        if eval_set is not None:
            valid_data = lgb.Dataset(X_val, label=y_val,
                                    categorical_feature=cat_indices if isinstance(cat_indices, list) else None,
                                    free_raw_data=False)
            valid_sets.append(valid_data)
            valid_names.append('valid')
        
        # Train model
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=verbose > 0),
                lgb.log_evaluation(verbose) if verbose > 0 else lgb.log_evaluation(0)
            ]
        )
        
        # Feature importance
        if hasattr(X_train, 'columns'):
            self.feature_importance_ = pd.Series(
                self.model.feature_importance(importance_type='gain'),
                index=X_train.columns
            ).sort_values(ascending=False)
        
        return self
    
    def predict(self, X, threshold: Optional[float] = None):
        """
        Predict class labels using custom threshold.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Features
        threshold : float, optional
            Classification threshold. If None, uses self.threshold (default: 0.65)
            
        Returns
        -------
        predictions : np.ndarray
            Binary predictions
        """
        if threshold is None:
            threshold = self.threshold
        return (self.predict_proba(X) > threshold).astype(int)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Convert object dtype to category for prediction
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = X[col].astype('category')
        
        return self.model.predict(X, num_iteration=self.model.best_iteration)


def train_second_view_model(X_train, y_train,
                            X_val=None, y_val=None,
                            model_params: Optional[Dict] = None,
                            categorical_features: Optional[List[str]] = None,
                            threshold: float = 0.65) -> SecondViewLightGBM:
    """
    Train second view model with threshold tuning.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : np.ndarray
        Training target
    X_val : pd.DataFrame, optional
        Validation features
    y_val : np.ndarray, optional
        Validation target
    model_params : dict, optional
        Model parameters
    categorical_features : list, optional
        List of categorical feature names
    threshold : float
        Classification threshold (default: 0.65, recommended: 0.6-0.7)
        
    Returns
    -------
    model : SecondViewLightGBM
        Trained model
    """
    # Default parameters
    if model_params is None:
        model_params = {}
    
    # Add threshold to model params
    model_params['threshold'] = threshold
    
    # Initialize model
    model = SecondViewLightGBM(**model_params)
    
    # Prepare validation set
    eval_set = None
    if X_val is not None and y_val is not None:
        eval_set = (X_val, y_val)
    
    # Train model
    model.fit(X_train, y_train,
              categorical_features=categorical_features,
              eval_set=eval_set)
    
    return model

