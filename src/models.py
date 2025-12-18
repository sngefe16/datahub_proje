"""
Machine learning models for IEEE Fraud Detection project.
Includes baseline and advanced models with proper handling of class imbalance.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from typing import Dict, Optional


class FraudDetectionModel:
    """Base class for fraud detection models."""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.model = None
        self.feature_importance_ = None
        
    def fit(self, X_train, y_train, **kwargs):
        """Train the model."""
        raise NotImplementedError
        
    def predict(self, X):
        """Predict class labels."""
        raise NotImplementedError
        
    def predict_proba(self, X):
        """Predict class probabilities."""
        raise NotImplementedError


class BaselineLogisticRegression(FraudDetectionModel):
    """Baseline Logistic Regression model."""
    
    def __init__(self, class_weight: str = 'balanced', **kwargs):
        super().__init__('LogisticRegression')
        self.model = LogisticRegression(
            class_weight=class_weight,
            max_iter=1000,
            random_state=42,
            **kwargs
        )
    
    def fit(self, X_train, y_train, **kwargs):
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


class BaselineRandomForest(FraudDetectionModel):
    """Baseline Random Forest model."""
    
    def __init__(self, n_estimators: int = 100, 
                 class_weight: str = 'balanced',
                 max_depth: int = 10,
                 **kwargs):
        super().__init__('RandomForest')
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight=class_weight,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
            **kwargs
        )
    
    def fit(self, X_train, y_train, **kwargs):
        self.model.fit(X_train, y_train)
        self.feature_importance_ = pd.Series(
            self.model.feature_importances_,
            index=X_train.columns if hasattr(X_train, 'columns') else None
        )
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


class XGBoostModel(FraudDetectionModel):
    """XGBoost model optimized for fraud detection."""
    
    def __init__(self, 
                 n_estimators: int = 1000,
                 max_depth: int = 6,
                 learning_rate: float = 0.01,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 scale_pos_weight: Optional[float] = None,
                 **kwargs):
        super().__init__('XGBoost')
        
        # Calculate scale_pos_weight if not provided
        if scale_pos_weight is None:
            # This should be calculated from training data
            scale_pos_weight = 1.0
        
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='auc',
            tree_method='hist',
            **kwargs
        )
    
    def fit(self, X_train, y_train, 
            X_val=None, y_val=None,
            early_stopping_rounds=50,
            verbose=100,
            **kwargs):
        
        # Calculate scale_pos_weight from training data
        fraud_count = y_train.sum()
        non_fraud_count = len(y_train) - fraud_count
        self.model.set_params(scale_pos_weight=non_fraud_count / fraud_count)
        
        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['early_stopping_rounds'] = early_stopping_rounds
            fit_params['verbose'] = verbose
        
        self.model.fit(X_train, y_train, **fit_params)
        
        self.feature_importance_ = pd.Series(
            self.model.feature_importances_,
            index=X_train.columns if hasattr(X_train, 'columns') else None
        )
        
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


class LightGBMModel(FraudDetectionModel):
    """LightGBM model optimized for fraud detection."""
    
    def __init__(self,
                 n_estimators: int = 1000,
                 max_depth: int = 6,
                 learning_rate: float = 0.01,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 class_weight: Optional[Dict] = None,
                 **kwargs):
        super().__init__('LightGBM')
        
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            class_weight=class_weight,
            random_state=42,
            objective='binary',
            metric='auc',
            boosting_type='gbdt',
            n_jobs=-1,
            verbose=-1,
            **kwargs
        )
    
    def fit(self, X_train, y_train,
            X_val=None, y_val=None,
            early_stopping_rounds=50,
            verbose=100,
            **kwargs):
        
        # Calculate class_weight if not provided
        if self.model.class_weight is None:
            fraud_count = y_train.sum()
            non_fraud_count = len(y_train) - fraud_count
            self.model.set_params(
                class_weight={0: 1.0, 1: non_fraud_count / fraud_count}
            )
        
        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['early_stopping_rounds'] = early_stopping_rounds
            fit_params['callbacks'] = [lgb.early_stopping(early_stopping_rounds),
                                      lgb.log_evaluation(verbose)]
        
        self.model.fit(X_train, y_train, **fit_params)
        
        self.feature_importance_ = pd.Series(
            self.model.feature_importances_,
            index=X_train.columns if hasattr(X_train, 'columns') else None
        )
        
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


class CatBoostModel(FraudDetectionModel):
    """CatBoost model optimized for fraud detection."""
    
    def __init__(self,
                 n_estimators: int = 1000,
                 max_depth: int = 6,
                 learning_rate: float = 0.01,
                 class_weights: Optional[list] = None,
                 cat_features: Optional[list] = None,
                 **kwargs):
        super().__init__('CatBoost')
        
        self.model = CatBoostClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            class_weights=class_weights,
            cat_features=cat_features,
            random_state=42,
            eval_metric='AUC',
            verbose=100,
            **kwargs
        )
    
    def fit(self, X_train, y_train,
            X_val=None, y_val=None,
            early_stopping_rounds=50,
            **kwargs):
        
        # Calculate class_weights if not provided
        if self.model.class_weights is None:
            fraud_count = y_train.sum()
            non_fraud_count = len(y_train) - fraud_count
            self.model.set_params(
                class_weights=[1.0, non_fraud_count / fraud_count]
            )
        
        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = (X_val, y_val)
            fit_params['early_stopping_rounds'] = early_stopping_rounds
        
        self.model.fit(X_train, y_train, **fit_params)
        
        self.feature_importance_ = pd.Series(
            self.model.feature_importances_,
            index=X_train.columns if hasattr(X_train, 'columns') else None
        )
        
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


def get_model(model_name: str, **kwargs) -> FraudDetectionModel:
    """
    Factory function to get a model by name.
    
    Parameters
    ----------
    model_name : str
        Name of the model ('logistic', 'rf', 'xgb', 'lgb', 'catboost')
    **kwargs
        Model-specific parameters
        
    Returns
    -------
    model : FraudDetectionModel
        Model instance
    """
    models = {
        'logistic': BaselineLogisticRegression,
        'rf': BaselineRandomForest,
        'xgb': XGBoostModel,
        'lgb': LightGBMModel,
        'catboost': CatBoostModel
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    
    return models[model_name.lower()](**kwargs)









