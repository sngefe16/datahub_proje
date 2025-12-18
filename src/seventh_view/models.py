"""
Machine learning models for seventh view fraud detection.
Targets AUC-ROC > 0.95, Recall > 0.9, and Precision > 0.5.
Uses enhanced LightGBM with overfitting control, optimized feature selection,
improved SMOTE, and better threshold optimization.
Based on sixth view improvements and MODEL_COMPARISON2.md recommendations.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Optional, List
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Try to import Optuna and Hyperopt
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available. Install with: pip install optuna")

try:
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False
    print("Warning: Hyperopt not available. Install with: pip install hyperopt")


class SeventhViewLightGBM:
    """
    LightGBM model optimized for seventh view features.
    Enhanced with overfitting control, improved regularization, and optimized capacity.
    Based on sixth view improvements: reduced overfitting, better feature selection, improved SMOTE.
    """
    
    def __init__(self, 
                 n_estimators: int = 2000,  # Keep same for 1 hour training
                 max_depth: int = 11,  # Reduced from 13 for overfitting control (was 9 default, Optuna will tune 11-12)
                 learning_rate: float = 0.02,  # Keep same
                 num_leaves: int = 200,  # Reduced from 255 for overfitting control (was 127 default, Optuna will tune 200-220)
                 min_child_samples: int = 20,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 reg_alpha: float = 0.5,  # Increased from 0.1 for overfitting control
                 reg_lambda: float = 0.5,  # Increased from 0.1 for overfitting control
                 scale_pos_weight: Optional[float] = None,
                 random_state: int = 42,
                 threshold: float = 0.65,
                 **kwargs):
        """Initialize LightGBM model."""
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
            early_stopping_rounds: int = 50,  # More aggressive early stopping for overfitting control
            verbose: int = 100):
        """Train the model."""
        if self.scale_pos_weight is None:
            fraud_count = y_train.sum()
            non_fraud_count = len(y_train) - fraud_count
            self.scale_pos_weight = non_fraud_count / fraud_count if fraud_count > 0 else 1.0
        
        if categorical_features is None and hasattr(X_train, 'columns'):
            categorical_features = []
            for col in X_train.columns:
                if X_train[col].dtype == 'object' or X_train[col].dtype.name == 'category':
                    categorical_features.append(col)
                elif X_train[col].dtype in ['int8', 'int16', 'int32', 'int64']:
                    if X_train[col].nunique() < 50:
                        categorical_features.append(col)
        
        self.categorical_features = categorical_features
        
        # Store categorical categories for prediction alignment
        self.categorical_categories = {}
        
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.copy()
            for col in X_train.columns:
                if X_train[col].dtype == 'object':
                    X_train[col] = X_train[col].astype('category')
            
            # Store categories for each categorical feature
            if categorical_features:
                for col in categorical_features:
                    if col in X_train.columns:
                        if X_train[col].dtype.name == 'category':
                            # Convert categories to string to avoid type mixing issues
                            self.categorical_categories[col] = [str(c) for c in X_train[col].cat.categories.tolist()]
                        else:
                            # Store unique values as categories (convert to string)
                            unique_vals = X_train[col].dropna().unique().tolist()
                            self.categorical_categories[col] = [str(v) for v in unique_vals]
                
                cat_indices = [X_train.columns.get_loc(cat) for cat in categorical_features if cat in X_train.columns]
            else:
                cat_indices = 'auto'
        else:
            cat_indices = 'auto'
        
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
        
        if eval_set is not None:
            X_val, y_val = eval_set
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.copy()
                # Ensure validation set has same columns in same order as train set
                if list(X_val.columns) != list(X_train.columns):
                    X_val = X_val[X_train.columns]
                
                # Ensure validation set has same categorical features as train set
                # Align categories: union of train and validation categories
                for col in X_val.columns:
                    if col in X_train.columns and col in categorical_features:
                        # Get all unique values from both train and validation
                        train_vals = X_train[col].dropna().astype(str).unique().tolist()
                        val_vals = X_val[col].dropna().astype(str).unique().tolist()
                        # Combine all values, replace NaN strings with 'missing'
                        all_vals_raw = [v if v not in ['nan', 'None', 'NaN'] else 'missing' for v in train_vals + val_vals]
                        # Create unique sorted list - use set to ensure uniqueness
                        all_vals_set = set(all_vals_raw)
                        # Ensure 'missing' is in the set
                        all_vals_set.add('missing')
                        # Convert to sorted list
                        all_vals = sorted(list(all_vals_set))
                        
                        # Double-check uniqueness before creating Categorical
                        if len(all_vals) != len(set(all_vals)):
                            # Remove duplicates if any
                            all_vals = sorted(list(set(all_vals)))
                        
                        # Convert both to categorical with same categories
                        # Handle NaN values properly
                        X_train[col] = X_train[col].astype(str)
                        X_train[col] = X_train[col].replace(['nan', 'None', 'NaN'], 'missing')
                        X_train[col] = pd.Categorical(X_train[col], categories=all_vals)
                        
                        X_val[col] = X_val[col].astype(str)
                        X_val[col] = X_val[col].replace(['nan', 'None', 'NaN'], 'missing')
                        X_val[col] = pd.Categorical(X_val[col], categories=all_vals)
        
        # Use feature names instead of indices for categorical features to avoid mismatch
        if isinstance(cat_indices, list) and isinstance(X_train, pd.DataFrame):
            cat_feature_names = [X_train.columns[i] for i in cat_indices if i < len(X_train.columns)]
        else:
            cat_feature_names = None
        
        train_data = lgb.Dataset(X_train, label=y_train, 
                                categorical_feature=cat_feature_names,
                                free_raw_data=False)
        
        valid_sets = [train_data]
        valid_names = ['train']
        if eval_set is not None:
            # Use same categorical feature names for validation set
            valid_data = lgb.Dataset(X_val, label=y_val,
                                    categorical_feature=cat_feature_names,
                                    free_raw_data=False)
            valid_sets.append(valid_data)
            valid_names.append('valid')
        
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
        
        if hasattr(X_train, 'columns'):
            self.feature_importance_ = pd.Series(
                self.model.feature_importance(importance_type='gain'),
                index=X_train.columns
            ).sort_values(ascending=False)
        
        return self
    
    def predict(self, X, threshold: Optional[float] = None):
        """Predict class labels using custom threshold."""
        if threshold is None:
            threshold = self.threshold
        return (self.predict_proba(X) > threshold).astype(int)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            # Ensure same columns and order as training
            if hasattr(self, 'categorical_features') and self.categorical_features:
                # Align categorical features with training categories
                for col in X.columns:
                    if col in self.categorical_features and col in self.categorical_categories:
                        # Get training categories (already strings from fit)
                        train_categories = self.categorical_categories[col]
                        
                        # Convert to string first
                        X[col] = X[col].astype(str)
                        X[col] = X[col].replace(['nan', 'None', 'NaN', '<NA>'], 'missing')
                        
                        # Ensure 'missing' is in categories
                        all_categories = list(train_categories) + ['missing']
                        # Remove duplicates and sort (all are strings now)
                        all_categories = sorted(list(set(all_categories)))
                        
                        # Convert to categorical with same categories as training
                        X[col] = pd.Categorical(X[col], categories=all_categories)
                    elif X[col].dtype == 'object':
                        X[col] = X[col].astype('category')
        
        return self.model.predict(X, num_iteration=self.model.best_iteration)


def optimize_hyperparameters_optuna(X_train, y_train,
                                     X_val=None, y_val=None,
                                     categorical_features: Optional[List[str]] = None,
                                     n_trials: int = 50,  # Optimized: reduced from 100 for faster tuning
                                     timeout: Optional[int] = None,
                                     use_cv: bool = True,
                                     cv_folds: int = 3) -> Dict:  # Optimized: reduced from 5 to 3 for faster CV
    """
    Optimize hyperparameters using Optuna.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : np.ndarray
        Training target
    X_val : pd.DataFrame
        Validation features
    y_val : np.ndarray
        Validation target
    categorical_features : list, optional
        List of categorical feature names
    n_trials : int
        Number of optimization trials
    timeout : int, optional
        Timeout in seconds
        
    Returns
    -------
    best_params : dict
        Best hyperparameters found
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is not installed. Install with: pip install optuna")
    
    # Calculate scale_pos_weight
    fraud_count = y_train.sum()
    non_fraud_count = len(y_train) - fraud_count
    scale_pos_weight = non_fraud_count / fraud_count if fraud_count > 0 else 1.0
    
    # Prepare categorical features
    if categorical_features is None and hasattr(X_train, 'columns'):
        categorical_features = []
        for col in X_train.columns:
            if X_train[col].dtype == 'object' or X_train[col].dtype.name == 'category':
                categorical_features.append(col)
            elif X_train[col].dtype in ['int8', 'int16', 'int32', 'int64']:
                if X_train[col].nunique() < 50:
                    categorical_features.append(col)
    
    # Convert categorical features and align categories
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.copy()
        X_val = X_val.copy()
        
        # Align categorical features: ensure both train and validation have same categories
        for col in X_train.columns:
            if col in categorical_features:
                # Get all unique values from both train and validation
                train_vals = X_train[col].dropna().astype(str).unique().tolist()
                val_vals = X_val[col].dropna().astype(str).unique().tolist()
                # Combine all values, replace NaN strings with 'missing'
                all_vals_raw = [v if v not in ['nan', 'None', 'NaN'] else 'missing' for v in train_vals + val_vals]
                # Create unique sorted list
                all_vals = sorted(list(set(all_vals_raw)))
                
                # Ensure 'missing' is in the list
                if 'missing' not in all_vals:
                    all_vals.append('missing')
                    all_vals = sorted(all_vals)
                
                # Convert both to categorical with same categories
                X_train[col] = X_train[col].astype(str)
                X_train[col] = X_train[col].replace(['nan', 'None', 'NaN'], 'missing')
                X_train[col] = pd.Categorical(X_train[col], categories=all_vals)
                
                X_val[col] = X_val[col].astype(str)
                X_val[col] = X_val[col].replace(['nan', 'None', 'NaN'], 'missing')
                X_val[col] = pd.Categorical(X_val[col], categories=all_vals)
            elif X_train[col].dtype == 'object':
                X_train[col] = X_train[col].astype('category')
                X_val[col] = X_val[col].astype('category')
        
        if categorical_features:
            cat_indices = [X_train.columns.get_loc(cat) for cat in categorical_features if cat in X_train.columns]
        else:
            cat_indices = 'auto'
    else:
        cat_indices = 'auto'
    
    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 200, 220),  # Reduced for overfitting control (was 127-255)
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'max_depth': trial.suggest_int('max_depth', 11, 12),  # Reduced for overfitting control (was 8-14)
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 5.0, log=True),  # Increased min for overfitting control (was 0.01-10.0)
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0, log=True),  # Increased min for overfitting control (was 0.01-10.0)
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'verbose': -1
        }
        
        if use_cv and X_val is None:
            # Use CV for hyperparameter tuning
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scores = []
            
            for train_idx, val_idx in skf.split(X_train, y_train):
                X_train_fold = X_train.iloc[train_idx] if isinstance(X_train, pd.DataFrame) else X_train[train_idx]
                X_val_fold = X_train.iloc[val_idx] if isinstance(X_train, pd.DataFrame) else X_train[val_idx]
                y_train_fold = y_train[train_idx]
                y_val_fold = y_train[val_idx]
                
                train_data = lgb.Dataset(X_train_fold, label=y_train_fold,
                                        categorical_feature=cat_indices if isinstance(cat_indices, list) else None,
                                        free_raw_data=False)
                valid_data = lgb.Dataset(X_val_fold, label=y_val_fold,
                                        categorical_feature=cat_indices if isinstance(cat_indices, list) else None,
                                        free_raw_data=False)
                
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=2000,  # Optimized for 3-4 hour training: reduced from 2500
                    valid_sets=[valid_data],
                    valid_names=['valid'],
                    callbacks=[
                        lgb.early_stopping(50, verbose=False),  # More aggressive early stopping
                        lgb.log_evaluation(0)
                    ]
                )
                
                y_pred_proba = model.predict(X_val_fold, num_iteration=model.best_iteration)
                auc = roc_auc_score(y_val_fold, y_pred_proba)
                cv_scores.append(auc)
            
            return np.mean(cv_scores)  # Return mean CV score
        else:
            # Use validation set (faster)
            train_data = lgb.Dataset(X_train, label=y_train,
                                    categorical_feature=cat_indices if isinstance(cat_indices, list) else None,
                                    free_raw_data=False)
            valid_data = lgb.Dataset(X_val, label=y_val,
                                    categorical_feature=cat_indices if isinstance(cat_indices, list) else None,
                                    free_raw_data=False)
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=2500,  # Optimized: reduced for faster training
                valid_sets=[valid_data],
                valid_names=['valid'],
                callbacks=[
                    lgb.early_stopping(100, verbose=False),
                    lgb.log_evaluation(0)
                ]
            )
            
            y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
            auc = roc_auc_score(y_val, y_pred_proba)
            
            return auc
    
    study = optuna.create_study(direction='maximize', study_name='seventh_view_optuna')
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    best_params = study.best_params.copy()
    best_params['scale_pos_weight'] = scale_pos_weight
    best_params['random_state'] = 42
    best_params['objective'] = 'binary'
    best_params['metric'] = 'auc'
    best_params['boosting_type'] = 'gbdt'
    best_params['bagging_freq'] = int(best_params.get('bagging_freq', 5))
    
    return best_params


def optimize_hyperparameters_hyperopt(X_train, y_train,
                                      X_val, y_val,
                                      categorical_features: Optional[List[str]] = None,
                                      max_evals: int = 50) -> Dict:
    """
    Optimize hyperparameters using Hyperopt.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : np.ndarray
        Training target
    X_val : pd.DataFrame
        Validation features
    y_val : np.ndarray
        Validation target
    categorical_features : list, optional
        List of categorical feature names
    max_evals : int
        Maximum number of evaluations
        
    Returns
    -------
    best_params : dict
        Best hyperparameters found
    """
    if not HYPEROPT_AVAILABLE:
        raise ImportError("Hyperopt is not installed. Install with: pip install hyperopt")
    
    from sklearn.metrics import roc_auc_score
    
    # Calculate scale_pos_weight
    fraud_count = y_train.sum()
    non_fraud_count = len(y_train) - fraud_count
    scale_pos_weight = non_fraud_count / fraud_count if fraud_count > 0 else 1.0
    
    # Prepare categorical features
    if categorical_features is None and hasattr(X_train, 'columns'):
        categorical_features = []
        for col in X_train.columns:
            if X_train[col].dtype == 'object' or X_train[col].dtype.name == 'category':
                categorical_features.append(col)
            elif X_train[col].dtype in ['int8', 'int16', 'int32', 'int64']:
                if X_train[col].nunique() < 50:
                    categorical_features.append(col)
    
    # Convert categorical features and align categories
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.copy()
        X_val = X_val.copy()
        
        # Align categorical features: ensure both train and validation have same categories
        for col in X_train.columns:
            if col in categorical_features:
                # Get all unique values from both train and validation
                train_vals = X_train[col].dropna().astype(str).unique().tolist()
                val_vals = X_val[col].dropna().astype(str).unique().tolist()
                # Combine all values, replace NaN strings with 'missing'
                all_vals_raw = [v if v not in ['nan', 'None', 'NaN'] else 'missing' for v in train_vals + val_vals]
                # Create unique sorted list
                all_vals = sorted(list(set(all_vals_raw)))
                
                # Ensure 'missing' is in the list
                if 'missing' not in all_vals:
                    all_vals.append('missing')
                    all_vals = sorted(all_vals)
                
                # Convert both to categorical with same categories
                X_train[col] = X_train[col].astype(str)
                X_train[col] = X_train[col].replace(['nan', 'None', 'NaN'], 'missing')
                X_train[col] = pd.Categorical(X_train[col], categories=all_vals)
                
                X_val[col] = X_val[col].astype(str)
                X_val[col] = X_val[col].replace(['nan', 'None', 'NaN'], 'missing')
                X_val[col] = pd.Categorical(X_val[col], categories=all_vals)
            elif X_train[col].dtype == 'object':
                X_train[col] = X_train[col].astype('category')
                X_val[col] = X_val[col].astype('category')
        
        if categorical_features:
            cat_indices = [X_train.columns.get_loc(cat) for cat in categorical_features if cat in X_train.columns]
        else:
            cat_indices = 'auto'
    else:
        cat_indices = 'auto'
    
    def objective(params):
        params_dict = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': int(params['num_leaves']),
            'learning_rate': params['learning_rate'],
            'feature_fraction': params['feature_fraction'],
            'bagging_fraction': params['bagging_fraction'],
            'bagging_freq': int(params['bagging_freq']),
            'min_child_samples': int(params['min_child_samples']),
            'max_depth': int(params['max_depth']),
            'reg_alpha': params['reg_alpha'],
            'reg_lambda': params['reg_lambda'],
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'verbose': -1
        }
        
        train_data = lgb.Dataset(X_train, label=y_train,
                                categorical_feature=cat_indices if isinstance(cat_indices, list) else None,
                                free_raw_data=False)
        valid_data = lgb.Dataset(X_val, label=y_val,
                                categorical_feature=cat_indices if isinstance(cat_indices, list) else None,
                                free_raw_data=False)
        
        model = lgb.train(
            params_dict,
            train_data,
            num_boost_round=4000,  # Increased from 3000 for higher capacity
            valid_sets=[valid_data],
            valid_names=['valid'],
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(0)
            ]
        )
        
        y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
        auc = roc_auc_score(y_val, y_pred_proba)
        
        return {'loss': -auc, 'status': STATUS_OK}
    
    space = {
        'num_leaves': hp.quniform('num_leaves', 127, 511, 1),  # Increased range for higher capacity (127-511)
        'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.05)),  # Lower min for more capacity
        'feature_fraction': hp.uniform('feature_fraction', 0.6, 1.0),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.6, 1.0),
        'bagging_freq': hp.quniform('bagging_freq', 1, 7, 1),
        'min_child_samples': hp.quniform('min_child_samples', 5, 100, 1),
        'max_depth': hp.quniform('max_depth', 12, 20, 1),  # Increased range for higher capacity (12-20)
        'reg_alpha': hp.loguniform('reg_alpha', np.log(0.01), np.log(10.0)),
        'reg_lambda': hp.loguniform('reg_lambda', np.log(0.01), np.log(10.0))
    }
    
    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    
    best_params = {
        'num_leaves': int(best['num_leaves']),
        'learning_rate': best['learning_rate'],
        'feature_fraction': best['feature_fraction'],
        'bagging_fraction': best['bagging_fraction'],
        'bagging_freq': int(best['bagging_freq']),
        'min_child_samples': int(best['min_child_samples']),
        'max_depth': int(best['max_depth']),
        'reg_alpha': best['reg_alpha'],
        'reg_lambda': best['reg_lambda'],
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt'
    }
    
    return best_params


# Try to import XGBoost and CatBoost for ensemble
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available. Install with: pip install catboost")


class EnsembleModel:
    """
    Ensemble model combining LightGBM, XGBoost, and CatBoost.
    Uses weighted average of predictions based on CV scores.
    Optimized for sixth view with balanced capacity and speed.
    """
    
    def __init__(self,
                 lgb_model: Optional['SeventhViewLightGBM'] = None,
                 xgb_model: Optional[object] = None,
                 cat_model: Optional[object] = None,
                 weights: Optional[List[float]] = None,
                 threshold: float = 0.65):
        """
        Initialize ensemble model.
        
        Parameters
        ----------
        lgb_model : SeventhViewLightGBM, optional
            Trained LightGBM model
        xgb_model : object, optional
            Trained XGBoost model
        cat_model : object, optional
            Trained CatBoost model
        weights : list, optional
            Weights for each model [lgb_weight, xgb_weight, cat_weight]
            If None, equal weights will be used
        threshold : float
            Classification threshold
        """
        self.lgb_model = lgb_model
        self.xgb_model = xgb_model
        self.cat_model = cat_model
        self.threshold = threshold
        
        # Set weights
        if weights is None:
            n_models = sum([lgb_model is not None, xgb_model is not None, cat_model is not None])
            if n_models > 0:
                self.weights = [1.0 / n_models] * n_models
            else:
                self.weights = [1.0]
        else:
            self.weights = weights
        
        # Normalize weights
        total_weight = sum(self.weights)
        if total_weight > 0:
            self.weights = [w / total_weight for w in self.weights]
    
    def predict_proba(self, X):
        """Predict probabilities using ensemble."""
        predictions = []
        
        if self.lgb_model is not None:
            pred = self.lgb_model.predict_proba(X)
            predictions.append(pred)
        
        if self.xgb_model is not None and XGBOOST_AVAILABLE:
            # Convert categorical features for XGBoost prediction
            X_xgb = X.copy()
            
            # First, convert all categorical/object columns to numeric using label encoders
            if hasattr(self.xgb_model, 'label_encoders'):
                for col, le in self.xgb_model.label_encoders.items():
                    if col in X_xgb.columns:
                        # Handle unseen categories - convert to string first
                        X_xgb[col] = X_xgb[col].astype(str)
                        # Map unseen categories to the most common category
                        known_categories = set(le.classes_)
                        X_xgb[col] = X_xgb[col].apply(lambda x: x if x in known_categories else le.classes_[0])
                        X_xgb[col] = le.transform(X_xgb[col]).astype(int)
            
            # Second pass: Convert ALL remaining categorical/object columns to numeric
            # This handles columns that might not have been in label_encoders
            for col in X_xgb.columns:
                # Check if column is categorical or object type
                dtype_name = str(X_xgb[col].dtype)
                is_categorical = (
                    X_xgb[col].dtype == 'object' or 
                    dtype_name == 'category' or
                    'category' in dtype_name.lower() or
                    X_xgb[col].dtype.name == 'category' or
                    not pd.api.types.is_numeric_dtype(X_xgb[col])
                )
                
                if is_categorical:
                    # Handle categorical dtype specifically
                    if X_xgb[col].dtype.name == 'category':
                        # Convert categorical to codes first
                        X_xgb[col] = X_xgb[col].cat.codes.astype(int)
                    else:
                        # Convert to string first, then to numeric
                        X_xgb[col] = X_xgb[col].astype(str)
                        # Replace NaN strings
                        X_xgb[col] = X_xgb[col].replace(['nan', 'None', 'NaN', '<NA>', 'NaT'], '0')
                        # Convert to numeric
                        X_xgb[col] = pd.to_numeric(X_xgb[col], errors='coerce').fillna(0).astype(int)
            
            # Final verification: ensure ALL columns are numeric (int or float)
            # Check again for any remaining categorical/object columns
            for col in X_xgb.columns:
                dtype_name = str(X_xgb[col].dtype)
                if (X_xgb[col].dtype.name == 'category' or
                    dtype_name == 'category' or
                    'category' in dtype_name.lower()):
                    # Convert categorical to codes
                    X_xgb[col] = X_xgb[col].cat.codes.astype(int)
                elif not pd.api.types.is_numeric_dtype(X_xgb[col]):
                    # Force conversion for any remaining non-numeric
                    X_xgb[col] = X_xgb[col].astype(str)
                    X_xgb[col] = X_xgb[col].replace(['nan', 'None', 'NaN', '<NA>', 'NaT'], '0')
                    X_xgb[col] = pd.to_numeric(X_xgb[col], errors='coerce').fillna(0).astype(int)
            
            xgb_data = xgb.DMatrix(X_xgb)
            pred = self.xgb_model.predict(xgb_data)
            predictions.append(pred)
        
        if self.cat_model is not None and CATBOOST_AVAILABLE:
            pred = self.cat_model.predict_proba(X)[:, 1]
            predictions.append(pred)
        
        if len(predictions) == 0:
            raise ValueError("No models available for prediction")
        
        # Weighted average
        ensemble_pred = np.zeros(len(X))
        for i, pred in enumerate(predictions):
            ensemble_pred += self.weights[i] * pred
        
        return ensemble_pred
    
    def predict(self, X, threshold: Optional[float] = None):
        """Predict class labels."""
        if threshold is None:
            threshold = self.threshold
        return (self.predict_proba(X) > threshold).astype(int)


def train_seventh_view_model(X_train, y_train,
                           X_val=None, y_val=None,
                           model_params: Optional[Dict] = None,
                           categorical_features: Optional[List[str]] = None,
                           threshold: float = 0.65,
                           use_hyperparameter_tuning: bool = False,
                           tuning_method: str = 'optuna',
                           n_trials: int = 50,
                           use_cv_for_tuning: bool = False,
                           cv_folds: int = 3) -> SeventhViewLightGBM:  # Optimized: reduced from 5 to 3
    """
    Train seventh view model with optional hyperparameter tuning.
    Optimized for AUC-ROC > 0.95, Recall > 0.9, and Precision > 0.5.
    
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
        Model parameters (will be optimized if use_hyperparameter_tuning=True)
    categorical_features : list, optional
        List of categorical feature names
    threshold : float
        Classification threshold
    use_hyperparameter_tuning : bool
        Whether to use hyperparameter tuning
    tuning_method : str
        'optuna' or 'hyperopt'
    n_trials : int
        Number of trials for hyperparameter tuning
        
    Returns
    -------
    model : SeventhViewLightGBM
        Trained model
    """
    if model_params is None:
        model_params = {}
    
    # Hyperparameter tuning - ACTIVE NOW
    if use_hyperparameter_tuning:
        print("Optimizing hyperparameters using {}...".format(tuning_method))
        if tuning_method == 'optuna':
            best_params = optimize_hyperparameters_optuna(
                X_train, y_train,
                X_val=X_val, y_val=y_val,
                categorical_features=categorical_features,
                n_trials=n_trials,
                use_cv=use_cv_for_tuning,  # Use CV if explicitly requested
                cv_folds=cv_folds
            )
        elif tuning_method == 'hyperopt':
            best_params = optimize_hyperparameters_hyperopt(
                X_train, y_train,
                X_val, y_val,
                categorical_features=categorical_features,
                max_evals=n_trials
            )
        else:
            raise ValueError(f"Unknown tuning method: {tuning_method}")
        
        # Update model_params with optimized parameters
        model_params.update({
            'num_leaves': best_params.get('num_leaves', 127),
            'learning_rate': best_params.get('learning_rate', 0.01),
            'colsample_bytree': best_params.get('feature_fraction', 0.8),
            'subsample': best_params.get('bagging_fraction', 0.8),
            'min_child_samples': best_params.get('min_child_samples', 20),
            'max_depth': best_params.get('max_depth', 12),
            'reg_alpha': best_params.get('reg_alpha', 0.1),
            'reg_lambda': best_params.get('reg_lambda', 0.1),
            'scale_pos_weight': best_params.get('scale_pos_weight', None),
            'n_estimators': 3000  # Increased default
        })
        print(f"Best hyperparameters found: {best_params}")
    
    # Original hyperparameter tuning code (commented out for fast testing)
    # if use_hyperparameter_tuning and X_val is not None and y_val is not None:
    #     print(f"Optimizing hyperparameters using {tuning_method}...")
    #     if tuning_method == 'optuna':
    #         best_params = optimize_hyperparameters_optuna(
    #             X_train, y_train, X_val, y_val,
    #             categorical_features=categorical_features,
    #             n_trials=n_trials
    #         )
    #     elif tuning_method == 'hyperopt':
    #         best_params = optimize_hyperparameters_hyperopt(
    #             X_train, y_train, X_val, y_val,
    #             categorical_features=categorical_features,
    #             max_evals=n_trials
    #         )
    #     else:
    #         raise ValueError(f"Unknown tuning method: {tuning_method}")
    #     
    #     # Update model_params with optimized parameters (enhanced defaults for fifth view)
    #     model_params.update({
    #         'num_leaves': best_params.get('num_leaves', 127),  # Increased default
    #         'learning_rate': best_params.get('learning_rate', 0.01),
    #         'colsample_bytree': best_params.get('feature_fraction', 0.8),
    #         'subsample': best_params.get('bagging_fraction', 0.8),
    #         'min_child_samples': best_params.get('min_child_samples', 20),
    #         'max_depth': best_params.get('max_depth', 12),  # Increased default
    #         'reg_alpha': best_params.get('reg_alpha', 0.1),
    #         'reg_lambda': best_params.get('reg_lambda', 0.1),
    #         'scale_pos_weight': best_params.get('scale_pos_weight', None),
    #         'n_estimators': 3000  # Increased default
    #     })
    #     print(f"Best parameters found: {best_params}")
    
    model_params['threshold'] = threshold
    
    model = SeventhViewLightGBM(**model_params)
    
    eval_set = None
    if X_val is not None and y_val is not None:
        eval_set = (X_val, y_val)
    
    model.fit(X_train, y_train,
              categorical_features=categorical_features,
              eval_set=eval_set)
    
    return model


def train_ensemble_model(X_train, y_train,
                        X_val=None, y_val=None,
                        categorical_features: Optional[List[str]] = None,
                        use_lgb: bool = True,
                        use_xgb: bool = False,
                        use_cat: bool = False,
                        lgb_params: Optional[Dict] = None,
                        xgb_params: Optional[Dict] = None,
                        cat_params: Optional[Dict] = None,
                        weights: Optional[List[float]] = None,
                        threshold: float = 0.65,
                        use_hyperparameter_tuning: bool = False,
                        tuning_method: str = 'optuna',
                        n_trials: int = 50,
                        use_cv_for_weights: bool = True,
                        cv_folds: int = 3) -> EnsembleModel:  # Optimized: reduced from 5 to 3 for faster CV
    """
    Train ensemble model with LightGBM, XGBoost, and/or CatBoost.
    
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
    categorical_features : list, optional
        List of categorical feature names
    use_lgb : bool
        Whether to use LightGBM
    use_xgb : bool
        Whether to use XGBoost
    use_cat : bool
        Whether to use CatBoost
    lgb_params : dict, optional
        LightGBM parameters
    xgb_params : dict, optional
        XGBoost parameters
    cat_params : dict, optional
        CatBoost parameters
    weights : list, optional
        Weights for each model [lgb, xgb, cat]
    threshold : float
        Classification threshold
    use_hyperparameter_tuning : bool
        Whether to use hyperparameter tuning (only for LightGBM)
    tuning_method : str
        'optuna' or 'hyperopt'
    n_trials : int
        Number of trials for hyperparameter tuning
        
    Returns
    -------
    ensemble : EnsembleModel
        Trained ensemble model
    """
    lgb_model = None
    xgb_model = None
    cat_model = None
    
    # Train LightGBM
    if use_lgb:
        print("Training LightGBM model...")
        lgb_model = train_seventh_view_model(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            model_params=lgb_params,
            categorical_features=categorical_features,
            threshold=threshold,
            use_hyperparameter_tuning=use_hyperparameter_tuning,
            tuning_method=tuning_method,
            n_trials=n_trials,
            use_cv_for_tuning=use_cv_for_weights,  # Use CV for tuning if CV is enabled for weights
            cv_folds=cv_folds
        )
    
    # Train XGBoost
    if use_xgb and XGBOOST_AVAILABLE:
        print("Training XGBoost model...")
        if xgb_params is None:
            xgb_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 7,
                'learning_rate': 0.01,
                'n_estimators': 2000,  # Optimized for 3-4 hour training: reduced from 4000
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'scale_pos_weight': (len(y_train) - y_train.sum()) / y_train.sum() if y_train.sum() > 0 else 1.0
            }
        
        # Convert categorical features to integer for XGBoost
        # XGBoost doesn't support categorical dtype directly
        X_train_xgb = X_train.copy()
        X_val_xgb = X_val.copy() if X_val is not None else None
        
        from sklearn.preprocessing import LabelEncoder
        label_encoders = {}
        
        # Convert ALL columns to ensure they're numeric
        # First pass: identify and convert categorical/object columns
        for col in X_train_xgb.columns:
            # Check if column needs conversion
            needs_conversion = False
            dtype_name = str(X_train_xgb[col].dtype)
            
            if (X_train_xgb[col].dtype == 'object' or 
                dtype_name == 'category' or
                'category' in dtype_name.lower() or
                X_train_xgb[col].dtype.name == 'category' or
                not pd.api.types.is_numeric_dtype(X_train_xgb[col])):
                needs_conversion = True
            
            if needs_conversion:
                le = LabelEncoder()
                # Convert to string first to handle all cases (including NaN)
                train_col_str = X_train_xgb[col].astype(str)
                
                # Fit on combined train and validation data
                if X_val_xgb is not None:
                    val_col_str = X_val_xgb[col].astype(str)
                    combined = pd.concat([train_col_str, val_col_str], axis=0)
                else:
                    combined = train_col_str
                
                le.fit(combined)
                label_encoders[col] = le
                
                # Transform train - ensure it's integer type
                X_train_xgb[col] = le.transform(train_col_str).astype(int)
                
                # Transform validation if exists - ensure it's integer type
                if X_val_xgb is not None:
                    X_val_xgb[col] = le.transform(val_col_str).astype(int)
        
        # Second pass: Force convert any remaining non-numeric columns
        for col in X_train_xgb.columns:
            if not pd.api.types.is_numeric_dtype(X_train_xgb[col]):
                # Force conversion to numeric
                X_train_xgb[col] = pd.to_numeric(X_train_xgb[col], errors='coerce').fillna(0).astype(int)
            
            if X_val_xgb is not None:
                if not pd.api.types.is_numeric_dtype(X_val_xgb[col]):
                    X_val_xgb[col] = pd.to_numeric(X_val_xgb[col], errors='coerce').fillna(0).astype(int)
        
        # Final verification: ensure all dtypes are numeric
        # Convert any remaining non-numeric columns to int64
        for col in X_train_xgb.columns:
            if not pd.api.types.is_numeric_dtype(X_train_xgb[col]):
                X_train_xgb[col] = X_train_xgb[col].astype('int64')
        if X_val_xgb is not None:
            for col in X_val_xgb.columns:
                if not pd.api.types.is_numeric_dtype(X_val_xgb[col]):
                    X_val_xgb[col] = X_val_xgb[col].astype('int64')
        
        xgb_train = xgb.DMatrix(X_train_xgb, label=y_train)
        if X_val_xgb is not None and y_val is not None:
            xgb_val = xgb.DMatrix(X_val_xgb, label=y_val)
            xgb_model = xgb.train(
                xgb_params,
                xgb_train,
                num_boost_round=xgb_params.get('n_estimators', 2000),  # Optimized for 3-4 hour training
                evals=[(xgb_train, 'train'), (xgb_val, 'val')],
                early_stopping_rounds=100,
                verbose_eval=False
            )
        else:
            xgb_model = xgb.train(
                xgb_params,
                xgb_train,
                num_boost_round=xgb_params.get('n_estimators', 2000),  # Optimized for 3-4 hour training
                verbose_eval=False
            )
        
        # Store label encoders for prediction
        xgb_model.label_encoders = label_encoders
        xgb_model.feature_names = X_train_xgb.columns.tolist()
    
    # Train CatBoost
    if use_cat and CATBOOST_AVAILABLE:
        print("Training CatBoost model...")
        if cat_params is None:
            cat_params = {
                'iterations': 2000,  # Optimized for 3-4 hour training: reduced from 2500
                'learning_rate': 0.01,
                'depth': 7,
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'random_seed': 42,
                'scale_pos_weight': (len(y_train) - y_train.sum()) / y_train.sum() if y_train.sum() > 0 else 1.0,
                'verbose': False
            }
        
        # Prepare data for CatBoost - ensure categorical features are integer or string
        X_train_cat = X_train.copy()
        X_val_cat = X_val.copy() if X_val is not None else None
        
        # Convert categorical features to string for CatBoost
        # CatBoost requires categorical features to be integer or string (not float)
        if categorical_features:
            for col in categorical_features:
                if col in X_train_cat.columns:
                    # Convert to string - handle all cases (float, int, category, object, NaN)
                    # First convert to string, replacing NaN values
                    X_train_cat[col] = X_train_cat[col].astype(str)
                    X_train_cat[col] = X_train_cat[col].replace(['nan', 'None', 'NaN', '<NA>', 'NaT'], 'missing')
                    
                    # If values are numeric (float/int), convert to integer string
                    try:
                        # Try to convert to numeric first
                        numeric_vals = pd.to_numeric(X_train_cat[col].replace('missing', '0'), errors='coerce')
                        if numeric_vals.notna().all():
                            # All values are numeric, convert to integer string
                            X_train_cat[col] = numeric_vals.fillna(0).astype(int).astype(str)
                        else:
                            # Some values are not numeric, keep as string
                            pass
                    except:
                        # Keep as string if conversion fails
                        pass
                    
                    if X_val_cat is not None and col in X_val_cat.columns:
                        X_val_cat[col] = X_val_cat[col].astype(str)
                        X_val_cat[col] = X_val_cat[col].replace(['nan', 'None', 'NaN', '<NA>', 'NaT'], 'missing')
                        try:
                            numeric_vals = pd.to_numeric(X_val_cat[col].replace('missing', '0'), errors='coerce')
                            if numeric_vals.notna().all():
                                X_val_cat[col] = numeric_vals.fillna(0).astype(int).astype(str)
                        except:
                            pass
            
            cat_features = [X_train_cat.columns.get_loc(cat) for cat in categorical_features if cat in X_train_cat.columns]
        else:
            cat_features = None
        
        cat_train = cb.Pool(X_train_cat, y_train, cat_features=cat_features)
        if X_val_cat is not None and y_val is not None:
            cat_val = cb.Pool(X_val_cat, y_val, cat_features=cat_features)
            cat_model = cb.CatBoostClassifier(**cat_params)
            cat_model.fit(cat_train, eval_set=cat_val, early_stopping_rounds=100, verbose=False)
        else:
            cat_model = cb.CatBoostClassifier(**cat_params)
            cat_model.fit(cat_train, verbose=False)
        
        # Store the processed data info for prediction
        cat_model.cat_features_processed = True
        cat_model.categorical_features = categorical_features
    
    # Compute ensemble weights based on CV scores if requested
    if weights is None and use_cv_for_weights:
        print("\nComputing ensemble weights based on CV scores...")
        # Combine train and validation for CV
        X_cv = pd.concat([X_train, X_val], axis=0).reset_index(drop=True) if X_val is not None else X_train
        y_cv = np.concatenate([y_train, y_val], axis=0) if y_val is not None else y_train
        
        models_list = []
        model_names = []
        
        if lgb_model is not None:
            # Create a fresh model for CV
            from src.seventh_view.models import SeventhViewLightGBM
            cv_lgb = SeventhViewLightGBM(
                n_estimators=lgb_model.n_estimators,
                learning_rate=lgb_model.learning_rate,
                max_depth=lgb_model.max_depth,
                num_leaves=lgb_model.num_leaves,
                min_child_samples=lgb_model.min_child_samples,
                subsample=lgb_model.subsample,
                colsample_bytree=lgb_model.colsample_bytree,
                reg_alpha=lgb_model.reg_alpha,
                reg_lambda=lgb_model.reg_lambda,
                scale_pos_weight=lgb_model.scale_pos_weight,
                random_state=lgb_model.random_state,
                threshold=threshold
            )
            models_list.append(cv_lgb)
            model_names.append('LightGBM')
        
        if xgb_model is not None and XGBOOST_AVAILABLE:
            # For XGBoost, we'll use a wrapper that handles categorical features
            # Create a simple wrapper for CV
            class XGBoostCVWrapper:
                def __init__(self, xgb_params, label_encoders, feature_names):
                    self.xgb_params = xgb_params
                    self.label_encoders = label_encoders
                    self.feature_names = feature_names
                    self.model = None
                
                def fit(self, X, y, categorical_features=None):
                    # Convert categorical features
                    X_processed = X.copy()
                    from sklearn.preprocessing import LabelEncoder
                    
                    for col in X_processed.columns:
                        if col in self.label_encoders:
                            le = self.label_encoders[col]
                            X_processed[col] = X_processed[col].astype(str)
                            known_categories = set(le.classes_)
                            X_processed[col] = X_processed[col].apply(
                                lambda x: x if x in known_categories else le.classes_[0]
                            )
                            X_processed[col] = le.transform(X_processed[col]).astype(int)
                        elif not pd.api.types.is_numeric_dtype(X_processed[col]):
                            X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce').fillna(0).astype(int)
                    
                    xgb_data = xgb.DMatrix(X_processed, label=y)
                    self.model = xgb.train(
                        self.xgb_params,
                        xgb_data,
                        num_boost_round=self.xgb_params.get('n_estimators', 2500),  # Optimized: reduced for faster training
                        verbose_eval=False
                    )
                
                def predict_proba(self, X):
                    X_processed = X.copy()
                    for col in X_processed.columns:
                        if col in self.label_encoders:
                            le = self.label_encoders[col]
                            X_processed[col] = X_processed[col].astype(str)
                            known_categories = set(le.classes_)
                            X_processed[col] = X_processed[col].apply(
                                lambda x: x if x in known_categories else le.classes_[0]
                            )
                            X_processed[col] = le.transform(X_processed[col]).astype(int)
                        elif not pd.api.types.is_numeric_dtype(X_processed[col]):
                            X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce').fillna(0).astype(int)
                    
                    xgb_data = xgb.DMatrix(X_processed)
                    pred = self.model.predict(xgb_data)
                    return pred
            
            cv_xgb = XGBoostCVWrapper(
                xgb_params if xgb_params else {},
                xgb_model.label_encoders if hasattr(xgb_model, 'label_encoders') else {},
                xgb_model.feature_names if hasattr(xgb_model, 'feature_names') else X_train.columns.tolist()
            )
            models_list.append(cv_xgb)
            model_names.append('XGBoost')
        
        if cat_model is not None and CATBOOST_AVAILABLE:
            # For CatBoost, create a wrapper
            class CatBoostCVWrapper:
                def __init__(self, cat_params, categorical_features):
                    self.cat_params = cat_params
                    self.categorical_features = categorical_features
                    self.model = None
                
                def fit(self, X, y, categorical_features=None):
                    X_processed = X.copy()
                    if self.categorical_features:
                        for col in self.categorical_features:
                            if col in X_processed.columns:
                                X_processed[col] = X_processed[col].astype(str)
                                X_processed[col] = X_processed[col].replace(['nan', 'None', 'NaN', '<NA>'], 'missing')
                    
                    cat_features = [X_processed.columns.get_loc(cat) for cat in self.categorical_features 
                                   if cat in X_processed.columns] if self.categorical_features else None
                    cat_train = cb.Pool(X_processed, y, cat_features=cat_features)
                    self.model = cb.CatBoostClassifier(**self.cat_params)
                    self.model.fit(cat_train, verbose=False)
                
                def predict_proba(self, X):
                    X_processed = X.copy()
                    if self.categorical_features:
                        for col in self.categorical_features:
                            if col in X_processed.columns:
                                X_processed[col] = X_processed[col].astype(str)
                                X_processed[col] = X_processed[col].replace(['nan', 'None', 'NaN', '<NA>'], 'missing')
                    
                    cat_features = [X_processed.columns.get_loc(cat) for cat in self.categorical_features 
                                   if cat in X_processed.columns] if self.categorical_features else None
                    cat_data = cb.Pool(X_processed, cat_features=cat_features)
                    return self.model.predict_proba(cat_data)[:, 1]
            
            cv_cat = CatBoostCVWrapper(
                cat_params if cat_params else {},
                categorical_features if categorical_features else []
            )
            models_list.append(cv_cat)
            model_names.append('CatBoost')
        
        if len(models_list) > 0:
            from src.seventh_view.evaluation import compute_ensemble_weights_from_cv
            weights = compute_ensemble_weights_from_cv(
                models_list, model_names, X_cv, y_cv,
                cv=cv_folds, scoring='roc_auc', verbose=True
            )
        else:
            # Fallback to equal weights
            n_models = sum([lgb_model is not None, xgb_model is not None, cat_model is not None])
            weights = [1.0 / n_models] * n_models if n_models > 0 else [1.0]
    
    # Create ensemble
    ensemble = EnsembleModel(
        lgb_model=lgb_model,
        xgb_model=xgb_model,
        cat_model=cat_model,
        weights=weights,
        threshold=threshold
    )
    
    return ensemble

