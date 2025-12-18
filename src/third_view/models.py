"""LightGBM models with Optuna/Hyperopt hyperparameter tuning."""

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


class ThirdViewLightGBM:
    """
    LightGBM model optimized for third view features.
    Includes hyperparameter tuning support with Optuna/Hyperopt.
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
            early_stopping_rounds: int = 100,
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
                for col in X_val.columns:
                    if X_val[col].dtype == 'object':
                        X_val[col] = X_val[col].astype('category')
        
        train_data = lgb.Dataset(X_train, label=y_train, 
                                categorical_feature=cat_indices if isinstance(cat_indices, list) else None,
                                free_raw_data=False)
        
        valid_sets = [train_data]
        valid_names = ['train']
        if eval_set is not None:
            valid_data = lgb.Dataset(X_val, label=y_val,
                                    categorical_feature=cat_indices if isinstance(cat_indices, list) else None,
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
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = X[col].astype('category')
        
        return self.model.predict(X, num_iteration=self.model.best_iteration)


def optimize_hyperparameters_optuna(X_train, y_train,
                                     X_val, y_val,
                                     categorical_features: Optional[List[str]] = None,
                                     n_trials: int = 50,
                                     timeout: Optional[int] = None) -> Dict:
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
    
    # Convert categorical features
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.copy()
        X_val = X_val.copy()
        for col in X_train.columns:
            if X_train[col].dtype == 'object':
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
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
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
            params,
            train_data,
            num_boost_round=2000,
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
    
    study = optuna.create_study(direction='maximize', study_name='third_view_optuna')
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
    
    # Convert categorical features
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.copy()
        X_val = X_val.copy()
        for col in X_train.columns:
            if X_train[col].dtype == 'object':
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
            num_boost_round=2000,
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
        'num_leaves': hp.quniform('num_leaves', 15, 127, 1),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.05)),
        'feature_fraction': hp.uniform('feature_fraction', 0.6, 1.0),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.6, 1.0),
        'bagging_freq': hp.quniform('bagging_freq', 1, 7, 1),
        'min_child_samples': hp.quniform('min_child_samples', 5, 100, 1),
        'max_depth': hp.quniform('max_depth', 3, 12, 1),
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


def train_third_view_model(X_train, y_train,
                           X_val=None, y_val=None,
                           model_params: Optional[Dict] = None,
                           categorical_features: Optional[List[str]] = None,
                           threshold: float = 0.65,
                           use_hyperparameter_tuning: bool = False,
                           tuning_method: str = 'optuna',
                           n_trials: int = 50) -> ThirdViewLightGBM:
    """
    Train third view model with optional hyperparameter tuning.
    
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
    model : ThirdViewLightGBM
        Trained model
    """
    if model_params is None:
        model_params = {}
    
    # Hyperparameter tuning
    if use_hyperparameter_tuning and X_val is not None and y_val is not None:
        print(f"Optimizing hyperparameters using {tuning_method}...")
        if tuning_method == 'optuna':
            best_params = optimize_hyperparameters_optuna(
                X_train, y_train, X_val, y_val,
                categorical_features=categorical_features,
                n_trials=n_trials
            )
        elif tuning_method == 'hyperopt':
            best_params = optimize_hyperparameters_hyperopt(
                X_train, y_train, X_val, y_val,
                categorical_features=categorical_features,
                max_evals=n_trials
            )
        else:
            raise ValueError(f"Unknown tuning method: {tuning_method}")
        
        # Update model_params with optimized parameters
        model_params.update({
            'num_leaves': best_params.get('num_leaves', 31),
            'learning_rate': best_params.get('learning_rate', 0.01),
            'colsample_bytree': best_params.get('feature_fraction', 0.8),
            'subsample': best_params.get('bagging_fraction', 0.8),
            'min_child_samples': best_params.get('min_child_samples', 20),
            'max_depth': best_params.get('max_depth', 7),
            'reg_alpha': best_params.get('reg_alpha', 0.1),
            'reg_lambda': best_params.get('reg_lambda', 0.1),
            'scale_pos_weight': best_params.get('scale_pos_weight', None)
        })
        print(f"Best parameters found: {best_params}")
    
    model_params['threshold'] = threshold
    
    model = ThirdViewLightGBM(**model_params)
    
    eval_set = None
    if X_val is not None and y_val is not None:
        eval_set = (X_val, y_val)
    
    model.fit(X_train, y_train,
              categorical_features=categorical_features,
              eval_set=eval_set)
    
    return model

