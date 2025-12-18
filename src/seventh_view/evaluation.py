"""
Evaluation utilities for seventh view model.
Extends sixth view with improved threshold optimization for better recall.
Based on sixth view improvements from MODEL_COMPARISON2.md.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
    f1_score, precision_score, recall_score
)
from sklearn.model_selection import StratifiedKFold
from typing import Tuple, Optional, List, Callable
import warnings
warnings.filterwarnings('ignore')


class TimeBasedStratifiedKFold:
    """
    Time-based stratified K-Fold cross-validator.
    Splits data based on TransactionDT to prevent leakage while maintaining class distribution.
    Based on FraudSquad approach.
    
    Parameters
    ----------
    n_splits : int
        Number of folds
    time_col : str
        Time column name (default: 'TransactionDT')
    """
    def __init__(self, n_splits: int = 5, time_col: str = 'TransactionDT'):
        self.n_splits = n_splits
        self.time_col = time_col
    
    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.
        
        Parameters
        ----------
        X : array-like or DataFrame
            Features
        y : array-like, optional
            Target variable
        groups : array-like, optional
            Group labels
            
        Yields
        ------
        train_idx : ndarray
            Training set indices
        val_idx : ndarray
            Validation set indices
        """
        if isinstance(X, pd.DataFrame):
            if self.time_col not in X.columns:
                # Fallback to StratifiedKFold if time column not found
                skf = StratifiedKFold(n_splits=self.n_splits, shuffle=False, random_state=42)
                return skf.split(X, y)
            time_values = X[self.time_col].values
        else:
            # If X is not DataFrame, assume it's numpy array and time_col is index
            time_values = np.arange(len(X))
        
        # Sort by time
        sorted_indices = np.argsort(time_values)
        
        # Calculate fold sizes
        n_samples = len(sorted_indices)
        fold_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            # Validation set: current fold
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            
            val_idx = sorted_indices[val_start:val_end]
            train_idx = np.concatenate([sorted_indices[:val_start], sorted_indices[val_end:]])
            
            yield train_idx, val_idx
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator."""
        return self.n_splits


def evaluate_model(y_true: np.ndarray, 
                   y_pred: np.ndarray,
                   y_pred_proba: Optional[np.ndarray] = None,
                   threshold: float = 0.65,
                   verbose: bool = True) -> dict:
    """Evaluate model performance with multiple metrics."""
    metrics = {}
    metrics['threshold'] = threshold
    
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['recall'] = recall_score(y_true, y_pred)
    metrics['f1'] = f1_score(y_true, y_pred)
    
    if y_pred_proba is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['avg_precision'] = average_precision_score(y_true, y_pred_proba)
    
    cm = confusion_matrix(y_true, y_pred)
    metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = cm.ravel()
    
    if verbose:
        print("=" * 50)
        print("Model Evaluation Metrics")
        print("=" * 50)
        print(f"Threshold:       {threshold:.2f}")
        print(f"AUC-ROC:         {metrics.get('auc_roc', 'N/A'):.4f}")
        print(f"Average Precision: {metrics.get('avg_precision', 'N/A'):.4f}")
        print(f"Precision:       {metrics['precision']:.4f}")
        print(f"Recall:          {metrics['recall']:.4f}")
        print(f"F1-Score:        {metrics['f1']:.4f}")
        print("\nConfusion Matrix:")
        print(f"True Negatives:  {metrics['tn']}")
        print(f"False Positives: {metrics['fp']}")
        print(f"False Negatives: {metrics['fn']}")
        print(f"True Positives:  {metrics['tp']}")
        print("=" * 50)
    
    return metrics


def evaluate_multiple_thresholds(y_true: np.ndarray,
                                  y_pred_proba: np.ndarray,
                                  thresholds: List[float] = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
                                  verbose: bool = True) -> pd.DataFrame:
    """Evaluate model performance at multiple thresholds."""
    results = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)
        
        metrics = {
            'threshold': threshold,
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'tp': ((y_true == 1) & (y_pred == 1)).sum(),
            'fp': ((y_true == 0) & (y_pred == 1)).sum(),
            'fn': ((y_true == 1) & (y_pred == 0)).sum(),
            'tn': ((y_true == 0) & (y_pred == 0)).sum()
        }
        results.append(metrics)
    
    results_df = pd.DataFrame(results)
    
    if verbose:
        print("=" * 70)
        print("Threshold Comparison")
        print("=" * 70)
        print(results_df.to_string(index=False))
        print("=" * 70)
    
    return results_df


def find_optimal_threshold(y_true: np.ndarray,
                          y_pred_proba: np.ndarray,
                          metric: str = 'f1',
                          threshold_range: Tuple[float, float] = (0.5, 0.8),
                          n_thresholds: int = 50) -> float:
    """Find optimal threshold based on specified metric."""
    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)
    best_threshold = threshold_range[0]
    best_score = -1
    
    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold


def find_threshold_multi_objective_optimized(y_true: np.ndarray,
                                             y_pred_proba: np.ndarray,
                                             min_auc_roc: float = 0.95,
                                             min_recall: float = 0.9,
                                             min_precision: float = 0.5,
                                             threshold_range: Tuple[float, float] = (0.01, 0.99),
                                             n_thresholds: int = 200,
                                             verbose: bool = True) -> Tuple[float, dict]:
    """
    Find threshold that achieves:
    - AUC-ROC >= min_auc_roc
    - Recall >= min_recall
    - Precision >= min_precision
    While maximizing all three metrics.
    
    This is optimized for sixth view realistic targets: 
    - AUC-ROC >= 0.94 (target: 0.94-0.945)
    - Recall >= 0.85 (target: 0.85-0.90)
    - Precision >= 0.40 (target: 0.45-0.55)
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred_proba : np.ndarray
        Predicted probabilities
    min_recall : float
        Minimum required recall
    threshold_range : tuple
        Range of thresholds to search
    n_thresholds : int
        Number of thresholds to try
    verbose : bool
        Whether to print results
        
    Returns
    -------
    best_threshold : float
        Best threshold found
    best_metrics : dict
        Metrics at best threshold
    """
    # Calculate AUC-ROC first (threshold-independent)
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    
    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)
    
    valid_thresholds = []
    best_threshold = threshold_range[0]
    best_metrics = None
    best_composite_score = -1
    
    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        
        # Check if all constraints are met
        meets_auc = auc_roc >= min_auc_roc
        meets_recall = recall >= min_recall
        meets_precision = precision >= min_precision
        
        if meets_auc and meets_recall and meets_precision:
            # Composite score: weighted combination of all metrics
            # Prioritize precision (business impact) while maintaining recall and AUC
            composite_score = (0.4 * precision + 0.3 * recall + 0.3 * auc_roc)
            
            valid_thresholds.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'auc_roc': auc_roc,
                'f1': f1_score(y_true, y_pred),
                'composite_score': composite_score
            })
            
            if composite_score > best_composite_score:
                best_composite_score = composite_score
                best_threshold = threshold
                best_metrics = {
                    'threshold': threshold,
                    'precision': precision,
                    'recall': recall,
                    'auc_roc': auc_roc,
                    'f1': f1_score(y_true, y_pred),
                    'tp': ((y_true == 1) & (y_pred == 1)).sum(),
                    'fp': ((y_true == 0) & (y_pred == 1)).sum(),
                    'fn': ((y_true == 1) & (y_pred == 0)).sum(),
                    'tn': ((y_true == 0) & (y_pred == 0)).sum(),
                    'composite_score': composite_score
                }
    
    # If no threshold meets all requirements, find best compromise
    if best_metrics is None:
        best_composite_score = -1
        for threshold in thresholds:
            y_pred = (y_pred_proba > threshold).astype(int)
            recall = recall_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            
            # Penalty for not meeting requirements
            penalty = 0
            if recall < min_recall:
                penalty += (min_recall - recall) * 2  # Heavy penalty for low recall
            if precision < min_precision:
                penalty += (min_precision - precision) * 1.5  # Medium penalty for low precision
            if auc_roc < min_auc_roc:
                penalty += (min_auc_roc - auc_roc) * 1.0  # Light penalty for low AUC (already calculated)
            
            composite_score = (0.4 * precision + 0.3 * recall + 0.3 * auc_roc) - penalty
            
            if composite_score > best_composite_score:
                best_composite_score = composite_score
                best_threshold = threshold
                best_metrics = {
                    'threshold': threshold,
                    'precision': precision,
                    'recall': recall,
                    'auc_roc': auc_roc,
                    'f1': f1_score(y_true, y_pred),
                    'tp': ((y_true == 1) & (y_pred == 1)).sum(),
                    'fp': ((y_true == 0) & (y_pred == 1)).sum(),
                    'fn': ((y_true == 1) & (y_pred == 0)).sum(),
                    'tn': ((y_true == 0) & (y_pred == 0)).sum(),
                    'composite_score': composite_score
                }
    
    if verbose:
        print("=" * 70)
        print(f"Multi-Objective Threshold Optimization")
        print(f"Targets: AUC-ROC >= {min_auc_roc}, Recall >= {min_recall}, Precision >= {min_precision}")
        print("=" * 70)
        if best_metrics:
            print(f"Best Threshold:  {best_metrics['threshold']:.4f}")
            print(f"AUC-ROC:         {best_metrics['auc_roc']:.4f} {'✅' if best_metrics['auc_roc'] >= min_auc_roc else '❌'}")
            print(f"Precision:       {best_metrics['precision']:.4f} {'✅' if best_metrics['precision'] >= min_precision else '❌'}")
            print(f"Recall:          {best_metrics['recall']:.4f} {'✅' if best_metrics['recall'] >= min_recall else '❌'}")
            print(f"F1-Score:        {best_metrics['f1']:.4f}")
            print(f"Composite Score: {best_metrics.get('composite_score', 0):.4f}")
            print(f"\nConfusion Matrix:")
            print(f"True Positives:  {best_metrics['tp']}")
            print(f"False Positives: {best_metrics['fp']}")
            print(f"False Negatives: {best_metrics['fn']}")
            print(f"True Negatives:  {best_metrics['tn']}")
            print(f"\nValid thresholds found: {len(valid_thresholds)}")
            if len(valid_thresholds) > 0:
                print(f"Precision range: {min(v['precision'] for v in valid_thresholds):.4f} - {max(v['precision'] for v in valid_thresholds):.4f}")
                print(f"Recall range: {min(v['recall'] for v in valid_thresholds):.4f} - {max(v['recall'] for v in valid_thresholds):.4f}")
        print("=" * 70)
    
    return best_threshold, best_metrics


def compute_ensemble_weights_from_cv(models: List[Callable],
                                     model_names: List[str],
                                     X: pd.DataFrame,
                                     y: np.ndarray,
                                     cv: int = 5,
                                     scoring: str = 'roc_auc',
                                     verbose: bool = True) -> List[float]:
    """
    Compute ensemble weights based on cross-validation scores.
    
    Parameters
    ----------
    models : list
        List of model instances (must have fit and predict_proba methods)
    model_names : list
        List of model names (for logging)
    X : pd.DataFrame
        Feature dataframe
    y : np.ndarray
        Target array
    cv : int
        Number of cross-validation folds
    scoring : str
        Scoring metric ('roc_auc', 'average_precision', 'f1', 'precision', 'recall')
    verbose : bool
        Whether to print results
        
    Returns
    -------
    weights : list
        Normalized weights for each model (sum to 1.0)
    """
    if len(models) != len(model_names):
        raise ValueError("models and model_names must have same length")
    
    cv_scores = []
    
    for model, name in zip(models, model_names):
        if verbose:
            print(f"\nComputing CV score for {name}...")
        
        cv_results = evaluate_with_cross_validation(
            model, X, y, cv=cv, scoring=scoring, verbose=False
        )
        cv_scores.append(cv_results['mean'])
        
        if verbose:
            print(f"{name} CV {scoring}: {cv_results['mean']:.4f} (+/- {cv_results['std']:.4f})")
    
    # Compute weights: higher CV score = higher weight
    # Use softmax-like normalization: weights proportional to exp(score)
    # This gives more weight to better models
    import math
    exp_scores = [math.exp(score * 10) for score in cv_scores]  # Scale by 10 for better separation
    total_exp = sum(exp_scores)
    weights = [exp_score / total_exp for exp_score in exp_scores]
    
    if verbose:
        print("\n" + "=" * 70)
        print("Ensemble Weights (based on CV scores)")
        print("=" * 70)
        for name, score, weight in zip(model_names, cv_scores, weights):
            print(f"{name}: CV={score:.4f}, Weight={weight:.4f} ({weight*100:.2f}%)")
        print("=" * 70)
    
    return weights


def evaluate_with_cross_validation(model: Callable,
                                   X: pd.DataFrame,
                                   y: np.ndarray,
                                   cv: int = 5,
                                   scoring: str = 'roc_auc',
                                   verbose: bool = True) -> dict:
    """
    Evaluate model using cross-validation.
    
    Parameters
    ----------
    model : callable
        Model with fit and predict_proba methods
    X : pd.DataFrame
        Feature dataframe
    y : np.ndarray
        Target array
    cv : int
        Number of cross-validation folds
    scoring : str
        Scoring metric ('roc_auc', 'average_precision', 'f1', 'precision', 'recall')
    verbose : bool
        Whether to print results
        
    Returns
    -------
    results : dict
        Dictionary with CV results
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Map scoring string to metric function
    metric_map = {
        'roc_auc': roc_auc_score,
        'average_precision': average_precision_score,
        'f1': f1_score,
        'precision': precision_score,
        'recall': recall_score
    }
    
    if scoring not in metric_map:
        raise ValueError(f"Unknown scoring metric: {scoring}")
    
    cv_scores = []
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]
        
        # Identify categorical features for this fold
        categorical_features_fold = []
        for col in X_train_fold.columns:
            if X_train_fold[col].dtype == 'object' or X_train_fold[col].dtype.name == 'category':
                categorical_features_fold.append(col)
            elif X_train_fold[col].dtype in ['int8', 'int16', 'int32', 'int64']:
                if X_train_fold[col].nunique() < 50:
                    categorical_features_fold.append(col)
        
        # Train model with categorical features
        if hasattr(model, 'fit'):
            if hasattr(model, 'categorical_features') or 'categorical_features' in model.fit.__code__.co_varnames:
                # Model supports categorical_features parameter
                model.fit(X_train_fold, y_train_fold, categorical_features=categorical_features_fold)
            else:
                model.fit(X_train_fold, y_train_fold)
        else:
            raise ValueError("Model must have a fit method")
        
        # Predict
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_val_fold)
            if y_pred_proba.ndim > 1:
                y_pred_proba = y_pred_proba[:, 1]
        else:
            y_pred_proba = model.predict(X_val_fold)
        
        # Calculate metric
        if scoring in ['roc_auc', 'average_precision']:
            score = metric_map[scoring](y_val_fold, y_pred_proba)
        else:
            # For classification metrics, need predictions
            y_pred = (y_pred_proba > 0.5).astype(int)
            score = metric_map[scoring](y_val_fold, y_pred)
        
        cv_scores.append(score)
        fold_results.append({
            'fold': fold + 1,
            'score': score
        })
    
    results = {
        'mean': np.mean(cv_scores),
        'std': np.std(cv_scores),
        'scores': cv_scores,
        'fold_results': fold_results
    }
    
    if verbose:
        print("=" * 70)
        print(f"Cross-Validation Results ({cv}-fold, {scoring})")
        print("=" * 70)
        for fold_result in fold_results:
            print(f"Fold {fold_result['fold']}: {fold_result['score']:.4f}")
        print("-" * 70)
        print(f"Mean: {results['mean']:.4f} (+/- {results['std']:.4f})")
        print("=" * 70)
    
    return results


def plot_feature_importance(model, top_n: int = 20, save_path: Optional[str] = None):
    """Plot feature importance."""
    if model.feature_importance_ is None:
        print("Feature importance not available")
        return
    
    top_features = model.feature_importance_.head(top_n)
    
    plt.figure(figsize=(10, 8))
    top_features.plot(kind='barh')
    plt.xlabel('Importance (Gain)')
    plt.title(f'Top {top_n} Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_roc_curve(y_true: np.ndarray, 
                   y_pred_proba: np.ndarray,
                   title: str = "ROC Curve",
                   save_path: Optional[str] = None):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_precision_recall_curve(y_true: np.ndarray,
                               y_pred_proba: np.ndarray,
                               title: str = "Precision-Recall Curve",
                               save_path: Optional[str] = None):
    """Plot Precision-Recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
