"""
Model evaluation utilities for IEEE Fraud Detection project.
Includes metrics, cross-validation, and visualization functions.
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
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def evaluate_model(y_true: np.ndarray, 
                   y_pred: np.ndarray,
                   y_pred_proba: Optional[np.ndarray] = None,
                   verbose: bool = True) -> dict:
    """
    Evaluate model performance with multiple metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_pred_proba : np.ndarray, optional
        Predicted probabilities
    verbose : bool
        Whether to print results
        
    Returns
    -------
    metrics : dict
        Dictionary of evaluation metrics
    """
    metrics = {}
    
    # Classification metrics
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['recall'] = recall_score(y_true, y_pred)
    metrics['f1'] = f1_score(y_true, y_pred)
    
    # AUC-ROC
    if y_pred_proba is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['avg_precision'] = average_precision_score(y_true, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = cm.ravel()
    
    if verbose:
        print("=" * 50)
        print("Model Evaluation Metrics")
        print("=" * 50)
        print(f"AUC-ROC:        {metrics.get('auc_roc', 'N/A'):.4f}")
        print(f"Average Precision: {metrics.get('avg_precision', 'N/A'):.4f}")
        print(f"Precision:      {metrics['precision']:.4f}")
        print(f"Recall:         {metrics['recall']:.4f}")
        print(f"F1-Score:       {metrics['f1']:.4f}")
        print("\nConfusion Matrix:")
        print(f"True Negatives:  {metrics['tn']}")
        print(f"False Positives: {metrics['fp']}")
        print(f"False Negatives: {metrics['fn']}")
        print(f"True Positives:  {metrics['tp']}")
        print("=" * 50)
    
    return metrics


def plot_roc_curve(y_true: np.ndarray, 
                   y_pred_proba: np.ndarray,
                   title: str = "ROC Curve",
                   save_path: Optional[str] = None):
    """
    Plot ROC curve.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred_proba : np.ndarray
        Predicted probabilities
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
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
    """
    Plot Precision-Recall curve.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred_proba : np.ndarray
        Predicted probabilities
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
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


def plot_confusion_matrix(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         title: str = "Confusion Matrix",
                         save_path: Optional[str] = None):
    """
    Plot confusion matrix.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Fraud', 'Fraud'],
                yticklabels=['Not Fraud', 'Fraud'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def cross_validate_model(model, X, y, 
                        cv_folds: int = 5,
                        scoring: str = 'roc_auc',
                        use_time_split: bool = True,
                        time_col: Optional[str] = None) -> dict:
    """
    Perform cross-validation.
    
    Parameters
    ----------
    model : object
        Model with fit and predict_proba methods
    X : pd.DataFrame or np.ndarray
        Features
    y : np.ndarray
        Target
    cv_folds : int
        Number of CV folds
    scoring : str
        Scoring metric
    use_time_split : bool
        Whether to use time-based split
    time_col : str, optional
        Column name for time-based splitting
        
    Returns
    -------
    cv_results : dict
        Cross-validation results
    """
    if use_time_split and time_col is not None and hasattr(X, 'columns'):
        # Time-based split
        cv = TimeSeriesSplit(n_splits=cv_folds)
        time_values = X[time_col] if isinstance(X, pd.DataFrame) else None
    else:
        # Stratified K-Fold
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train_fold = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx]
        X_val_fold = X.iloc[val_idx] if isinstance(X, pd.DataFrame) else X[val_idx]
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]
        
        # Train model
        model.fit(X_train_fold, y_train_fold)
        
        # Predict
        y_pred_proba = model.predict_proba(X_val_fold)
        
        # Calculate score
        if scoring == 'roc_auc':
            score = roc_auc_score(y_val_fold, y_pred_proba)
        elif scoring == 'f1':
            y_pred = (y_pred_proba > 0.5).astype(int)
            score = f1_score(y_val_fold, y_pred)
        else:
            score = roc_auc_score(y_val_fold, y_pred_proba)
        
        scores.append(score)
        print(f"Fold {fold + 1}: {scoring} = {score:.4f}")
    
    cv_results = {
        'scores': scores,
        'mean': np.mean(scores),
        'std': np.std(scores),
        'min': np.min(scores),
        'max': np.max(scores)
    }
    
    print(f"\nCross-Validation Results:")
    print(f"Mean {scoring}: {cv_results['mean']:.4f} (+/- {cv_results['std']:.4f})")
    print(f"Min: {cv_results['min']:.4f}, Max: {cv_results['max']:.4f}")
    
    return cv_results









