"""
Evaluation utilities for second view model.
Includes threshold-based evaluation for better precision-recall balance.
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
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


def evaluate_model(y_true: np.ndarray, 
                   y_pred: np.ndarray,
                   y_pred_proba: Optional[np.ndarray] = None,
                   threshold: float = 0.65,
                   verbose: bool = True) -> dict:
    """
    Evaluate model performance with multiple metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels (using threshold)
    y_pred_proba : np.ndarray, optional
        Predicted probabilities
    threshold : float
        Threshold used for predictions
    verbose : bool
        Whether to print results
        
    Returns
    -------
    metrics : dict
        Dictionary of evaluation metrics
    """
    metrics = {}
    metrics['threshold'] = threshold
    
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
    """
    Evaluate model performance at multiple thresholds.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred_proba : np.ndarray
        Predicted probabilities
    thresholds : list
        List of thresholds to evaluate
    verbose : bool
        Whether to print results
        
    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with metrics for each threshold
    """
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
    """
    Find optimal threshold based on specified metric.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred_proba : np.ndarray
        Predicted probabilities
    metric : str
        Metric to optimize ('f1', 'precision', 'recall', or custom)
    threshold_range : tuple
        (min_threshold, max_threshold)
    n_thresholds : int
        Number of thresholds to test
        
    Returns
    -------
    optimal_threshold : float
        Optimal threshold value
    """
    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)
    best_score = -1
    best_threshold = 0.5
    
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

