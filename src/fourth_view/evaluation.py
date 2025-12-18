"""
Evaluation utilities for fourth view model.
Includes threshold optimization for recall >= 0.9 with maximum precision.
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


def find_threshold_recall_precision_optimized(y_true: np.ndarray,
                                             y_pred_proba: np.ndarray,
                                             min_recall: float = 0.9,
                                             threshold_range: Tuple[float, float] = (0.1, 0.9),
                                             n_thresholds: int = 200,
                                             verbose: bool = True) -> Tuple[float, dict]:
    """
    Find threshold that maximizes precision while maintaining recall >= min_recall.
    
    This function finds the threshold that:
    1. Ensures recall >= min_recall (e.g., 0.9)
    2. Among all thresholds satisfying condition 1, maximizes precision
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred_proba : np.ndarray
        Predicted probabilities
    min_recall : float
        Minimum required recall (default: 0.9)
    threshold_range : tuple
        (min_threshold, max_threshold) to search
    n_thresholds : int
        Number of thresholds to test
    verbose : bool
        Whether to print information
        
    Returns
    -------
    best_threshold : float
        Optimal threshold
    best_metrics : dict
        Metrics at optimal threshold
    """
    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)
    
    best_precision = -1
    best_threshold = None
    best_metrics = None
    valid_thresholds = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)
        
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        
        # Check if recall requirement is met
        if recall >= min_recall:
            valid_thresholds.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1_score(y_true, y_pred)
            })
            
            # Maximize precision among valid thresholds
            if precision > best_precision:
                best_precision = precision
                best_threshold = threshold
                best_metrics = {
                    'threshold': threshold,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1_score(y_true, y_pred),
                    'tp': ((y_true == 1) & (y_pred == 1)).sum(),
                    'fp': ((y_true == 0) & (y_pred == 1)).sum(),
                    'fn': ((y_true == 1) & (y_pred == 0)).sum(),
                    'tn': ((y_true == 0) & (y_pred == 0)).sum()
                }
    
    if best_threshold is None:
        if verbose:
            print(f"Warning: No threshold found with recall >= {min_recall}")
            print("Trying to find threshold with maximum recall...")
        
        # If no threshold meets recall requirement, find one with maximum recall
        best_recall = -1
        for threshold in thresholds:
            y_pred = (y_pred_proba > threshold).astype(int)
            recall = recall_score(y_true, y_pred)
            
            if recall > best_recall:
                best_recall = recall
                best_threshold = threshold
                best_metrics = {
                    'threshold': threshold,
                    'precision': precision_score(y_true, y_pred),
                    'recall': recall,
                    'f1': f1_score(y_true, y_pred),
                    'tp': ((y_true == 1) & (y_pred == 1)).sum(),
                    'fp': ((y_true == 0) & (y_pred == 1)).sum(),
                    'fn': ((y_true == 1) & (y_pred == 0)).sum(),
                    'tn': ((y_true == 0) & (y_pred == 0)).sum()
                }
    
    if verbose:
        print("=" * 70)
        print(f"Threshold Optimization (Recall >= {min_recall}, Maximize Precision)")
        print("=" * 70)
        if best_metrics:
            print(f"Best Threshold:  {best_metrics['threshold']:.4f}")
            print(f"Precision:       {best_metrics['precision']:.4f}")
            print(f"Recall:          {best_metrics['recall']:.4f}")
            print(f"F1-Score:        {best_metrics['f1']:.4f}")
            print(f"\nConfusion Matrix:")
            print(f"True Positives:  {best_metrics['tp']}")
            print(f"False Positives: {best_metrics['fp']}")
            print(f"False Negatives: {best_metrics['fn']}")
            print(f"True Negatives:  {best_metrics['tn']}")
            print(f"\nValid thresholds found: {len(valid_thresholds)}")
            if len(valid_thresholds) > 0:
                print(f"Precision range: {min(v['precision'] for v in valid_thresholds):.4f} - {max(v['precision'] for v in valid_thresholds):.4f}")
        print("=" * 70)
    
    return best_threshold, best_metrics


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

