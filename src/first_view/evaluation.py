"""
Evaluation utilities for first view model.
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


def plot_feature_importance(model, top_n: int = 20, save_path: Optional[str] = None):
    """
    Plot feature importance.
    
    Parameters
    ----------
    model : FirstViewLightGBM
        Trained model
    top_n : int
        Number of top features to show
    save_path : str, optional
        Path to save the plot
    """
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


