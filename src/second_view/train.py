"""
Training script for second view model.
Optimized for PyCharm execution (both script and console).
Includes threshold tuning (0.6-0.7) for better precision.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path for PyCharm
def get_project_root():
    """Get project root directory, works in both script and console."""
    if '__file__' in globals():
        return Path(__file__).parent.parent.parent
    
    cwd = Path(os.getcwd())
    
    if (cwd / 'src').exists() and (cwd / 'data').exists():
        return cwd
    
    current = cwd
    for _ in range(5):
        if (current / 'src').exists() and (current / 'data').exists():
            return current
        current = current.parent
    
    return cwd

project_root = get_project_root()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.second_view.data_loader import load_second_view_data
from src.second_view.preprocessing import preprocess_second_view_data
from src.second_view.feature_engineering import create_all_second_view_features
from src.second_view.models import train_second_view_model
from src.second_view.evaluation import (
    evaluate_model, 
    evaluate_multiple_thresholds,
    find_optimal_threshold,
    plot_feature_importance, 
    plot_roc_curve,
    plot_precision_recall_curve
)
from sklearn.model_selection import train_test_split


def main(threshold: float = 0.65):
    """
    Main training pipeline for second view model.
    
    Parameters
    ----------
    threshold : float
        Classification threshold (default: 0.65, recommended: 0.6-0.7)
    """
    print("=" * 70)
    print("Second View Model Training Pipeline")
    print("=" * 70)
    
    # 1. Load data
    print("\n[1/7] Loading data...")
    train_df, test_df = load_second_view_data(sample_size=None)  # Use all data
    
    # 2. Preprocessing
    print("\n[2/7] Preprocessing data...")
    train_df = preprocess_second_view_data(train_df, is_train=True)
    test_df = preprocess_second_view_data(test_df, is_train=False)
    
    # 3. Feature engineering
    print("\n[3/7] Creating features...")
    train_df = create_all_second_view_features(train_df, is_train=True)
    test_df = create_all_second_view_features(test_df, is_train=False)
    
    # 4. Prepare features and target
    print("\n[4/7] Preparing features...")
    exclude_cols = ['TransactionID', 'isFraud', 'TransactionDT']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    X_train = train_df[feature_cols]
    y_train = train_df['isFraud'].values
    
    # Split into train and validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_train
    )
    
    print(f"Training set: {X_train_split.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Fraud rate - Train: {y_train_split.mean():.4f}, Val: {y_val.mean():.4f}")
    
    # Identify categorical features
    categorical_features = []
    for col in feature_cols:
        if X_train_split[col].dtype == 'object' or X_train_split[col].dtype.name == 'category':
            categorical_features.append(col)
        elif X_train_split[col].dtype in ['int8', 'int16', 'int32', 'int64']:
            if X_train_split[col].nunique() < 50:
                categorical_features.append(col)
    
    print(f"Categorical features: {len(categorical_features)}")
    
    # 5. Train model
    print("\n[5/7] Training model...")
    model = train_second_view_model(
        X_train_split, y_train_split,
        X_val=X_val, y_val=y_val,
        categorical_features=categorical_features,
        model_params={
            'n_estimators': 2000,
            'learning_rate': 0.01,
            'max_depth': 7,
            'num_leaves': 31,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42
        },
        threshold=threshold
    )
    
    # 6. Evaluate model with multiple thresholds
    print("\n[6/7] Evaluating model...")
    
    # Validation predictions (probabilities)
    y_val_pred_proba = model.predict_proba(X_val)
    
    # Evaluate at multiple thresholds
    print("\nEvaluating at multiple thresholds...")
    threshold_results = evaluate_multiple_thresholds(
        y_val, y_val_pred_proba,
        thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
        verbose=True
    )
    
    # Find optimal threshold
    print("\nFinding optimal threshold (F1-score)...")
    optimal_threshold = find_optimal_threshold(
        y_val, y_val_pred_proba,
        metric='f1',
        threshold_range=(0.5, 0.8),
        n_thresholds=50
    )
    print(f"Optimal threshold (F1): {optimal_threshold:.4f}")
    
    # Evaluate with selected threshold
    y_val_pred = model.predict(X_val, threshold=threshold)
    metrics = evaluate_model(y_val, y_val_pred, y_val_pred_proba, threshold=threshold)
    
    # Feature importance
    print("\nTop 20 Most Important Features:")
    if model.feature_importance_ is not None:
        print(model.feature_importance_.head(20))
    
    # Plot feature importance
    try:
        plot_feature_importance(model, top_n=20)
    except Exception as e:
        print(f"Could not plot feature importance: {e}")
    
    # Plot ROC curve
    try:
        plot_roc_curve(y_val, y_val_pred_proba, title="Second View Model - ROC Curve")
    except Exception as e:
        print(f"Could not plot ROC curve: {e}")
    
    # Plot Precision-Recall curve
    try:
        plot_precision_recall_curve(y_val, y_val_pred_proba, title="Second View Model - Precision-Recall Curve")
    except Exception as e:
        print(f"Could not plot Precision-Recall curve: {e}")
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    
    # 7. Test set predictions
    print("\n[7/7] Making test set predictions...")
    
    X_test = test_df[feature_cols]
    
    test_pred_proba = model.predict_proba(X_test)
    test_pred = model.predict(X_test, threshold=threshold)
    
    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'TransactionID': test_df['TransactionID'].values,
        'isFraud': test_pred_proba
    })
    
    # Save submission file
    submission_path = project_root / "results" / "submission_second_view.csv"
    submission_path.parent.mkdir(exist_ok=True)
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file created at: {submission_path}")
    
    # Test set statistics
    print("\nTest Set Prediction Statistics:")
    print(f"  Total predictions: {len(test_pred_proba):,}")
    print(f"  Mean predicted probability: {test_pred_proba.mean():.4f}")
    print(f"  Median predicted probability: {np.median(test_pred_proba):.4f}")
    print(f"  Min predicted probability: {test_pred_proba.min():.4f}")
    print(f"  Max predicted probability: {test_pred_proba.max():.4f}")
    print(f"  Predictions > {threshold} (predicted fraud): {(test_pred > 0).sum():,} ({(test_pred > 0).mean()*100:.2f}%)")
    print(f"  Predictions <= {threshold} (predicted normal): {(test_pred == 0).sum():,} ({(test_pred == 0).mean()*100:.2f}%)")
    
    print("\n" + "=" * 70)
    print("Test Set Predictions Complete!")
    print("=" * 70)
    
    return model, metrics, threshold_results, optimal_threshold


if __name__ == "__main__":
    # Default threshold: 0.65 (between 0.6-0.7 as recommended)
    model, metrics, threshold_results, optimal_threshold = main(threshold=0.65)

