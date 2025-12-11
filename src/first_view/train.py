"""
Training script for first view model.
"""

"""
Training script for first view model.
Optimized for PyCharm execution (both script and console).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path for PyCharm
# Handle both script execution and console execution
def get_project_root():
    """Get project root directory, works in both script and console."""
    # Try to use __file__ (works in script execution)
    if '__file__' in globals():
        return Path(__file__).parent.parent.parent
    
    # Fallback for console execution - find project root from current directory
    cwd = Path(os.getcwd())
    
    # Check if current directory is project root
    if (cwd / 'src').exists() and (cwd / 'data').exists():
        return cwd
    
    # Try to find project root by going up from current directory
    current = cwd
    for _ in range(5):  # Max 5 levels up
        if (current / 'src').exists() and (current / 'data').exists():
            return current
        current = current.parent
    
    # Final fallback - use current directory
    return cwd

project_root = get_project_root()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.first_view.data_loader import load_first_view_data
from src.first_view.preprocessing import preprocess_first_view_data
from src.first_view.feature_engineering import create_all_first_view_features
from src.first_view.models import train_first_view_model
from src.first_view.evaluation import evaluate_model, plot_feature_importance, plot_roc_curve
from sklearn.model_selection import train_test_split


def main():
    """
    Main training pipeline for first view model.
    """
    print("=" * 70)
    print("First View Model Training Pipeline")
    print("=" * 70)
    
    # 1. Load data
    print("\n[1/6] Loading data...")
    train_df, test_df = load_first_view_data(sample_size=None)  # Use all data
    
    # 2. Preprocessing
    print("\n[2/6] Preprocessing data...")
    train_df = preprocess_first_view_data(train_df, is_train=True)
    test_df = preprocess_first_view_data(test_df, is_train=False)
    
    # 3. Feature engineering
    print("\n[3/6] Creating features...")
    train_df = create_all_first_view_features(train_df, is_train=True)
    test_df = create_all_first_view_features(test_df, is_train=False)
    
    # 4. Prepare features and target
    print("\n[4/6] Preparing features...")
    # Exclude non-feature columns
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
    print("\n[5/6] Training model...")
    model = train_first_view_model(
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
        }
    )
    
    # 6. Evaluate model
    print("\n[6/6] Evaluating model...")
    
    # Validation predictions
    y_val_pred_proba = model.predict_proba(X_val)
    y_val_pred = model.predict(X_val)
    
    # Evaluate
    metrics = evaluate_model(y_val, y_val_pred, y_val_pred_proba)
    
    # Feature importance
    print("\nTop 20 Most Important Features:")
    if model.feature_importance_ is not None:
        print(model.feature_importance_.head(20))
    
    # Plot feature importance
    try:
        plot_feature_importance(model, top_n=20)
    except:
        print("Could not plot feature importance")
    
    # Plot ROC curve
    try:
        plot_roc_curve(y_val, y_val_pred_proba, title="First View Model - ROC Curve")
    except:
        print("Could not plot ROC curve")
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    
    # 7. Test set predictions
    print("\n" + "=" * 70)
    print("Test Set Predictions")
    print("=" * 70)
    
    # Prepare test features (same columns as training)
    X_test = test_df[feature_cols]
    
    print(f"Test set shape: {X_test.shape}")
    print("Making predictions on test set...")
    
    # Make predictions
    test_pred_proba = model.predict_proba(X_test)
    test_pred = model.predict(X_test)
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'TransactionID': test_df['TransactionID'].values,
        'isFraud': test_pred_proba
    })
    
    # Save submission
    from src.utils import get_project_root
    project_root = get_project_root()
    submission_path = project_root / 'submission_first_view.csv'
    submission_df.to_csv(submission_path, index=False)
    print(f"\nSubmission saved to: {submission_path}")
    
    # Test set statistics
    print("\nTest Set Prediction Statistics:")
    print(f"  Total predictions: {len(test_pred_proba):,}")
    print(f"  Mean predicted probability: {test_pred_proba.mean():.4f}")
    print(f"  Median predicted probability: {np.median(test_pred_proba):.4f}")
    print(f"  Min predicted probability: {test_pred_proba.min():.4f}")
    print(f"  Max predicted probability: {test_pred_proba.max():.4f}")
    print(f"  Predictions > 0.5 (predicted fraud): {(test_pred > 0).sum():,} ({(test_pred > 0).mean()*100:.2f}%)")
    print(f"  Predictions <= 0.5 (predicted normal): {(test_pred == 0).sum():,} ({(test_pred == 0).mean()*100:.2f}%)")
    
    # Distribution of predictions
    print("\nPrediction Distribution (by bins):")
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(test_pred_proba, bins=bins)
    for i in range(len(bins)-1):
        print(f"  {bins[i]:.1f} - {bins[i+1]:.1f}: {hist[i]:,} ({hist[i]/len(test_pred_proba)*100:.2f}%)")
    
    print("\n" + "=" * 70)
    print("Test Set Predictions Complete!")
    print("=" * 70)
    
    return model, metrics, submission_df


if __name__ == "__main__":
    model, metrics, submission_df = main()

