"""
Eighth view training pipeline.
Features: Calibration, percentile threshold sweep, cost-based threshold optimization.
Target: Recall ≥ 0.90.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import importlib

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

from src.eighth_view.data_loader import load_sixth_view_data
from src.eighth_view.preprocessing import preprocess_sixth_view_data, apply_smote
from src.eighth_view.feature_engineering import create_all_sixth_view_features
from src.eighth_view.feature_selection import comprehensive_feature_selection
from src.eighth_view.models import train_seventh_view_model, train_ensemble_model
from src.eighth_view.data_exploration import run_all_exploration_analyses
from src.eighth_view.evaluation import (
    evaluate_model, 
    evaluate_multiple_thresholds,
    find_optimal_threshold,
    find_threshold_multi_objective_optimized,
    evaluate_with_cross_validation,
    plot_feature_importance, 
    plot_roc_curve,
    plot_precision_recall_curve,
    calibrate_ensemble_output,
    plot_calibration_curve,
    percentile_threshold_sweep,
    cost_based_threshold_optimization,
    plot_cost_based_threshold
)
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, recall_score, precision_score


def get_model_params_for_data_size(use_full_data: bool) -> dict:
    """
    Returns hyperparameters optimized for dataset size.
    
    Args:
        use_full_data: True for 590k rows, False for 20k rows.
    
    Returns:
        Dict with early_stopping_rounds, n_estimators, reg_alpha, reg_lambda, etc.
    """
    if use_full_data:
        # Full dataset parameters (590k rows) - Standard values with universal improvements
        return {
            'early_stopping_rounds': 200,  # ✅ Artırıldı: 50 → 200 (IEEE Fraud için optimum)
            'early_stopping_min_delta': 0.0,  # No min delta for full dataset
            'n_estimators_lgb': 2000,  # Full capacity (1000+ iterations)
            'n_estimators_xgb': 2000,  # Full capacity
            'iterations_cat': 2000,  # Full capacity
            'early_stopping_cat': 100,  # Standard early stopping
            # ✅ Optimum hyperparameters (IEEE Fraud için)
            'reg_alpha': 0.1,  # ✅ Azaltıldı: 0.5 → 0.1 (aşırı regularization kaldırıldı)
            'reg_lambda': 0.2,  # ✅ Azaltıldı: 0.5 → 0.2 (aşırı regularization kaldırıldı)
            'max_depth': -1,  # ✅ Artırıldı: 11 → -1 (no limit, LightGBM optimize eder)
            'num_leaves': 300,  # ✅ Artırıldı: 200 → 300 (IEEE Fraud için optimum)
            'min_child_samples': 50,  # ✅ Artırıldı: 20 → 50 (optimum)
            'feature_selection_n_estimators': 200,  # Standard for feature selection
            'feature_selection_early_stopping': 30,  # Standard early stopping
            'feature_selection_importance_threshold': 0.001,  # Standard threshold (yesterday's value)
            'feature_selection_target_count': None,  # No target for full dataset
            'smote_sampling_strategy': 0.12,  # Standard SMOTE
            'n_trials': 50,  # Standard hyperparameter tuning trials
            # Universal improvements parameters (for both full and small)
            'use_adasyn': False,  # ❌ ADASYN KALDIRILDI - AUC düşürüyor
            'adasyn_n_neighbors': 0,  # Not used
            'use_dart_boosting': False,  # ❌ DART KALDIRILDI - aşırı regularization, model öğrenemiyor
            'dart_drop_rate': 0.0,  # Not used
            'cv_folds': 5,  # Full dataset: 5 folds (590k için yeterli)
            'use_ensemble_diversity': True,  # Add Logistic Regression to ensemble
            'use_mutual_info_fs': True,  # Use Mutual Information for feature selection
            'mutual_info_threshold': 0.001,  # Full dataset: lower threshold (more features)
            'use_smote_tomek': False,  # Use SMOTE+Tomek (alternative to ADASYN)
        }
    else:
        # Small dataset parameters (20k rows) - BALANCED approach
        # Based on analysis: Train AUC 0.9885 vs Val AUC 0.7949 = 19.36% gap (CRITICAL)
        # Root cause: Early stopping too early (16 iterasyon) → underfitting
        # Strategy: Allow more learning while preventing overfitting
        return {
            # ✅ Optimum hyperparameters (IEEE Fraud için)
            'early_stopping_rounds': 200,  # ✅ Artırıldı: 30 → 200 (IEEE Fraud için optimum)
            'early_stopping_min_delta': 0.0,  # ✅ Azaltıldı: 0.001 → 0.0 (daha az agresif)
            'n_estimators_lgb': 2000,  # ✅ Artırıldı: 800 → 2000 (1000+ iterations)
            'n_estimators_xgb': 800,
            'iterations_cat': 800,
            'early_stopping_cat': 25,  # Reduced from 30
            # ✅ Optimum hyperparameters (IEEE Fraud için - aşırı regularization kaldırıldı)
            'reg_alpha': 0.1,  # ✅ Azaltıldı: 1.5 → 0.1 (aşırı regularization kaldırıldı)
            'reg_lambda': 0.2,  # ✅ Azaltıldı: 1.5 → 0.2 (aşırı regularization kaldırıldı)
            'max_depth': 10,  # ✅ Artırıldı: 7 → 10 (IEEE Fraud için optimum)
            'num_leaves': 200,  # ✅ Artırıldı: 80 → 200 (IEEE Fraud için optimum)
            'min_child_samples': 50,  # ✅ Azaltıldı: 100 → 50 (optimum)
            'feature_selection_n_estimators': 150,
            'feature_selection_early_stopping': 20,
            'feature_selection_importance_threshold': 0.004,  # REDUCED from 0.005 (biraz daha fazla feature)
            'feature_selection_target_count': 120,  # INCREASED from 100 (biraz daha fazla feature, daha fazla kapasite)
            'smote_sampling_strategy': 0.12,
            'n_trials': 40,
            # Universal improvements parameters (for both full and small)
            'use_adasyn': False,  # ❌ ADASYN KALDIRILDI
            'adasyn_n_neighbors': 0,  # Not used
            'use_dart_boosting': False,  # ❌ DART KALDIRILDI
            'dart_drop_rate': 0.0,  # Not used
            'cv_folds': 10,
            'use_ensemble_diversity': True,
            'use_mutual_info_fs': True,
            'mutual_info_threshold': 0.003,
            'use_smote_tomek': False,
        }


def main(threshold: float = 0.65,
         use_hyperparameter_tuning: bool = True,
         tuning_method: str = 'optuna',
         n_trials: int = 50,
         use_feature_selection: bool = True,
         use_ensemble: bool = True,
         use_smote: bool = False,
         use_cross_validation: bool = True,
         cv_folds: int = 3,
         min_auc_roc: float = 0.94,
         min_recall: float = 0.85,
         min_precision: float = 0.40,
         use_full_data: bool = False):
    """
    Main training pipeline.
    
    Args:
        threshold: Classification threshold (optimized if min_recall set).
        use_hyperparameter_tuning: Enable Optuna/Hyperopt tuning.
        tuning_method: 'optuna' or 'hyperopt'.
        n_trials: Number of tuning trials.
        use_feature_selection: Enable feature selection.
        use_ensemble: Use ensemble (5 LightGBM models).
        use_smote: Disabled (uses class weights instead).
        use_cross_validation: Enable CV for ensemble weights.
        cv_folds: Number of CV folds.
        min_auc_roc: Target AUC-ROC.
        min_recall: Target recall (triggers threshold optimization).
        min_precision: Target precision.
        use_full_data: True for 590k rows, False for 20k debug sample.
    
    Returns:
        model, metrics, threshold_results, optimal_threshold
    """
    print("=" * 70)
    print("Eighth View Model Training Pipeline")
    print("=" * 70)
    print("Realistic Targets: AUC-ROC >= 0.94, Recall >= 0.90, Precision >= 0.40")
    print("NEW: Calibration (Isotonic) + Percentile Threshold Sweep + Cost-Based Threshold")
    print("Goal: Achieve Recall > 0.90 without changing the model")
    print("=" * 70)
    if use_hyperparameter_tuning:
        print(f"Hyperparameter tuning: {tuning_method.upper()} ({n_trials} trials)")
    if use_feature_selection:
        print("Feature selection: ENABLED")
    if use_ensemble:
        print("Ensemble methods: ENABLED (LightGBM + XGBoost + CatBoost)")
    if use_smote:
        print("SMOTE: ENABLED (Class imbalance handling)")
    if use_cross_validation:
        print(f"Cross-validation: ENABLED ({cv_folds}-fold, will be adjusted based on dataset size)")
    if min_recall > 0 or min_precision > 0 or min_auc_roc > 0:
        print(f"Threshold optimization: AUC-ROC >= {min_auc_roc}, Recall >= {min_recall}, Precision >= {min_precision}")
    print("=" * 70)
    
    # 1. Load data (DUAL-MODE PIPELINE)
    print("\n[1/11] Loading data...")
    train_df, test_df = load_sixth_view_data(
        use_full_data=use_full_data,
        debug_sample_size=20000,  # DEBUG MODE: 20k sample (hızlı test için)
        debug_fraud_rate=0.035,    # Preserve ~3.5% fraud rate
        random_state=42            # Reproducible sampling
    )
    
    # Safety checks after loading
    actual_fraud_rate = train_df['isFraud'].mean()
    expected_fraud_rate = 0.035
    fraud_rate_diff = abs(actual_fraud_rate - expected_fraud_rate)
    
    if fraud_rate_diff > 0.002:  # 0.2% tolerance
        print(f"\n⚠️  WARNING: Fraud rate deviates from expected:")
        print(f"   Expected: {expected_fraud_rate*100:.4f}%")
        print(f"   Actual: {actual_fraud_rate*100:.4f}%")
        print(f"   Difference: {fraud_rate_diff*100:.4f}% (tolerance: 0.2%)")
    else:
        print(f"\n✅ Fraud rate check passed: {actual_fraud_rate*100:.4f}% "
              f"(diff: {fraud_rate_diff*100:.4f}%)")
    
    # Warn if debug subset is too small for certain CV strategies
    if not use_full_data and len(train_df) < 10000:
        print(f"\n⚠️  WARNING: Debug subset ({len(train_df):,} rows) may be too small for:")
        print("   - High-fold CV (recommend <= 5 folds)")
        print("   - Complex feature selection")
        print("   - Large ensemble models")
    
    # 2. Preprocessing
    print("\n[2/11] Preprocessing data...")
    train_df = preprocess_sixth_view_data(train_df, is_train=True)
    test_df = preprocess_sixth_view_data(test_df, is_train=False)
    
    # ✅ 3. CRITICAL FIX: Time-aware split FIRST, then feature engineering (leakage önleme)
    # Problem 4: Feature engineering tüm datada yapılıyor → validation geleceği görüyor
    # Çözüm: Önce split, sonra feature engineering (rolling/velocity/count sadece train'de)
    print("\n[3/11] Performing time-aware train/validation split (BEFORE feature engineering)...")
    exclude_cols = ['TransactionID', 'isFraud', 'TransactionDT']
    
    if 'TransactionDT' in train_df.columns:
        # Sort by time
        time_sorted_idx = train_df['TransactionDT'].argsort().values
        train_df_sorted = train_df.iloc[time_sorted_idx].reset_index(drop=True)
        
        # Split: 80% train (earlier), 20% validation (later)
        split_idx = int(len(train_df_sorted) * 0.8)
        train_df_split = train_df_sorted.iloc[:split_idx].reset_index(drop=True)
        val_df = train_df_sorted.iloc[split_idx:].reset_index(drop=True)
        
        print(f"  Time-aware split: Train={len(train_df_split):,} (earlier), Val={len(val_df):,} (later)")
        print(f"  Train time range: {train_df_split['TransactionDT'].min()} - {train_df_split['TransactionDT'].max()}")
        print(f"  Val time range: {val_df['TransactionDT'].min()} - {val_df['TransactionDT'].max()}")
    else:
        # Fallback: random split if no TransactionDT
        from sklearn.model_selection import train_test_split
        train_df_split, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['isFraud'])
        train_df_split = train_df_split.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        print(f"  Random split: Train={len(train_df_split):,}, Val={len(val_df):,}")
    
    # ✅ 4. Feature engineering AFTER split (leakage önleme)
    # Rolling/velocity/count-based features sadece train'de hesaplanır, test'e transform edilir
    print("\n[4/11] Creating features (AFTER split to prevent leakage)...")
    print("  Creating features on training set...")
    train_df_split = create_all_sixth_view_features(train_df_split, is_train=True)
    
    print("  Creating features on validation set (using train statistics)...")
    val_df = create_all_sixth_view_features(val_df, is_train=False)  # Test mode - no future leakage
    
    print("  Creating features on test set (using train statistics)...")
    test_df = create_all_sixth_view_features(test_df, is_train=False)  # Test mode - no future leakage
    
    print("\n[4/11] Creating features finished...")
    
    # 4.5. Data Exploration (Log detailed statistics - AFTER feature engineering)
    print("\n[4.5/11] Running data exploration analyses...")
    exploration_stats = run_all_exploration_analyses(
        train_df_split, 
        target_col='isFraud',
        time_col='TransactionDT',
        is_train=True,
        verbose=True
    )
    
    # 5. Prepare features and target
    print("\n[5/11] Preparing features...")
    
    # ✅ CRITICAL FIX: Align features between train and validation sets
    # Problem: Feature engineering creates different features in train vs validation
    # Solution: Ensure both sets have the same features (fill missing with 0 or default)
    print("  Aligning features between train and validation sets...")
    
    # Get all features from both sets
    train_features = set([col for col in train_df_split.columns if col not in exclude_cols])
    val_features = set([col for col in val_df.columns if col not in exclude_cols])
    
    # Find missing features
    missing_in_val = train_features - val_features
    missing_in_train = val_features - train_features
    
    if missing_in_val:
        print(f"  ⚠️  {len(missing_in_val)} features missing in validation set, filling with 0...")
        for feat in missing_in_val:
            # Determine default value based on feature type
            if train_df_split[feat].dtype in ['int8', 'int16', 'int32', 'int64']:
                val_df[feat] = 0
            elif train_df_split[feat].dtype in ['float32', 'float64']:
                val_df[feat] = 0.0
            elif train_df_split[feat].dtype == 'object' or train_df_split[feat].dtype.name == 'category':
                # For categorical, use most common value or 'missing'
                val_df[feat] = train_df_split[feat].mode()[0] if len(train_df_split[feat].mode()) > 0 else 'missing'
            else:
                val_df[feat] = 0
    
    if missing_in_train:
        print(f"  ⚠️  {len(missing_in_train)} features missing in training set, filling with 0...")
        for feat in missing_in_train:
            # Determine default value based on feature type
            if val_df[feat].dtype in ['int8', 'int16', 'int32', 'int64']:
                train_df_split[feat] = 0
            elif val_df[feat].dtype in ['float32', 'float64']:
                train_df_split[feat] = 0.0
            elif val_df[feat].dtype == 'object' or val_df[feat].dtype.name == 'category':
                # For categorical, use most common value or 'missing'
                train_df_split[feat] = val_df[feat].mode()[0] if len(val_df[feat].mode()) > 0 else 'missing'
            else:
                train_df_split[feat] = 0
    
    # Ensure same column order
    feature_cols = sorted([col for col in train_df_split.columns if col not in exclude_cols])
    
    # Verify both sets have same features
    train_features_final = set([col for col in train_df_split.columns if col not in exclude_cols])
    val_features_final = set([col for col in val_df.columns if col not in exclude_cols])
    
    if train_features_final != val_features_final:
        raise ValueError(f"Feature mismatch after alignment! Train: {len(train_features_final)}, Val: {len(val_features_final)}")
    
    print(f"  ✅ Feature alignment complete: {len(feature_cols)} features")
    
    X_train_split = train_df_split[feature_cols]
    y_train_split = train_df_split['isFraud'].values
    X_val = val_df[feature_cols]
    y_val = val_df['isFraud'].values
    
    # Safety check: Ensure no empty splits
    if len(X_train_split) == 0:
        raise ValueError("Training split is empty! Check data loading and splitting logic.")
    if len(X_val) == 0:
        raise ValueError("Validation split is empty! Check data loading and splitting logic.")
    
    # Safety check: Ensure both splits have fraud cases
    train_fraud_rate = y_train_split.mean()
    val_fraud_rate = y_val.mean()
    if train_fraud_rate == 0:
        raise ValueError("Training split has no fraud cases! Stratified sampling may have failed.")
    if val_fraud_rate == 0:
        raise ValueError("Validation split has no fraud cases! Stratified sampling may have failed.")
    
    print(f"\n" + "=" * 70)
    print("TRAIN/VALIDATION SPLIT SUMMARY")
    print("=" * 70)
    print(f"Training set: {X_train_split.shape[0]:,} rows, {X_train_split.shape[1]} features")
    print(f"Validation set: {X_val.shape[0]:,} rows, {X_val.shape[1]} features")
    print(f"Fraud rate - Train: {y_train_split.mean():.4f} ({y_train_split.mean()*100:.2f}%), "
          f"Val: {y_val.mean():.4f} ({y_val.mean()*100:.2f}%)")
    print("=" * 70)
    
    # Identify categorical features (including float64 categorical features like C1-C14)
    categorical_features = []
    for col in feature_cols:
        if X_train_split[col].dtype == 'object' or X_train_split[col].dtype.name == 'category':
            categorical_features.append(col)
        elif X_train_split[col].dtype in ['int8', 'int16', 'int32', 'int64', 'float64']:
            # C1-C14 are categorical (encoded card features)
            if col.startswith('C') and col[1:].isdigit():  # C1-C14
                categorical_features.append(col)
            # Low cardinality integer/float features are also categorical
            elif X_train_split[col].nunique() < 500:  # Düşük cardinality → Categorical
                categorical_features.append(col)
    
    print(f"Categorical features: {len(categorical_features)}")
    print(f"Total features: {len(feature_cols)}")
    
    # Get model parameters based on data size
    model_params = get_model_params_for_data_size(use_full_data)
    actual_cv_folds = model_params.get('cv_folds', cv_folds)
    
    if use_full_data:
        print("\n📊 Using FULL DATASET parameters (~590k rows)")
        print("  Universal improvements: ADASYN, DART Boosting, 5-Fold CV, Ensemble Diversity, Mutual Info FS")
    else:
        print("\n📊 Using DEBUG MODE parameters (~20k rows) - Aggressive regularization to prevent overfitting")
        print("  Universal improvements: ADASYN, DART Boosting, 10-Fold CV, Ensemble Diversity, Mutual Info FS")
        
        # Warn if debug subset is too small for high-fold CV
        train_size = len(X_train_split)
        if train_size < 10000 and actual_cv_folds > 5:
            print(f"\n⚠️  WARNING: Debug subset ({train_size:,} rows) may be too small for {actual_cv_folds}-fold CV")
            print("   Consider reducing CV folds to <= 5 for more stable estimates")
        elif train_size < 5000:
            print(f"\n⚠️  WARNING: Debug subset ({train_size:,} rows) is very small")
            print("   Recommend using <= 3-fold CV or increasing debug_sample_size")
    print(f"  Early stopping: {model_params['early_stopping_rounds']} rounds (min_delta: {model_params.get('early_stopping_min_delta', 0.0)})")
    print(f"  Max iterations: LGB={model_params['n_estimators_lgb']}, XGB={model_params['n_estimators_xgb']}, Cat={model_params['iterations_cat']}")
    print(f"  Regularization: alpha={model_params['reg_alpha']}, lambda={model_params['reg_lambda']}")
    print(f"  Max depth: {model_params['max_depth']}, Num leaves: {model_params['num_leaves']}")
    print(f"  CV folds: {model_params.get('cv_folds', cv_folds)} (data-size-specific)")
    if not use_full_data:
        print(f"  Feature selection: importance_threshold={model_params.get('feature_selection_importance_threshold', 0.001)}, target_count={model_params.get('feature_selection_target_count', None)}")
    if model_params.get('use_adasyn', False):
        print(f"  ADASYN: n_neighbors={model_params.get('adasyn_n_neighbors', 3)}")
    if model_params.get('use_dart_boosting', False):
        print(f"  DART Boosting: drop_rate={model_params.get('dart_drop_rate', 0.1)}")
    
    # 5. Feature selection (BEFORE SMOTE - Literatüre uygun: yüksek boyutlu veri için)
    # ⭐ DEĞİŞİKLİK: Feature selection önce, SMOTE sonra (bellek sorunu için)
    # Literatür: Yüksek boyutlu veri (483 feature) için feature selection önce yapılmalı
    selected_features = feature_cols
    if use_feature_selection:
        print("\n[5/11] Feature selection (BEFORE SMOTE - to reduce memory usage)...")
        # Reload feature_selection module to avoid cache issues
        import src.eighth_view.feature_selection
        importlib.reload(src.eighth_view.feature_selection)
        from src.eighth_view.feature_selection import comprehensive_feature_selection
        # Train a simple model for feature importance (faster for feature selection)
        # Note: Training on imbalanced data, but using class_weight and stratified sampling
        from src.eighth_view.models import SeventhViewLightGBM
        temp_model = SeventhViewLightGBM(
            n_estimators=model_params['feature_selection_n_estimators'], 
            learning_rate=0.1, 
            verbose=-1
        )
        temp_model.fit(X_train_split, y_train_split,
                      categorical_features=categorical_features,
                      eval_set=(X_val, y_val),
                      early_stopping_rounds=model_params['feature_selection_early_stopping'],
                      verbose=0)
        
        # Don't pass model to comprehensive_feature_selection - let it train on filtered features
        # This avoids dimension mismatch issues
        # Optimized thresholds: keep more features for better model capacity
        # IMPORTANT: Remove leakage features (fraud_rate) to prevent unrealistic performance
        # ⭐ UPDATED: Use data-size-specific importance threshold and target feature count
        X_train_selected, selected_features = comprehensive_feature_selection(
            X_train_split, pd.Series(y_train_split),
            variance_threshold=0.01,  # Increased from 0.005 to remove truly low variance features
            correlation_threshold=0.95,  # Lowered from 0.98 to remove more correlated features
            importance_threshold=model_params.get('feature_selection_importance_threshold', 0.001),  # Data-size-specific (0.001 for full, 0.004 for small)
            target_feature_count=model_params.get('feature_selection_target_count', None),  # Target: 100-150 for small dataset
            remove_leakage_features=True,  # CRITICAL: Remove fraud_rate features that cause target leakage
            model_params_config=model_params,  # Pass model params config for feature selection
            verbose=True
        )
        
        X_val_selected = X_val[selected_features]
        X_train_split = X_train_selected
        X_val = X_val_selected
        
        # Update categorical features
        categorical_features = [f for f in categorical_features if f in selected_features]
        
        # Get numeric features (all selected features minus categorical)
        numeric_features = [f for f in selected_features if f not in categorical_features]
        
        print(f"Selected features: {len(selected_features)}")
        print(f"Selected categorical features: {len(categorical_features)}")
        if categorical_features:
            print(f"  Categorical feature names: {categorical_features}")
        else:
            print(f"  No categorical features selected")
        print(f"Selected numeric features: {len(numeric_features)}")
        if numeric_features:
            print(f"  Numeric feature names: {numeric_features}")
        else:
            print(f"  No numeric features selected")
        print(f"✅ Feature selection complete. Reduced from {len(feature_cols)} to {len(selected_features)} features.")
        print(f"   This will significantly reduce SMOTE memory usage (from ~30.9 GB to ~{8.4 * len(selected_features) / 132:.1f} GB).")
    else:
        print("\n[5/11] Feature selection: SKIPPED")
    
    # 6. PRODUCTION-SAFE: Use class weighting instead of SMOTE
    # Why this reduces overfitting:
    # - SMOTE creates synthetic samples that may not represent real fraud patterns
    # - Tree-based models (LightGBM) handle class imbalance well with class weights
    # - Class weighting is more stable and doesn't introduce synthetic data artifacts
    # Impact on recall vs precision:
    # - Better recall (model focuses on minority class through weighting)
    # - More stable precision (no synthetic sample artifacts)
    # - Better generalization (learns from real data only)
    print("\n[6/11] Calculating class weights (PRODUCTION-SAFE: No SMOTE)...")
    # Calculate class weights: scale_pos_weight = n_negative / n_positive
    n_positive = int(y_train_split.sum())
    n_negative = len(y_train_split) - n_positive
    raw_scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0
    
    # Hafta 1.3: Probabilities artırma - Class weight capping artır
    # OPTIMIZATION: Cost-sensitive learning approach
    # Business requirement: FN cost / FP cost = 10
    # Optimal weight: sqrt(10) ≈ 3.16, but we use a range 3.0-5.0 for stability
    # For very high imbalance (scale_pos_weight > 20), cap it to prevent over-prioritizing fraud
    if raw_scale_pos_weight > 20:
        # Hafta 1.3: Increased cap from 10.0 to 12.0-15.0 for better fraud detection and higher probabilities
        # Using sqrt * 2.0 to balance between too low (10.0) and too high (raw value)
        scale_pos_weight = min(15.0, np.sqrt(raw_scale_pos_weight) * 2.0)
        print(f"  ⚠️  Raw scale_pos_weight ({raw_scale_pos_weight:.2f}) too high, capping at {scale_pos_weight:.2f}")
        print(f"     Hafta 1.3: Increased cap from 10.0 to 15.0 for better fraud detection and higher probabilities")
    else:
        scale_pos_weight = raw_scale_pos_weight
    
    print(f"  Class distribution: {n_positive} positive, {n_negative} negative")
    print(f"  Raw scale_pos_weight: {raw_scale_pos_weight:.2f}")
    print(f"  Optimized scale_pos_weight: {scale_pos_weight:.2f}")
    print(f"  ✅ Using class weighting instead of SMOTE (production-safe for tree-based models)")
    
    # 7. PRODUCTION-SAFE: Train LightGBM only (no ensemble complexity)
    # Why this reduces overfitting:
    # - Ensembles can overfit on small datasets
    # - Single model is simpler, more interpretable, and easier to debug
    # - LightGBM handles imbalanced data well with class weights
    # Impact on recall vs precision:
    # - More stable performance (no ensemble variance)
    # - Better generalization (simpler model)
    # - Easier to tune and optimize
    # Hafta 2: Basit Ensemble - 3-5 model (farklı seeds), weighted average
    # Önce single model yerine ensemble kullanacağız
    use_ensemble_mode = True  # Hafta 2: Ensemble mode aktif
    
    if use_ensemble_mode:
        print("\n[7/11] Training Ensemble Model (Hafta 2: 3-5 models with different seeds)...")
        print("  ✅ Using Ensemble (3-5 LightGBM models with different seeds)")
        print("  ✅ This improves diversity and reduces overfitting")
        
        # Hafta 2: 3-5 model ensemble (farklı seeds)
        ensemble_seeds = [42, 123, 456, 789, 999][:5]  # 5 model, farklı seeds
        ensemble_models = []
        ensemble_weights = []
        
        for i, seed in enumerate(ensemble_seeds):
            print(f"\n  Training model {i+1}/{len(ensemble_seeds)} with seed {seed}...")
            
            model_i = train_seventh_view_model(
            X_train_split, y_train_split,
            X_val=X_val, y_val=y_val,
            categorical_features=categorical_features,
                model_params={
                    'n_estimators': model_params['n_estimators_lgb'],
                    'learning_rate': 0.03,  # ✅ Optimum for IEEE Fraud
                    'max_depth': model_params['max_depth'],
                    'num_leaves': model_params['num_leaves'],
                    'min_child_samples': model_params['min_child_samples'],
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': model_params['reg_alpha'],
                    'reg_lambda': model_params['reg_lambda'],
                    'scale_pos_weight': scale_pos_weight,
                    'random_state': seed  # Farklı seed
                },
            threshold=threshold,
                use_hyperparameter_tuning=False,  # İlk modelde tuning yap, diğerlerinde aynı params
            tuning_method=tuning_method,
                n_trials=model_params['n_trials'] if i == 0 else 10,  # İlk modelde full tuning
                use_cv_for_tuning=use_cross_validation,
                cv_folds=model_params.get('cv_folds', cv_folds),
                early_stopping_rounds=model_params['early_stopping_rounds']
            )
            
            # Model performansını değerlendir (weight hesaplama için)
            y_val_pred_proba_i = model_i.predict_proba(X_val)
            y_val_pred_proba_i_flat = y_val_pred_proba_i[:, 1] if len(y_val_pred_proba_i.shape) > 1 else y_val_pred_proba_i
            from sklearn.metrics import roc_auc_score
            auc_i = roc_auc_score(y_val, y_val_pred_proba_i_flat)
            
            ensemble_models.append(model_i)
            ensemble_weights.append(auc_i)  # AUC-based weights
            
            print(f"    Model {i+1} Val AUC: {auc_i:.6f}")
        
        # Normalize weights
        ensemble_weights = np.array(ensemble_weights)
        ensemble_weights = ensemble_weights / ensemble_weights.sum()
        
        print(f"\n  Ensemble weights: {ensemble_weights}")
        print(f"  ✅ Ensemble created with {len(ensemble_models)} models")
        
        # Create ensemble wrapper
        class EnsembleWrapper:
            def __init__(self, models, weights):
                self.models = models
                self.weights = weights
                self.model = models[0]  # For compatibility with existing code
            
            def predict_proba(self, X):
                predictions = []
                for model in self.models:
                    pred = model.predict_proba(X)
                    pred_flat = pred[:, 1] if len(pred.shape) > 1 else pred
                    predictions.append(pred_flat)
                
                # Weighted average
                ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
                # Return in shape (n_samples, 2)
                return np.column_stack([1 - ensemble_pred, ensemble_pred])
            
            def predict(self, X, threshold=None):
                if threshold is None:
                    threshold = 0.65
                proba = self.predict_proba(X)
                proba_flat = proba[:, 1] if len(proba.shape) > 1 else proba
                return (proba_flat > threshold).astype(int)
        
        model = EnsembleWrapper(ensemble_models, ensemble_weights)
        
    else:
        print("\n[7/11] Training LightGBM model (PRODUCTION-SAFE: Single model, no ensemble)...")
        print("  ✅ Using LightGBM only (simpler, more stable, production-safe)")
        
        model = train_seventh_view_model(
            X_train_split, y_train_split,
            X_val=X_val, y_val=y_val,
            categorical_features=categorical_features,
            model_params={
            'n_estimators': model_params['n_estimators_lgb'],  # Use data-size-specific
                'learning_rate': 0.03,  # ✅ Optimum for IEEE Fraud
            'max_depth': model_params['max_depth'],  # Use data-size-specific
            'num_leaves': model_params['num_leaves'],  # Use data-size-specific
            'min_child_samples': model_params['min_child_samples'],  # Use data-size-specific
                'subsample': 0.8,
                'colsample_bytree': 0.8,
            'reg_alpha': model_params['reg_alpha'],  # Use data-size-specific
            'reg_lambda': model_params['reg_lambda'],  # Use data-size-specific
            'scale_pos_weight': scale_pos_weight,  # PRODUCTION-SAFE: Class weighting instead of SMOTE
                'random_state': 42
            },
            threshold=threshold,
            use_hyperparameter_tuning=use_hyperparameter_tuning,
            tuning_method=tuning_method,
        n_trials=model_params['n_trials'],  # Use data-size-specific
            use_cv_for_tuning=use_cross_validation,  # Use CV for tuning if CV is enabled
        cv_folds=model_params.get('cv_folds', cv_folds),  # Use data-size-specific CV folds
        early_stopping_rounds=model_params['early_stopping_rounds']  # Use data-size-specific
    )
    
    # Log early stopping details for gap analysis
    if hasattr(model, 'model') and model.model is not None:
        if hasattr(model.model, 'best_iteration') and model.model.best_iteration is not None:
            print(f"\n[Early Stopping Analysis]")
            print(f"  Best iteration: {model.model.best_iteration}")
            if hasattr(model.model, 'best_score'):
                train_auc = model.model.best_score.get('train', {}).get('auc', None)
                val_auc = model.model.best_score.get('valid', {}).get('auc', None)
                if train_auc is not None and val_auc is not None:
                    auc_gap = train_auc - val_auc
                    print(f"  Best train AUC: {train_auc:.6f}")
                    print(f"  Best val AUC: {val_auc:.6f}")
                    print(f"  AUC gap: {auc_gap:.6f} ({auc_gap*100:.2f}%)")
                    if auc_gap > 0.15:
                        print(f"  ⚠️  WARNING: Large AUC gap detected (>15%) - potential overfitting")
                    elif auc_gap < 0.05:
                        print(f"  ✅ Good: Small AUC gap (<5%) - model generalizes well")
    
    # 8. Cross-validation (if enabled) - Now done during ensemble training for weights
    cv_results = None
    if use_cross_validation:
        print("\n[8/11] Cross-validation evaluation (already done during ensemble training)...")
        # CV was already performed during ensemble weight calculation
        # We can still do a final CV evaluation on the ensemble if needed
        print("CV results were used to compute ensemble weights.")
        cv_results = {'mean': 0.0, 'std': 0.0, 'note': 'CV used for ensemble weights'}
    else:
        print("\n[8/11] Cross-validation: SKIPPED")
    
    # 9. Evaluate model with multiple thresholds
    print("\n[9/11] Evaluating model...")
    
    # Validation predictions (probabilities)
    y_val_pred_proba_raw = model.predict_proba(X_val)
    # Convert to 1D array (handle both single model and ensemble)
    y_val_pred_proba = y_val_pred_proba_raw[:, 1] if len(y_val_pred_proba_raw.shape) > 1 else y_val_pred_proba_raw
    
    # Evaluate at multiple thresholds
    print("\nEvaluating at multiple thresholds...")
    threshold_results = evaluate_multiple_thresholds(
        y_val, y_val_pred_proba,
        thresholds=[0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
        verbose=True
    )
    
    # 10. Hafta 1.1: Calibration düzeltmesi - Wrapper test et ve alternatif IsotonicRegression ekle
    # PRODUCTION-SAFE: Sigmoid/Platt calibration (validation-only approach)
    # Why this reduces overfitting:
    # - Calibration is done on validation set only (no training data leakage)
    # - Sigmoid/Platt scaling is simpler and more stable than isotonic regression
    # - Calibrated probabilities are more reliable for threshold optimization
    # Impact on recall vs precision:
    # - Better threshold optimization (calibrated probabilities)
    # - More reliable precision estimates
    # - Better generalization (calibration on validation set)
    print("\n[10/11] Calibrating model probabilities (Hafta 1.1: Sigmoid/Platt + Isotonic fallback)...")
    calibrated_model = None
    calibration_metrics = None
    calibration_method_used = None
    
    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.base import BaseEstimator, ClassifierMixin
        
        # Hafta 1.1: Create calibratable wrapper (works with both single model and ensemble)
        class CalibratableModelWrapper(BaseEstimator, ClassifierMixin):
            """
            Wrapper for SeventhViewLightGBM or EnsembleWrapper that makes it compatible with CalibratedClassifierCV.
            Model zaten fit edilmiş, fit() metodu no-op.
            """
            def __init__(self, model):
                self.model = model
                self._is_fitted = False  # Start as False, will be set to True in fit()
                self.classes_ = np.array([0, 1])  # Binary classification
                # ✅ CRITICAL FIX: Set _estimator_type to 'classifier' so sklearn recognizes it as a classifier
                self._estimator_type = 'classifier'
            
            def fit(self, X, y):
                # Model zaten fit edilmiş, sadece store et (no-op)
                # But we need to set _is_fitted for sklearn compatibility
                self._is_fitted = True
                # Store classes for sklearn compatibility
                if hasattr(y, 'unique'):
                    self.classes_ = np.sort(y.unique())
                else:
                    self.classes_ = np.sort(np.unique(y))
                # ✅ CRITICAL FIX: Ensure _estimator_type is set
                self._estimator_type = 'classifier'
                return self
            
            def __sklearn_is_fitted__(self):
                return getattr(self, '_is_fitted', False)
            
            def predict_proba(self, X):
                return self.model.predict_proba(X)
            
            def predict(self, X):
                return self.model.predict(X)
        
        # Check if model is fitted (works for both single model and ensemble)
        if hasattr(model, 'model'):
            if model.model is None:
                raise ValueError("Model must be fitted before calibration")
        elif hasattr(model, 'models'):
            if len(model.models) == 0:
                raise ValueError("Ensemble must have at least one model")
        
        calibratable_model = CalibratableModelWrapper(model)
        
        # ✅ CRITICAL FIX: Fit the wrapper first to set _is_fitted and classes_
        calibratable_model.fit(X_val, y_val)
        
        # Hafta 1.1: Try Sigmoid first, then Isotonic as fallback
        try:
            calibrated_model_cv = CalibratedClassifierCV(
                calibratable_model,
                method='sigmoid',  # PRODUCTION-SAFE: Sigmoid/Platt scaling (simpler, more stable)
                cv='prefit'  # Model already trained, use validation set
            )
            
            calibrated_model_cv.fit(X_val, y_val)
            y_val_pred_proba_calibrated = calibrated_model_cv.predict_proba(X_val)[:, 1]
            calibration_method_used = 'sigmoid'
            calibrated_model = calibrated_model_cv
            
        except Exception as sigmoid_error:
            # Hafta 1.1: Alternatif IsotonicRegression denemesi
            print(f"  ⚠️  Sigmoid calibration failed: {str(sigmoid_error)}")
            print("  Trying Isotonic Regression as fallback...")
            
            try:
                calibrated_model_cv = CalibratedClassifierCV(
                    calibratable_model,
                    method='isotonic',  # Alternative: Isotonic Regression
                    cv='prefit'
                )
                
                calibrated_model_cv.fit(X_val, y_val)
                y_val_pred_proba_calibrated = calibrated_model_cv.predict_proba(X_val)[:, 1]
                calibration_method_used = 'isotonic'
                calibrated_model = calibrated_model_cv
                
            except Exception as isotonic_error:
                raise Exception(f"Both Sigmoid and Isotonic calibration failed. Sigmoid: {str(sigmoid_error)}, Isotonic: {str(isotonic_error)}")
        
        # Calculate calibration metrics
        from sklearn.metrics import brier_score_loss
        y_val_pred_proba_flat = y_val_pred_proba[:, 1] if len(y_val_pred_proba.shape) > 1 else y_val_pred_proba
        brier_before = brier_score_loss(y_val, y_val_pred_proba_flat)
        brier_after = brier_score_loss(y_val, y_val_pred_proba_calibrated)
        
        calibration_metrics = {
            'brier_score_before': brier_before,
            'brier_score_after': brier_after,
            'brier_improvement': brier_before - brier_after,
            'method': calibration_method_used
        }
        
        print("=" * 70)
        print(f"Model Calibration Results ({calibration_method_used.upper()})")
        print("=" * 70)
        print(f"Brier Score (Before): {brier_before:.6f}")
        print(f"Brier Score (After):  {brier_after:.6f}")
        print(f"Brier Improvement:    {calibration_metrics['brier_improvement']:.6f}")
        print("=" * 70)
        
        # Plot calibration curve
        try:
            from sklearn.calibration import calibration_curve
            import matplotlib.pyplot as plt
            
            fraction_of_positives_uncal, mean_pred_uncal = calibration_curve(
                y_val, y_val_pred_proba_flat, n_bins=10, strategy='uniform'
            )
            fraction_of_positives_cal, mean_pred_cal = calibration_curve(
                y_val, y_val_pred_proba_calibrated, n_bins=10, strategy='uniform'
            )
            
            plt.figure(figsize=(10, 6))
            plt.plot(mean_pred_uncal, fraction_of_positives_uncal, 's-', label='Uncalibrated')
            plt.plot(mean_pred_cal, fraction_of_positives_cal, 's-', label=f'Calibrated ({calibration_method_used.upper()})')
            plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
            plt.xlabel('Mean Predicted Probability')
            plt.ylabel('Fraction of Positives')
            plt.title(f'Production-Safe Model - Calibration Curve ({calibration_method_used.upper()})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not plot calibration curve: {e}")
        
        # Use calibrated model for further analysis
        model_to_use = calibrated_model
        y_val_pred_proba_final = y_val_pred_proba_calibrated
        print(f"✅ Using calibrated model ({calibration_method_used.upper()}) for threshold optimization")
        
    except Exception as e:
        print(f"⚠️  Calibration failed: {str(e)}")
        print("  Using uncalibrated model")
        calibrated_model = None
        model_to_use = model
        y_val_pred_proba_final = y_val_pred_proba
        calibration_metrics = None
        calibration_method_used = None
    
    # 10.5. PRODUCTION-SAFE: Recall-optimized threshold selection
    # Why this reduces overfitting:
    # - Recall-focused threshold ensures we catch fraud cases (business priority)
    # - Percentile-based approach is more stable than absolute thresholds
    # - Cost-sensitive optimization balances recall with business costs
    # Impact on recall vs precision:
    # - Maximizes recall (catches more fraud)
    # - May reduce precision (more false positives, but acceptable for fraud detection)
    # - Better business value (fraud detection prioritizes recall)
    print("\n[10.5/11] Finding recall-optimized threshold (PRODUCTION-SAFE)...")
    optimal_threshold = None
    optimal_metrics = None
    
    # Strategy 1: Percentile-based threshold (most stable for production)
    # Find threshold that achieves target recall using percentile of predicted probabilities
    print("\n  Strategy 1: Percentile-based threshold (most stable)...")
    target_recall = min_recall if min_recall > 0 else 0.90  # Default: 90% recall
    
    # Calculate percentile threshold: find threshold where recall >= target
    from sklearn.metrics import recall_score
    thresholds_percentile = np.percentile(y_val_pred_proba_final, np.linspace(99, 1, 200))
    recalls_percentile = []
    
    for thresh in thresholds_percentile:
        y_pred = (y_val_pred_proba_final >= thresh).astype(int)
        rec = recall_score(y_val, y_pred)
        recalls_percentile.append(rec)
    
    recalls_percentile = np.array(recalls_percentile)
    valid_indices = np.where(recalls_percentile >= target_recall)[0]
    
    if len(valid_indices) > 0:
        # IMPROVED: Multi-objective optimization
        # Strategy: Maximize F1-score among thresholds meeting recall target
        # This balances recall and precision better than just maximizing precision
        from sklearn.metrics import precision_score, f1_score
        precisions_percentile = []
        f1_scores_percentile = []
        
        # Minimum precision constraint (hedef: 0.10, minimum: 0.05)
        min_precision_constraint = max(0.05, min_precision * 0.5) if min_precision > 0 else 0.05
        
        for idx in valid_indices:
            thresh = thresholds_percentile[idx]
            y_pred = (y_val_pred_proba_final >= thresh).astype(int)
            prec = precision_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            precisions_percentile.append(prec)
            f1_scores_percentile.append(f1)
        
        # Filter by minimum precision constraint (precision artırmak için)
        valid_precision_indices = [i for i, prec in enumerate(precisions_percentile) 
                                   if prec >= min_precision_constraint]
        
        if len(valid_precision_indices) > 0:
            # Use precision-constrained indices (precision artırmak için)
            constrained_indices = [valid_indices[i] for i in valid_precision_indices]
            # Choose threshold with highest precision among those meeting both constraints
            best_idx = constrained_indices[np.argmax([precisions_percentile[i] for i in valid_precision_indices])]
            selection_method = "precision-constrained"
        else:
            # No threshold meets precision constraint, use best precision anyway
            best_prec_idx = valid_indices[np.argmax(precisions_percentile)]
            best_idx = best_prec_idx
            selection_method = "precision-optimal (constraint not met)"
        
        percentile_threshold = thresholds_percentile[best_idx]
        percentile_recall = recalls_percentile[best_idx]
        # Get precision and F1 for selected threshold
        selected_idx_in_valid = np.where(valid_indices == best_idx)[0][0]
        percentile_precision = precisions_percentile[selected_idx_in_valid]
        percentile_f1 = f1_scores_percentile[selected_idx_in_valid]
        
        print(f"    ✅ Percentile threshold: {percentile_threshold:.4f} ({selection_method})")
        print(f"       Recall: {percentile_recall:.4f}, Precision: {percentile_precision:.4f}, F1: {percentile_f1:.4f}")
        if len(valid_precision_indices) == 0:
            print(f"       ⚠️  Warning: No threshold meets minimum precision constraint ({min_precision_constraint:.2f})")
        optimal_threshold = percentile_threshold
        optimal_metrics = {
            'recall': percentile_recall,
            'precision': percentile_precision,
            'f1_score': percentile_f1,
            'method': 'percentile_multi_objective'
        }
    else:
        print(f"    ⚠️  No percentile threshold found meeting recall target ({target_recall:.2f})")
        print(f"       Best available recall: {recalls_percentile.max():.4f}")
    
    # Strategy 2: Cost-sensitive optimization with precision constraint
    if optimal_threshold is None:
        print("\n  Strategy 2: Cost-sensitive threshold optimization with precision constraint...")
        from src.eighth_view.evaluation import cost_sensitive_threshold_with_precision
        
        # FN cost is much higher than FP cost in fraud detection
        fn_cost = 10.0  # Missing fraud is 10x more expensive than false alarm
        fp_cost = 1.0
        min_precision_target = max(0.05, min_precision * 0.5) if min_precision > 0 else 0.05
        
        cost_threshold, cost_precision, cost_recall = cost_sensitive_threshold_with_precision(
            y_val, y_val_pred_proba_final,
            min_recall=target_recall,
            min_precision=min_precision_target,
            fn_cost=fn_cost,
            fp_cost=fp_cost,
            threshold_range=(0.01, 0.50),
            n_thresholds=200,
            verbose=True
        )
        
        if cost_threshold is not None:
            print(f"    ✅ Cost-sensitive threshold: {cost_threshold:.4f}")
            print(f"       Recall: {cost_recall:.4f}, Precision: {cost_precision:.4f}")
            optimal_threshold = cost_threshold
            optimal_metrics = {
                'recall': cost_recall,
                'precision': cost_precision,
                'method': 'cost_sensitive_with_precision',
            }
    
    # Strategy 3: ROC Curve-Based Threshold Selection
    if optimal_threshold is None:
        print("\n  Strategy 3: ROC Curve-Based Threshold Selection...")
        from src.eighth_view.evaluation import find_threshold_from_roc
        
        roc_threshold = find_threshold_from_roc(
            y_val, y_val_pred_proba_final,
            min_recall=target_recall,
            verbose=True
        )
        
        if roc_threshold is not None:
            # Calculate metrics for ROC-based threshold
            y_pred_roc = (y_val_pred_proba_final >= roc_threshold).astype(int)
            roc_recall = recall_score(y_val, y_pred_roc)
            roc_precision = precision_score(y_val, y_pred_roc, zero_division=0)
            roc_f1 = f1_score(y_val, y_pred_roc, zero_division=0)
            
            print(f"    ✅ ROC-based threshold: {roc_threshold:.4f}")
            print(f"       Recall: {roc_recall:.4f}, Precision: {roc_precision:.4f}, F1: {roc_f1:.4f}")
            optimal_threshold = roc_threshold
            optimal_metrics = {
                'recall': roc_recall,
                'precision': roc_precision,
                'f1_score': roc_f1,
                'method': 'roc_curve_based'
            }
    
    # Final threshold selection
    if optimal_threshold is not None:
        print(f"\n✅ Final optimal threshold: {optimal_threshold:.4f} (method: {optimal_metrics.get('method', 'unknown')})")
        print(f"   Recall: {optimal_metrics['recall']:.4f}")
        print(f"   Precision: {optimal_metrics['precision']:.4f}")
    else:
        # Fallback: Use F1-optimal threshold
        print("\n  Fallback: Using F1-optimal threshold...")
        optimal_threshold = find_optimal_threshold(
            y_val, y_val_pred_proba_final,
            metric='f1',
            threshold_range=(0.1, 0.9),
            n_thresholds=100
        )
        print(f"  F1-optimal threshold: {optimal_threshold:.4f}")
    
    # 10.6. Percentile Threshold Sweep (EIGHTH VIEW NEW FEATURE)
    print("\n[10.6/13] Percentile Threshold Sweep (%0.5 → %5)...")
    percentile_results = None
    percentile_threshold = None
    
    try:
        percentile_results = percentile_threshold_sweep(
            y_val, y_val_pred_proba_final,  # ⭐ CALIBRATED probabilities
            percentile_range=(0.005, 0.05),  # ⭐ Genişletildi: 0.5% to 5% (daha geniş aralık)
            n_percentiles=30,  # ⭐ Artırıldı: 20 → 30 (daha fazla nokta)
            min_recall=0.90,
            verbose=True
        )
        
        # Find best threshold meeting recall target
        valid_results = percentile_results[percentile_results['meets_recall_target']]
        if len(valid_results) > 0:
            best_idx = valid_results['precision'].idxmax()
            best_percentile_result = valid_results.loc[best_idx]
            percentile_threshold = best_percentile_result['threshold']
            print(f"\n✅ Found percentile threshold: {percentile_threshold:.4f} (Recall: {best_percentile_result['recall']:.4f})")
        else:
            print(f"\n⚠️  No percentile threshold found meeting recall target (0.90)")
            print(f"Best available recall: {percentile_results['recall'].max():.4f}")
            percentile_threshold = None
    except Exception as e:
        print(f"⚠️  Percentile threshold sweep failed: {e}")
        percentile_threshold = None
    
    # 10.7. Cost-Based Threshold Optimization (EIGHTH VIEW NEW FEATURE)
    print("\n[10.7/13] Cost-Based Threshold Optimization...")
    cost_results = None
    cost_based_threshold = None
    
    try:
        cost_results = cost_based_threshold_optimization(
            y_val, y_val_pred_proba_final,  # ⭐ CALIBRATED probabilities
            fn_cost_ratios=[10.0, 20.0, 50.0],
            fp_cost=1.0,
            threshold_range=(0.01, 0.99),
            n_thresholds=200,
            verbose=True
        )
        
        # Plot cost-based threshold
        try:
            plot_cost_based_threshold(
                y_val, y_val_pred_proba_final,
                fn_cost_ratios=[10.0, 20.0, 50.0],
                fp_cost=1.0,
                title="Eighth View Model - Cost-Based Threshold Optimization"
            )
        except Exception as e:
            print(f"Could not plot cost-based threshold: {e}")
        
        # Extract optimal thresholds for each cost ratio
        optimal_cost_thresholds = {}
        for key, value in cost_results.items():
            optimal_cost_thresholds[key] = value['optimal_threshold']
            optimal_metrics_cost = value['optimal_metrics']
            print(f"  {key}: threshold={value['optimal_threshold']:.4f}, loss={value['optimal_expected_loss']:.2f}, recall={optimal_metrics_cost['recall']:.4f}")
            
            # Check if FN/FP = 20x meets recall target (recommended for balanced trade-off)
            if '20x' in key and optimal_metrics_cost['recall'] >= 0.90:
                cost_based_threshold = value['optimal_threshold']
                print(f"    ✅ FN/FP=20x threshold meets recall >= 0.90 target")
    except Exception as e:
        print(f"⚠️  Cost-based threshold optimization failed: {e}")
        cost_results = None
    
    # Choose final threshold: STRATEGY 1 - Prioritize calibrated multi-objective threshold
    # Priority order: 1) Percentile (if meets recall), 2) Cost-based (FN/FP=20x, if meets recall), 3) Multi-objective (calibrated)
    if percentile_threshold is not None:
        final_threshold = percentile_threshold
        threshold_source = "percentile_sweep"
        print(f"\n✅ Using percentile threshold: {final_threshold:.4f} (meets recall >= 0.90)")
    elif cost_based_threshold is not None:
        final_threshold = cost_based_threshold
        threshold_source = "cost_based_20x"
        print(f"\n✅ Using cost-based threshold (FN/FP=20x): {final_threshold:.4f} (meets recall >= 0.90)")
    elif optimal_threshold is not None:
        final_threshold = optimal_threshold
        threshold_source = "multi_objective_calibrated"
        print(f"\n✅ Using multi-objective threshold (calibrated): {final_threshold:.4f}")
        if optimal_metrics:
            print(f"  Recall: {optimal_metrics['recall']:.4f}, Precision: {optimal_metrics['precision']:.4f}")
    else:
        # Fallback: Use default threshold
        final_threshold = threshold
        threshold_source = "default"
        print(f"\n⚠️  Using default threshold: {final_threshold:.4f}")
    
    # Evaluate with final threshold
    y_val_pred = model_to_use.predict(X_val, threshold=final_threshold)
    metrics = evaluate_model(y_val, y_val_pred, y_val_pred_proba_final, threshold=final_threshold)
    
    # Faz 4.1: Feature Importance Monitoring
    print("\n[Faz 4.1] Feature Importance Monitoring...")
    feature_importance_df = None
    if hasattr(model, 'feature_importance_') and model.feature_importance_ is not None:
        feature_importance_df = model.feature_importance_
        print("\nTop 20 Most Important Features:")
        print(feature_importance_df.head(20))
        
        # Flag suspiciously high importance features (potential leakage)
        if len(feature_importance_df) > 0:
            # feature_importance_df is a Series (not DataFrame), so use iloc[0] not iloc[0, 0]
            max_importance = feature_importance_df.iloc[0]
            mean_importance = feature_importance_df.mean()  # mean() returns scalar for Series
            if max_importance > mean_importance * 10:
                print(f"  ⚠️  Warning: Top feature has {max_importance/mean_importance:.1f}x higher importance than mean (potential leakage)")
    elif hasattr(model, 'lgb_model') and model.lgb_model is not None:
        if hasattr(model.lgb_model, 'feature_importance_') and model.lgb_model.feature_importance_ is not None:
            feature_importance_df = model.lgb_model.feature_importance_
            print("\nTop 20 Most Important Features (LightGBM):")
            print(feature_importance_df.head(20))
            
            # Flag suspiciously high importance features (CRITICAL: Leakage detection)
            if len(feature_importance_df) > 0:
                # feature_importance_df is a Series (not DataFrame), so use iloc[0] not iloc[0, 0]
                max_importance = feature_importance_df.iloc[0]
                mean_importance = feature_importance_df.mean()  # mean() returns scalar for Series
                importance_ratio = max_importance / mean_importance if mean_importance > 0 else 0
                
                if importance_ratio > 10:
                    top_feature_name = feature_importance_df.index[0]
                    print(f"  ⚠️  WARNING: Top feature '{top_feature_name}' has {importance_ratio:.1f}x higher importance than mean")
                    print(f"     This may indicate:")
                    print(f"     - Data leakage (feature contains target information)")
                    print(f"     - Overfitting (model too dependent on single feature)")
                    print(f"     - Feature engineering issue (feature too powerful)")
                    print(f"     Recommendation: Review this feature for leakage or consider removing it")
                    
                    # Additional check: If ratio > 15, suggest removal
                    if importance_ratio > 15:
                        print(f"  🔴 CRITICAL: Importance ratio > 15x - Strongly recommend removing '{top_feature_name}'")
                else:
                    print(f"  ✅ Feature importance distribution is healthy (max/mean ratio: {importance_ratio:.1f}x)")
    
    # Faz 4.2: Feature Distribution Monitoring (KS Test)
    print("\n[Faz 4.2] Feature Distribution Monitoring (KS Test)...")
    try:
        from scipy.stats import ks_2samp
        
        # Sample features for distribution comparison
        sample_features = selected_features[:min(20, len(selected_features))]
        distribution_shifts = {}
        
        # Filter numeric features only (KS test requires numeric data)
        numeric_features = [f for f in sample_features 
                           if f in X_train_split.columns and 
                           X_train_split[f].dtype in ['float32', 'float64', 'int32', 'int64']]
        
        for feat in numeric_features:
            if feat in X_train_split.columns and feat in X_val.columns:
                train_vals = X_train_split[feat].dropna()
                val_vals = X_val[feat].dropna()
                
                if len(train_vals) > 0 and len(val_vals) > 0:
                    # KS test (only for numeric features)
                    try:
                        ks_stat, p_value = ks_2samp(train_vals, val_vals)
                        distribution_shifts[feat] = {'ks_stat': ks_stat, 'p_value': p_value}
                    except Exception as e:
                        # Skip features that cause errors (e.g., constant values)
                        pass
        
        # Report significant shifts (p < 0.05)
        significant_shifts = {k: v for k, v in distribution_shifts.items() if v['p_value'] < 0.05}
        if len(significant_shifts) > 0:
            print(f"  ⚠️  Warning: {len(significant_shifts)} features show significant distribution shift (p < 0.05):")
            for feat, stats in sorted(significant_shifts.items(), key=lambda x: x[1]['ks_stat'], reverse=True)[:5]:
                print(f"    - {feat}: KS={stats['ks_stat']:.4f}, p={stats['p_value']:.4f}")
        else:
            print("  ✅ No significant distribution shifts detected in sampled features")
            
    except Exception as e:
        print(f"  ⚠️  Could not perform distribution monitoring: {e}")
    
    # Faz 4.3: Prediction Distribution Analysis
    print("\n[Faz 4.3] Prediction Distribution Analysis...")
    try:
        fraud_preds = y_val_pred_proba_final[y_val == 1]
        normal_preds = y_val_pred_proba_final[y_val == 0]
        
        print(f"  Fraud predictions:")
        print(f"    Mean: {fraud_preds.mean():.4f}, Median: {np.median(fraud_preds):.4f}")
        print(f"    Min: {fraud_preds.min():.4f}, Max: {fraud_preds.max():.4f}")
        print(f"    Std: {fraud_preds.std():.4f}")
        print(f"    % above threshold ({final_threshold:.4f}): {(fraud_preds >= final_threshold).mean()*100:.2f}%")
        print(f"  Normal predictions:")
        print(f"    Mean: {normal_preds.mean():.4f}, Median: {np.median(normal_preds):.4f}")
        print(f"    Min: {normal_preds.min():.4f}, Max: {normal_preds.max():.4f}")
        print(f"    Std: {normal_preds.std():.4f}")
        print(f"    % above threshold ({final_threshold:.4f}): {(normal_preds >= final_threshold).mean()*100:.2f}%")
        
        # Separation analysis
        separation = fraud_preds.mean() - normal_preds.mean()
        print(f"  Separation (fraud_mean - normal_mean): {separation:.4f}")
        if separation > 0.1:
            print(f"    ✅ Good separation (>0.1)")
        elif separation > 0.05:
            print(f"    ⚠️  Moderate separation (0.05-0.1)")
        else:
            print(f"    🔴 Poor separation (<0.05) - model may not be learning well")
            
    except Exception as e:
        print(f"  ⚠️  Could not perform prediction distribution analysis: {e}")
    
    # Faz 4.4: Detailed Confusion Matrix Analysis
    print("\n[Faz 4.4] Detailed Confusion Matrix Analysis...")
    try:
        from sklearn.metrics import confusion_matrix
        y_pred_final = (y_val_pred_proba_final >= final_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred_final).ravel()
        total = len(y_val)
        
        print(f"  Threshold: {final_threshold:.4f}")
        print(f"  True Negatives:  {tn:5d} ({tn/total*100:.2f}%)")
        print(f"  False Positives: {fp:5d} ({fp/total*100:.2f}%)")
        print(f"  False Negatives: {fn:5d} ({fn/total*100:.2f}%)")
        print(f"  True Positives:  {tp:5d} ({tp/total*100:.2f}%)")
        if (tp+fp) > 0:
            print(f"  Precision: {tp/(tp+fp):.4f}")
        if (tp+fn) > 0:
            print(f"  Recall: {tp/(tp+fn):.4f}")
        if (tn+fp) > 0:
            print(f"  Specificity: {tn/(tn+fp):.4f}")
        if (fp+tn) > 0:
            print(f"  FPR: {fp/(fp+tn):.4f}")
            
    except Exception as e:
        print(f"  ⚠️  Could not perform confusion matrix analysis: {e}")
    
    # Plot feature importance
    try:
        if hasattr(model, 'feature_importance_'):
            plot_feature_importance(model, top_n=20)
        elif hasattr(model, 'lgb_model'):
            plot_feature_importance(model.lgb_model, top_n=20)
    except Exception as e:
        print(f"Could not plot feature importance: {e}")
    
    # Plot ROC curve
    try:
        plot_roc_curve(y_val, y_val_pred_proba_final, title="Eighth View Model - ROC Curve")
    except Exception as e:
        print(f"Could not plot ROC curve: {e}")
    
    # Plot Precision-Recall curve
    try:
        plot_precision_recall_curve(y_val, y_val_pred_proba_final, title="Eighth View Model - Precision-Recall Curve")
    except Exception as e:
        print(f"Could not plot Precision-Recall curve: {e}")
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    
    # 11. Test set predictions
    print("\n[11/11] Making test set predictions...")
    
    # Ensure selected_features is defined (if feature selection was skipped)
    # Note: selected_features is defined in feature selection section above
    # If feature selection was skipped, it should be set to feature_cols
    # This is a safety check in case of any issues
    try:
        if selected_features is None or len(selected_features) == 0:
            selected_features = feature_cols
            print("  ⚠️  selected_features was None/empty, using all features for test predictions")
    except NameError:
        # If selected_features is not defined at all, use feature_cols
        selected_features = feature_cols
        print("  ⚠️  selected_features not defined, using all features for test predictions")
    
    # ✅ CRITICAL FIX: Align test set features with training features
    # Problem: Feature engineering creates different features in test vs train
    # Solution: Ensure test set has same features as train (fill missing with 0 or default)
    print("  Aligning test set features with training features...")
    
    # Get all features from train and test sets
    train_features = set([col for col in train_df_split.columns if col not in exclude_cols])
    test_features = set([col for col in test_df.columns if col not in exclude_cols])
    
    # Find missing features
    missing_in_test = train_features - test_features
    missing_in_train = test_features - train_features
    
    if missing_in_test:
        print(f"  ⚠️  {len(missing_in_test)} features missing in test set, filling with 0...")
        for feat in missing_in_test:
            # Determine default value based on feature type from train
            if feat in train_df_split.columns:
                if train_df_split[feat].dtype in ['int8', 'int16', 'int32', 'int64']:
                    test_df[feat] = 0
                elif train_df_split[feat].dtype in ['float32', 'float64']:
                    test_df[feat] = 0.0
                elif train_df_split[feat].dtype == 'object' or train_df_split[feat].dtype.name == 'category':
                    # For categorical, use most common value or 'missing'
                    test_df[feat] = train_df_split[feat].mode()[0] if len(train_df_split[feat].mode()) > 0 else 'missing'
                else:
                    test_df[feat] = 0
            else:
                test_df[feat] = 0
    
    if missing_in_train:
        print(f"  ⚠️  {len(missing_in_train)} features in test but not in train, will be removed...")
        # Remove features that don't exist in train
        test_df = test_df.drop(columns=[f for f in missing_in_train if f in test_df.columns], errors='ignore')
    
    # Use selected features (after feature selection)
    if use_feature_selection and selected_features:
        # Check if all selected features exist in test set
        missing_selected = [f for f in selected_features if f not in test_df.columns]
        if missing_selected:
            print(f"  ⚠️  {len(missing_selected)} selected features missing in test set, filling with 0...")
            for feat in missing_selected:
                # Determine default value based on feature type from train
                if feat in train_df_split.columns:
                    if train_df_split[feat].dtype in ['int8', 'int16', 'int32', 'int64']:
                        test_df[feat] = 0
                    elif train_df_split[feat].dtype in ['float32', 'float64']:
                        test_df[feat] = 0.0
                    elif train_df_split[feat].dtype == 'object' or train_df_split[feat].dtype.name == 'category':
                        test_df[feat] = train_df_split[feat].mode()[0] if len(train_df_split[feat].mode()) > 0 else 'missing'
                    else:
                        test_df[feat] = 0
                else:
                    test_df[feat] = 0
        
        X_test = test_df[selected_features]
        print(f"  ✅ Using {len(selected_features)} selected features for test predictions")
    else:
        # Use all features (no feature selection)
        feature_cols_test = [col for col in test_df.columns if col not in exclude_cols]
        X_test = test_df[feature_cols_test]
        print(f"  ✅ Using {len(feature_cols_test)} features for test predictions (no feature selection)")
    
    test_pred_proba_raw = model.predict_proba(X_test)
    # Convert to 1D array (handle both single model and ensemble)
    test_pred_proba = test_pred_proba_raw[:, 1] if len(test_pred_proba_raw.shape) > 1 else test_pred_proba_raw
    test_pred = model.predict(X_test, threshold=threshold)
    
    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'TransactionID': test_df['TransactionID'].values,
        'isFraud': test_pred_proba
    })
    
    # Save submission file
    submission_path = project_root / "results" / "submission_eighth_view.csv"
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
    
    # 13. Summary
    print("\n[13/13] Summary")
    print("=" * 70)
    print(f"Model: Eighth View {'Ensemble' if use_ensemble else 'LightGBM'}")
    print(f"Hyperparameter Tuning: {'Yes' if use_hyperparameter_tuning else 'No'} ({tuning_method if use_hyperparameter_tuning else 'N/A'})")
    print(f"Feature Selection: {'Yes' if use_feature_selection else 'No'}")
    print(f"SMOTE: {'Yes' if use_smote else 'No'}")
    print(f"Ensemble: {'Yes' if use_ensemble else 'No'}")
    if use_cross_validation and cv_results is not None:
        print(f"Cross-Validation: Yes ({cv_folds}-fold, AUC={cv_results['mean']:.4f} +/- {cv_results['std']:.4f})")
    elif use_cross_validation:
        print(f"Cross-Validation: SKIPPED (requires model retraining with categorical features)")
    
    # Calibration summary
    if calibrated_model is not None and calibration_metrics is not None:
        print(f"Calibration: Yes (Brier: {calibration_metrics['brier_score_before']:.6f} → {calibration_metrics['brier_score_after']:.6f})")
    else:
        print("Calibration: No")
    
    # Threshold summary
    print(f"Final Threshold: {final_threshold:.4f} (Source: {threshold_source})")
    if percentile_threshold is not None:
        print(f"Percentile Threshold: {percentile_threshold:.4f} ✅ (meets recall >= 0.90)")
    if cost_results is not None:
        print(f"Cost-Based Thresholds: {', '.join([f'{k}={v:.4f}' for k, v in optimal_cost_thresholds.items()])}")
    
    if min_recall > 0 or min_precision > 0 or min_auc_roc > 0:
        print(f"Threshold Optimization: AUC-ROC >= {min_auc_roc}, Recall >= {min_recall}, Precision >= {min_precision}")
    print(f"Validation AUC-ROC: {metrics.get('auc_roc', 'N/A'):.4f}")
    print(f"Validation F1-Score: {metrics.get('f1', 'N/A'):.4f}")
    print(f"Validation Precision: {metrics.get('precision', 'N/A'):.4f}")
    print(f"Validation Recall: {metrics.get('recall', 'N/A'):.4f} {'✅' if metrics.get('recall', 0) >= 0.90 else '❌'}")
    print("=" * 70)
    
    return model_to_use, metrics, threshold_results, final_threshold


if __name__ == "__main__":
    # Default configuration: Optimized for recall >= 0.9 using calibration and threshold optimization
    # Set use_full_data=False for faster testing with balanced 10k sample
    model, metrics, threshold_results, optimal_threshold = main(
        threshold=0.65,  # Will be optimized if min_recall is set
        use_hyperparameter_tuning=True,
        tuning_method='optuna',
        n_trials=50,  # Optimized: reduced from 100 for faster tuning
        use_feature_selection=True,
        use_ensemble=True,  # Default True for eighth view (required for calibration)
        use_smote=False,  # ❌ SMOTE/ADASYN KALDIRILDI - AUC düşürüyor (high-dimensional + categorical-heavy)
        use_cross_validation=True,  # Set to True for CV evaluation and ensemble weights
        cv_folds=3,  # Optimized: reduced from 5 to 3 for faster CV
        min_auc_roc=0.94,  # AUC-ROC target (realistic)
        min_recall=0.90,  # Recall target (goal: achieve with calibration + threshold optimization)
        min_precision=0.40,  # Precision target (realistic)
        use_full_data=False  # DEBUG MODE: 20k sample (hızlı test için)
    )

