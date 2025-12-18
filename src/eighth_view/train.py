"""
Training script for eighth view model.
Extends seventh view with:
1. Calibration (Isotonic) - Ensemble output calibration with validation-based fit
2. Percentile Threshold Sweep - %1 → %5 range, recall target ≥ 0.90
3. Cost-Based Threshold - FN/FP cost scenarios (10x, 20x, 50x), expected loss plots

Goal: Achieve Recall > 0.90 without changing the model, using calibration and threshold optimization.
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

from src.eighth_view.data_loader import load_sixth_view_data
from src.eighth_view.preprocessing import preprocess_sixth_view_data, apply_smote
from src.eighth_view.feature_engineering import create_all_sixth_view_features
from src.eighth_view.feature_selection import comprehensive_feature_selection
from src.eighth_view.models import train_seventh_view_model, train_ensemble_model
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


def main(threshold: float = 0.65,
         use_hyperparameter_tuning: bool = True,
         tuning_method: str = 'optuna',
         n_trials: int = 50,  # Optimized: reduced from 100 for faster tuning
         use_feature_selection: bool = True,
         use_ensemble: bool = True,  # Default True for seventh view
         use_smote: bool = True,  # SMOTE for class imbalance (FIXED)
         use_cross_validation: bool = True,  # Cross-validation enabled
         cv_folds: int = 3,  # Optimized: reduced from 5 to 3 for faster CV
         min_auc_roc: float = 0.94,  # Realistic: AUC-ROC target (0.94-0.945)
         min_recall: float = 0.85,  # Realistic: Recall target (0.85-0.90)
         min_precision: float = 0.40):  # Realistic: Precision target (0.40-0.50)
    """
    Main training pipeline for seventh view model.
    Realistic targets: AUC-ROC >= 0.94, Recall >= 0.85, Precision >= 0.40.
    
    Parameters
    ----------
    threshold : float
        Classification threshold (will be optimized if min_recall is set)
    use_hyperparameter_tuning : bool
        Whether to use hyperparameter tuning (Optuna/Hyperopt)
    tuning_method : str
        'optuna' or 'hyperopt'
    n_trials : int
        Number of trials for hyperparameter tuning
    use_feature_selection : bool
        Whether to use feature selection
    use_ensemble : bool
        Whether to use ensemble methods (LightGBM + XGBoost + CatBoost)
    min_recall : float
        Minimum required recall (default: 0.9). If set, threshold will be optimized.
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
        print(f"Cross-validation: ENABLED ({cv_folds}-fold)")
    if min_recall > 0 or min_precision > 0 or min_auc_roc > 0:
        print(f"Threshold optimization: AUC-ROC >= {min_auc_roc}, Recall >= {min_recall}, Precision >= {min_precision}")
    print("=" * 70)
    
    # 1. Load data
    print("\n[1/11] Loading data...")
    train_df, test_df = load_sixth_view_data(sample_size=None)  # Use all data
    
    # 2. Preprocessing
    print("\n[2/11] Preprocessing data...")
    train_df = preprocess_sixth_view_data(train_df, is_train=True)
    test_df = preprocess_sixth_view_data(test_df, is_train=False)
    
    # 3. Feature engineering
    print("\n[3/11] Creating features...")
    train_df = create_all_sixth_view_features(train_df, is_train=True)
    test_df = create_all_sixth_view_features(test_df, is_train=False)
    
    # 4. Prepare features and target
    print("\n[4/11] Preparing features...")
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
    print(f"Total features: {len(feature_cols)}")
    
    # 5. Apply SMOTE (if enabled)
    if use_smote:
        print("\n[5/11] Applying SMOTE for class imbalance...")
        X_train_split, y_train_split = apply_smote(
            X_train_split, y_train_split,
            sampling_strategy=0.12,  # 12% of majority class (increased from 0.1 to improve recall)
            k_neighbors=5,
            random_state=42,
            verbose=True
        )
        print(f"After SMOTE - Training set: {X_train_split.shape}")
        print(f"Fraud rate after SMOTE: {y_train_split.mean():.4f}")
    
    # 6. Feature selection
    selected_features = feature_cols
    if use_feature_selection:
        print("\n[6/11] Feature selection...")
        # Train a simple model for feature importance (faster for feature selection)
        from src.eighth_view.models import SeventhViewLightGBM
        temp_model = SeventhViewLightGBM(n_estimators=200, learning_rate=0.1, verbose=-1)
        temp_model.fit(X_train_split, y_train_split,
                      categorical_features=categorical_features,
                      eval_set=(X_val, y_val),
                      early_stopping_rounds=30,
                      verbose=0)
        
        # Don't pass model to comprehensive_feature_selection - let it train on filtered features
        # This avoids dimension mismatch issues
        # Optimized thresholds: keep more features for better model capacity
        X_train_selected, selected_features = comprehensive_feature_selection(
            X_train_split, pd.Series(y_train_split),
            model=None,  # Let it train on filtered features to avoid dimension mismatch
            variance_threshold=0.005,  # Lower threshold to keep more features
            correlation_threshold=0.98,  # Higher threshold to keep more correlated features
            importance_threshold=0.005,  # Lower threshold to keep more features
            verbose=True
        )
        
        X_val_selected = X_val[selected_features]
        X_train_split = X_train_selected
        X_val = X_val_selected
        
        # Update categorical features
        categorical_features = [f for f in categorical_features if f in selected_features]
        print(f"Selected features: {len(selected_features)}")
        print(f"Selected categorical features: {len(categorical_features)}")
    else:
        print("\n[6/11] Feature selection: SKIPPED")
    
    # 7. Train model (with optional ensemble and hyperparameter tuning)
    print("\n[7/11] Training model...")
    if use_ensemble:
        model = train_ensemble_model(
            X_train_split, y_train_split,
            X_val=X_val, y_val=y_val,
            categorical_features=categorical_features,
            use_lgb=True,
            use_xgb=True,
            use_cat=True,
            threshold=threshold,
            use_hyperparameter_tuning=use_hyperparameter_tuning,
            tuning_method=tuning_method,
            n_trials=n_trials,
            use_cv_for_weights=use_cross_validation,  # Use CV for weights if CV is enabled
            cv_folds=cv_folds
        )
    else:
        model = train_seventh_view_model(
            X_train_split, y_train_split,
            X_val=X_val, y_val=y_val,
            categorical_features=categorical_features,
            model_params={
                'n_estimators': 2000,  # Keep same for 1 hour training
                'learning_rate': 0.02,
                'max_depth': 11,  # Reduced for overfitting control (was 15)
                'num_leaves': 200,  # Reduced for overfitting control (was 255)
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.5,  # Increased for overfitting control (was 0.1)
                'reg_lambda': 0.5,  # Increased for overfitting control (was 0.1)
                'random_state': 42
            },
            threshold=threshold,
            use_hyperparameter_tuning=use_hyperparameter_tuning,
            tuning_method=tuning_method,
            n_trials=n_trials,
            use_cv_for_tuning=use_cross_validation,  # Use CV for tuning if CV is enabled
            cv_folds=cv_folds
        )
    
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
    y_val_pred_proba = model.predict_proba(X_val)
    
    # Evaluate at multiple thresholds
    print("\nEvaluating at multiple thresholds...")
    threshold_results = evaluate_multiple_thresholds(
        y_val, y_val_pred_proba,
        thresholds=[0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
        verbose=True
    )
    
    # 10. Calibrate ensemble output FIRST (EIGHTH VIEW NEW FEATURE)
    # IMPORTANT: Calibration must be done BEFORE threshold optimization
    print("\n[10/13] Calibrating ensemble output...")
    calibrated_model = None
    calibration_metrics = None
    
    if use_ensemble:
        try:
            calibrated_model, calibration_metrics = calibrate_ensemble_output(
                model, X_val, y_val, method='isotonic', verbose=True
            )
            
            # Get calibrated predictions
            y_val_pred_proba_calibrated = calibrated_model.predict_proba(X_val)
            
            # Plot calibration curve
            try:
                plot_calibration_curve(
                    y_val, y_val_pred_proba, y_val_pred_proba_calibrated,
                    title="Eighth View Model - Calibration Curve"
                )
            except Exception as e:
                print(f"Could not plot calibration curve: {e}")
            
            # Use calibrated model for further analysis
            model_to_use = calibrated_model
            y_val_pred_proba_final = y_val_pred_proba_calibrated
            print("✅ Using calibrated model for threshold optimization")
        except Exception as e:
            print(f"⚠️  Calibration failed: {e}")
            print("Using uncalibrated model")
            model_to_use = model
            y_val_pred_proba_final = y_val_pred_proba
    else:
        print("Calibration skipped (ensemble required)")
        model_to_use = model
        y_val_pred_proba_final = y_val_pred_proba
    
    # 10.5. Find optimal threshold AFTER calibration (STRATEGY 1: Re-Optimization)
    print("\n[10.5/13] Finding optimal threshold (AFTER calibration)...")
    optimal_threshold = None
    optimal_metrics = None
    
    if min_recall > 0 or min_precision > 0 or min_auc_roc > 0:
        # Find threshold with multi-objective optimization USING CALIBRATED PROBABILITIES
        print(f"Finding threshold with multi-objective optimization (CALIBRATED probabilities)...")
        print(f"  Targets: AUC-ROC >= {min_auc_roc}, Recall >= {min_recall}, Precision >= {min_precision}")
        optimal_threshold, optimal_metrics = find_threshold_multi_objective_optimized(
            y_val, y_val_pred_proba_final,  # ⭐ CALIBRATED probabilities kullanılıyor
            min_auc_roc=min_auc_roc,
            min_recall=min_recall,
            min_precision=min_precision,
            threshold_range=(0.2, 0.7),  # Realistic range for balanced precision-recall
            n_thresholds=100,  # Reduced for faster optimization
            verbose=True
        )
        
        if optimal_threshold is not None and optimal_metrics:
            print(f"\n✅ Found optimal threshold (calibrated): {optimal_threshold:.4f}")
            print(f"  AUC-ROC: {optimal_metrics.get('auc_roc', 0):.4f}")
            print(f"  Precision: {optimal_metrics['precision']:.4f}")
            print(f"  Recall: {optimal_metrics['recall']:.4f}")
            print(f"  F1-Score: {optimal_metrics['f1']:.4f}")
        else:
            print("⚠️  Warning: Could not find threshold meeting all requirements with calibrated probabilities")
            # Fallback: Find F1-optimal threshold with calibrated probabilities
            optimal_threshold = find_optimal_threshold(
                y_val, y_val_pred_proba_final,  # ⭐ CALIBRATED probabilities
                metric='f1',
                threshold_range=(0.3, 0.7),  # Adjusted range for calibrated probabilities
                n_thresholds=50
            )
            print(f"Using F1-optimal threshold (calibrated): {optimal_threshold:.4f}")
    else:
        # Find optimal threshold (F1-score) with calibrated probabilities
        print("Finding optimal threshold (F1-score) with calibrated probabilities...")
        optimal_threshold = find_optimal_threshold(
            y_val, y_val_pred_proba_final,  # ⭐ CALIBRATED probabilities
            metric='f1',
            threshold_range=(0.3, 0.7),  # Adjusted range for calibrated probabilities
            n_thresholds=50
        )
        print(f"Optimal threshold (F1, calibrated): {optimal_threshold:.4f}")
    
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
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importance_') and model.feature_importance_ is not None:
        print("\nTop 20 Most Important Features:")
        print(model.feature_importance_.head(20))
    elif hasattr(model, 'lgb_model') and model.lgb_model is not None:
        if hasattr(model.lgb_model, 'feature_importance_') and model.lgb_model.feature_importance_ is not None:
            print("\nTop 20 Most Important Features (LightGBM):")
            print(model.lgb_model.feature_importance_.head(20))
    
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
    
    # Check if all selected features exist in test set
    missing_features = [f for f in selected_features if f not in test_df.columns]
    if missing_features:
        print(f"Warning: {len(missing_features)} features missing in test set: {missing_features[:10]}")
        # Create X_test with all selected features, filling missing ones with 0
        X_test = pd.DataFrame(index=test_df.index)
        for feat in selected_features:
            if feat in test_df.columns:
                X_test[feat] = test_df[feat]
            else:
                # Fill missing features with 0 (especially UID fraud_rate features)
                X_test[feat] = 0
                print(f"  Filling missing feature '{feat}' with 0")
        # Reorder to match training order
        X_test = X_test[selected_features]
        print(f"Using {len(selected_features)} features for test predictions (filled {len(missing_features)} missing features with 0)")
    else:
        X_test = test_df[selected_features]
    
    test_pred_proba = model.predict_proba(X_test)
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
    model, metrics, threshold_results, optimal_threshold = main(
        threshold=0.65,  # Will be optimized if min_recall is set
        use_hyperparameter_tuning=True,
        tuning_method='optuna',
        n_trials=50,  # Optimized: reduced from 100 for faster tuning
        use_feature_selection=True,
        use_ensemble=True,  # Default True for eighth view (required for calibration)
        use_smote=True,  # SMOTE for class imbalance (FIXED: proper NaN handling)
        use_cross_validation=True,  # Set to True for CV evaluation and ensemble weights
        cv_folds=3,  # Optimized: reduced from 5 to 3 for faster CV
        min_auc_roc=0.94,  # AUC-ROC target (realistic)
        min_recall=0.90,  # Recall target (goal: achieve with calibration + threshold optimization)
        min_precision=0.40  # Precision target (realistic)
    )

