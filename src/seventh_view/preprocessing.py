"""Preprocessing: improved SMOTE sampling_strategy for better recall."""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try to import SMOTE
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("Warning: imbalanced-learn not available. Install with: pip install imbalanced-learn")


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Reduce memory usage by optimizing data types."""
    start_mem = df.memory_usage().sum() / 1024**2
    
    if verbose:
        print("\n" + "=" * 70)
        print("REDUCE MEMORY USAGE - DETAILED LOGGING")
        print("=" * 70)
    
    skipped_categorical = []
    skipped_special = []
    processed_numeric = []
    errors = []
    converted_to_category = []
    
    for col in df.columns:
        if col in ['TransactionID', 'isFraud']:
            skipped_special.append(col)
            continue
            
        col_type = df[col].dtype
        col_type_str = str(col_type)
        
        # CRITICAL: Check categorical FIRST using multiple methods
        is_categorical_by_dtype = col_type_str.startswith('category') or col_type_str == 'category'
        is_categorical_by_api = pd.api.types.is_categorical_dtype(df[col])
        has_cat_attribute = hasattr(df[col], 'cat')
        
        # Try to access .cat attribute to be absolutely sure
        is_categorical_actual = False
        try:
            if has_cat_attribute:
                _ = df[col].cat  # Try to access it
                is_categorical_actual = True
        except:
            pass
        
        is_categorical = is_categorical_by_dtype or is_categorical_by_api or is_categorical_actual
        
        # Check numeric (but only if NOT categorical)
        is_numeric = False
        if not is_categorical:
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
        
        if verbose:
            print(f"\nProcessing column: {col}")
            print(f"  dtype: {col_type_str}")
            print(f"  dtype.name: {col_type.name if hasattr(col_type, 'name') else 'N/A'}")
            print(f"  is_categorical_by_dtype: {is_categorical_by_dtype}")
            print(f"  is_categorical_by_api: {is_categorical_by_api}")
            print(f"  has_cat_attribute: {has_cat_attribute}")
            print(f"  is_categorical_actual: {is_categorical_actual}")
            print(f"  is_categorical (FINAL): {is_categorical}")
            print(f"  is_numeric_dtype: {is_numeric}")
            print(f"  col_type != object: {col_type != object}")
        
        # CRITICAL: Skip categorical columns FIRST - before ANY other operations
        if is_categorical:
            if verbose:
                print(f"  → SKIPPED (categorical - detected by: dtype={is_categorical_by_dtype}, api={is_categorical_by_api}, attr={has_cat_attribute})")
            skipped_categorical.append((col, col_type_str))
            continue
        
        # Only process numeric columns (skip object and category)
        # CRITICAL: Triple-check that it's NOT categorical and IS numeric before calling min/max
        # ADDITIONAL SAFETY: Check dtype.name explicitly to avoid any edge cases
        dtype_name = getattr(col_type, 'name', str(col_type))
        is_really_numeric = (
            col_type != object and 
            is_numeric and 
            not is_categorical and
            dtype_name not in ['category', 'object'] and
            not dtype_name.startswith('category')
        )
        
        if is_really_numeric:
            # FINAL SAFETY CHECK: Try to access .cat attribute - if it exists, skip BEFORE any min/max
            if hasattr(df[col], 'cat'):
                try:
                    _ = df[col].cat  # Try to access it - if this works, it's categorical
                    if verbose:
                        print(f"  → SKIPPED (has .cat attribute - is categorical)")
                    skipped_categorical.append((col, col_type_str))
                    continue
                except:
                    pass  # If accessing .cat fails, it's not categorical
            
            # CRITICAL: Before attempting min/max, do a final check by trying a safe operation
            # Try to check if column is actually categorical by attempting to access categories
            try:
                # This will fail if column is not categorical
                if hasattr(df[col], 'cat'):
                    _ = list(df[col].cat.categories)
                    if verbose:
                        print(f"  → SKIPPED (can access .cat.categories - is categorical)")
                    skipped_categorical.append((col, col_type_str))
                    continue
            except:
                pass  # If this fails, column is not categorical (or doesn't have .cat)
            
            # CRITICAL: Wrap the entire min/max operation in a try-except
            # This is the final safety net - if ANY error occurs, skip the column
            try:
                if verbose:
                    print(f"  → Processing numeric column (attempting min/max)...")
                    print(f"    Checking for NaN values: {df[col].isna().sum()} NaN values")
                
                # Use skipna=True to handle NaN values safely
                # But FIRST check if it's really numeric by trying a safe operation
                if df[col].isna().all():
                    if verbose:
                        print(f"    → SKIPPED (all values are NaN)")
                    continue
                
                # CRITICAL: Wrap min/max in try-except to catch ANY errors (especially categorical)
                # This is the most important safety check
                try:
                    c_min = df[col].min(skipna=True)
                    c_max = df[col].max(skipna=True)
                except Exception as e:
                    # If we get ANY error (especially categorical), skip this column
                    error_str = str(e).lower()
                    if any(keyword in error_str for keyword in ['categorical', 'ordered', 'category']):
                        if verbose:
                            print(f"    → ERROR (categorical detected): {str(e)}")
                            print(f"    → SKIPPED (is categorical despite checks)")
                        skipped_categorical.append((col, col_type_str))
                        continue
                    else:
                        # For other errors, log and skip
                        if verbose:
                            print(f"    → ERROR: {str(e)}")
                            print(f"    → SKIPPED (cannot optimize)")
                        errors.append((col, col_type_str, str(e)))
                        continue
                
                if verbose:
                    print(f"    min: {c_min}, max: {c_max}")
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                        if verbose:
                            print(f"    → Optimized to int8")
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                        if verbose:
                            print(f"    → Optimized to int16")
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                        if verbose:
                            print(f"    → Optimized to int32")
                processed_numeric.append(col)
            except (TypeError, ValueError) as e:
                # Skip columns that can't be optimized (e.g., unordered categorical)
                error_msg = str(e)
                if verbose:
                    print(f"  → ERROR: {error_msg}")
                    print(f"    → SKIPPED (cannot optimize)")
                errors.append((col, col_type_str, error_msg))
                continue
        else:
            if df[col].nunique() < 100:
                df[col] = df[col].astype('category')
                if verbose:
                    print(f"  → Converted to category (nunique < 100)")
                converted_to_category.append(col)
            else:
                if verbose:
                    print(f"  → SKIPPED (object type, nunique >= 100)")
    
    end_mem = df.memory_usage().sum() / 1024**2
    
    if verbose:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Skipped (categorical): {len(skipped_categorical)} columns")
        if skipped_categorical and len(skipped_categorical) <= 20:
            for col, dtype in skipped_categorical:
                print(f"  - {col} ({dtype})")
        print(f"Skipped (special): {len(skipped_special)} columns ({', '.join(skipped_special)})")
        print(f"Processed (numeric): {len(processed_numeric)} columns")
        if processed_numeric and len(processed_numeric) <= 20:
            print(f"  - {', '.join(processed_numeric)}")
        print(f"Errors: {len(errors)} columns")
        if errors:
            for col, dtype, error in errors:
                print(f"  - {col} ({dtype}): {error}")
        print(f"Converted to category: {len(converted_to_category)} columns")
        if converted_to_category and len(converted_to_category) <= 20:
            print(f"  - {', '.join(converted_to_category)}")
        print(f"\nMemory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB "
              f"({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")
        print("=" * 70)
    
    return df


def preprocess_sixth_view_data(df: pd.DataFrame, 
                               is_train: bool = True) -> pd.DataFrame:
    """
    Preprocess data for sixth view model with proper NaN handling for SMOTE.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    is_train : bool
        Whether this is training data
        
    Returns
    -------
    df : pd.DataFrame
        Preprocessed dataframe with NO NaN values (filled for SMOTE compatibility)
    """
    df = df.copy()
    
    print("Preprocessing data...")
    
    # === Transaction Features ===
    # Card features (card1-card6) - fill with -1 for missing
    for card_col in ['card1', 'card2', 'card3', 'card4', 'card5', 'card6']:
        if card_col in df.columns:
            df[f'{card_col}_isMissing'] = df[card_col].isna().astype(int)
            df[card_col] = df[card_col].fillna(-1)
    
    # Address features (addr1, addr2)
    for addr_col in ['addr1', 'addr2']:
        if addr_col in df.columns:
            df[f'{addr_col}_isMissing'] = df[addr_col].isna().astype(int)
            df[addr_col] = df[addr_col].fillna(-1)
    
    # Email features
    for email_col in ['P_emaildomain', 'R_emaildomain']:
        if email_col in df.columns:
            df[f'{email_col}_isMissing'] = df[email_col].isna().astype(int)
            df[email_col] = df[email_col].fillna('missing')
    
    # Transaction amount
    if 'TransactionAmt' in df.columns:
        if df['TransactionAmt'].isna().sum() > 0:
            df['TransactionAmt'] = df['TransactionAmt'].fillna(df['TransactionAmt'].median())
    
    # Transaction datetime
    if 'TransactionDT' in df.columns:
        if df['TransactionDT'].isna().sum() > 0:
            print("Warning: TransactionDT has missing values!")
            df['TransactionDT'] = df['TransactionDT'].fillna(df['TransactionDT'].median())
    
    # Product code
    if 'ProductCD' in df.columns:
        if df['ProductCD'].isna().sum() > 0:
            df['ProductCD'] = df['ProductCD'].fillna('missing')
    
    # IP/Distance features (dist1, dist2) - Categorical
    for dist_col in ['dist1', 'dist2']:
        if dist_col in df.columns:
            df[f'{dist_col}_isMissing'] = df[dist_col].isna().astype(int)
            # Categorical imputation: use mode or -1
            if df[dist_col].notna().sum() > 0:
                mode_values = df[dist_col].mode()
                fill_value = mode_values[0] if len(mode_values) > 0 else -1
            else:
                fill_value = -1
            df[dist_col] = df[dist_col].fillna(fill_value)
    
    # Card-related features (C1-C14) - Categorical imputation
    for c_col in [f'C{i}' for i in range(1, 15)]:
        if c_col in df.columns:
            df[f'{c_col}_isMissing'] = df[c_col].isna().astype(int)
            # Categorical imputation: use mode for categorical features
            if df[c_col].notna().sum() > 0:
                mode_values = df[c_col].mode()
                fill_value = mode_values[0] if len(mode_values) > 0 else -1
            else:
                fill_value = -1
            df[c_col] = df[c_col].fillna(fill_value)
            # Convert to categorical type
            df[c_col] = df[c_col].astype('category')
    
    # === Identity Features ===
    if 'DeviceType' in df.columns:
        df['DeviceType_isMissing'] = df['DeviceType'].isna().astype(int)
        df['DeviceType'] = df['DeviceType'].fillna('missing')
    
    if 'DeviceInfo' in df.columns:
        df['DeviceInfo_isMissing'] = df['DeviceInfo'].isna().astype(int)
        df['DeviceInfo'] = df['DeviceInfo'].fillna('missing')
    
    for id_col in ['id_28', 'id_29', 'id_30', 'id_31']:
        if id_col in df.columns:
            df[f'{id_col}_isMissing'] = df[id_col].isna().astype(int)
            # Categorical imputation: use mode or 'missing'
            if df[id_col].dtype in ['float64', 'int64', 'float32', 'int32', 'int16', 'int8']:
                # Use mode for categorical features
                if df[id_col].notna().sum() > 0:
                    mode_values = df[id_col].mode()
                    fill_value = mode_values[0] if len(mode_values) > 0 else -1
                else:
                    fill_value = -1
                df[id_col] = df[id_col].fillna(fill_value)
            else:
                df[id_col] = df[id_col].fillna('missing')
    
    # CRITICAL: Mark ALL features as categorical EXCEPT TransactionAmt and TransactionDT
    # According to requirements: "amount ve time özelliği hariç hepsi kategorik"
    print("Marking features as categorical (except TransactionAmt and TransactionDT)...")
    numeric_only_cols = ['TransactionAmt', 'TransactionDT']
    converted_to_categorical = []
    
    for col in df.columns:
        if col in numeric_only_cols or col in ['TransactionID', 'isFraud']:
            continue  # Keep these as numeric
        if col.endswith('_isMissing'):
            continue  # Keep missing indicators as numeric (int)
        
        # Skip if already categorical
        if pd.api.types.is_categorical_dtype(df[col]):
            continue
        
        # Skip if already object/string (will be converted to category later in reduce_memory_usage)
        if df[col].dtype == 'object':
            continue
        
        # Convert numeric columns (card1-card6, addr1-addr2, dist1-dist2, id_28-id_31, etc.) to categorical
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype('category')
            converted_to_categorical.append(col)
    
    if converted_to_categorical:
        print(f"  Converted {len(converted_to_categorical)} numeric columns to categorical")
        if len(converted_to_categorical) <= 20:
            print(f"  Columns: {', '.join(converted_to_categorical)}")
    
    # CRITICAL: Fill ALL remaining NaN values to ensure SMOTE compatibility
    # This is important - SMOTE cannot handle NaN values
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            fill_value = df[col].median() if df[col].notna().sum() > 0 else 0
            df[col] = df[col].fillna(fill_value)
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_cols:
        if df[col].isna().sum() > 0:
            # CRITICAL: For categorical columns, add 'missing' category first before fillna
            if pd.api.types.is_categorical_dtype(df[col]):
                if 'missing' not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories(['missing'])
            df[col] = df[col].fillna('missing')
    
    # Verify no NaN values remain
    remaining_nans = df.isna().sum().sum()
    if remaining_nans > 0:
        print(f"Warning: {remaining_nans} NaN values still remain after preprocessing!")
        # Force fill any remaining NaN
        df = df.fillna(0)
    
    # Reduce memory usage
    df = reduce_memory_usage(df, verbose=False)
    
    print(f"Preprocessing complete. Shape: {df.shape}")
    print(f"NaN values remaining: {df.isna().sum().sum()}")
    
    return df


def apply_smote(X: pd.DataFrame, 
                y: np.ndarray,
                sampling_strategy: float = 0.12,  # Increased from 0.1 to improve recall (fraud rate %9.09 → %12-15)
                k_neighbors: int = 5,
                random_state: int = 42,
                verbose: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance classes.
    ENHANCED: Properly handles NaN values by ensuring they're filled before SMOTE.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature dataframe (should have NO NaN values after preprocessing)
    y : np.ndarray
        Target array
    sampling_strategy : float
        Ratio of minority class to majority class after resampling (0.1 = 10% of majority)
    k_neighbors : int
        Number of nearest neighbors for SMOTE
    random_state : int
        Random seed
    verbose : bool
        Whether to print information
        
    Returns
    -------
    X_resampled : pd.DataFrame
        Resampled feature dataframe
    y_resampled : np.ndarray
        Resampled target array
    """
    if not SMOTE_AVAILABLE:
        if verbose:
            print("Warning: SMOTE not available. Skipping resampling.")
        return X, y
    
    # CRITICAL: Check for NaN values before SMOTE
    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        if verbose:
            print(f"Warning: {nan_count} NaN values found. Filling with median/0...")
        # Fill numeric columns with median
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if X[col].isna().sum() > 0:
                fill_value = X[col].median() if X[col].notna().sum() > 0 else 0
                X[col] = X[col].fillna(fill_value)
        
        # Fill categorical columns with 'missing'
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if X[col].isna().sum() > 0:
                X[col] = X[col].fillna('missing')
        
        # Final check
        if X.isna().sum().sum() > 0:
            X = X.fillna(0)
    
    # Separate numeric and categorical features
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]
    
    if len(categorical_cols) > 0:
        if verbose:
            print(f"Converting {len(categorical_cols)} categorical features to numeric for SMOTE...")
        
        # Convert categorical to numeric for SMOTE
        X_for_smote = X.copy()
        label_encoders = {}
        
        for col in categorical_cols:
            if X_for_smote[col].dtype.name == 'category':
                X_for_smote[col] = X_for_smote[col].cat.codes
            else:
                # Label encode
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                # Handle NaN values in string columns
                col_values = X_for_smote[col].astype(str)
                X_for_smote[col] = le.fit_transform(col_values)
                label_encoders[col] = le
    else:
        X_for_smote = X.copy()
    
    # Final NaN check before SMOTE
    if X_for_smote.isna().sum().sum() > 0:
        if verbose:
            print("Warning: NaN values still present after conversion. Filling with 0...")
        X_for_smote = X_for_smote.fillna(0)
    
    # Apply SMOTE
    if verbose:
        print(f"Applying SMOTE...")
        print(f"  Before: Fraud={y.sum()}, Non-fraud={len(y) - y.sum()}, Ratio={y.sum() / len(y):.4f}")
    
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=min(k_neighbors, y.sum() - 1),  # k_neighbors must be < n_minority
        random_state=random_state
        # Note: n_jobs parameter is not available in SMOTE
    )
    
    try:
        X_resampled, y_resampled = smote.fit_resample(X_for_smote, y)
        
        # Convert back to DataFrame with original column names
        X_resampled = pd.DataFrame(X_resampled, columns=X_for_smote.columns)
        
        # Restore categorical features if needed
        if len(categorical_cols) > 0:
            # Note: SMOTE generates synthetic samples, so categorical values may not be exact
            # Round to nearest integer category
            for col in categorical_cols:
                X_resampled[col] = X_resampled[col].round().astype(int)
                # Clip to valid range
                if col in X.columns:
                    if X[col].dtype.name == 'category':
                        max_cat = len(X[col].cat.categories) - 1
                        X_resampled[col] = X_resampled[col].clip(0, max_cat)
                    elif col in label_encoders:
                        # Clip to valid label encoder range
                        max_label = len(label_encoders[col].classes_) - 1
                        X_resampled[col] = X_resampled[col].clip(0, max_label)
        
        if verbose:
            print(f"  After: Fraud={y_resampled.sum()}, Non-fraud={len(y_resampled) - y_resampled.sum()}, Ratio={y_resampled.sum() / len(y_resampled):.4f}")
            print(f"  New samples: {len(y_resampled) - len(y)}")
        
        return X_resampled, y_resampled
    
    except Exception as e:
        if verbose:
            print(f"Error applying SMOTE: {e}")
            print("Returning original data without resampling.")
        return X, y

