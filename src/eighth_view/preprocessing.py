"""
Preprocessing utilities: type conversion, memory reduction, SMOTE.
"""

import pandas as pd
import numpy as np
from typing import Tuple
import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------------
# SMOTE availability
# -------------------------------------------------------
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


# -------------------------------------------------------
# FEATURE TYPE CONVERSIONS (PRODUCTION-SAFE)
# -------------------------------------------------------
def convert_feature_types(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Converts feature types: frequency features (nunique < 50), C1-C14, cyclical time features to categorical.
    
    Args:
        df: Input dataframe.
        verbose: Print conversion details.
    
    Returns:
        DataFrame with converted types.
    """
    df = df.copy()
    converted_count = 0
    
    if verbose:
        print("Converting feature types (Faz 1: Type Conversions)...")
    
    # Faz 1.1: Convert frequency features to categorical (nunique < 50)
    freq_patterns = ['_count', '_freq']
    for col in df.columns:
        if any(pattern in col for pattern in freq_patterns):
            if pd.api.types.is_numeric_dtype(df[col]):
                nunique = df[col].nunique()
                if nunique < 50 and nunique < len(df) * 0.1:
                    # Convert to categorical (ordinal)
                    df[col] = df[col].astype('category')
                    converted_count += 1
                    if verbose:
                        print(f"  ✓ {col}: numeric → categorical (nunique={nunique})")
    
    # Faz 1.2: Convert C1-C14 to categorical
    c_cols = [f'C{i}' for i in range(1, 15)]
    for col in c_cols:
        if col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].astype('category')
                converted_count += 1
                if verbose:
                    print(f"  ✓ {col}: numeric → categorical")
    
    # Faz 1.3: Convert cyclical time features to categorical (ordinal)
    cyclical_time_features = ['day_of_week', 'hour']
    for col in cyclical_time_features:
        if col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].astype('category')
                converted_count += 1
                if verbose:
                    print(f"  ✓ {col}: numeric → categorical (ordinal)")
    
    if verbose:
        print(f"  Total features converted: {converted_count}")
    
    return df


# -------------------------------------------------------
# MEMORY REDUCTION (SAFE)
# -------------------------------------------------------
def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        dtype = df[col].dtype

        # 🚫 NEVER touch categorical or object
        if (
            pd.api.types.is_categorical_dtype(dtype)
            or dtype == "object"
            or col in ["TransactionID", "isFraud"]
        ):
            continue

        # ✅ ONLY pure numeric
        if pd.api.types.is_integer_dtype(dtype):
            c_min, c_max = df[col].min(), df[col].max()
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)

        elif pd.api.types.is_float_dtype(dtype):
            df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"Memory: {start_mem:.2f} → {end_mem:.2f} MB "
              f"({100 * (start_mem - end_mem) / start_mem:.1f}% ↓)")

    return df


# -------------------------------------------------------
# MAIN PREPROCESS
# -------------------------------------------------------
def preprocess_sixth_view_data(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Preprocess data for sixth view model with proper NaN handling for SMOTE.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    is_train : bool
        Whether this is training data (affects verbose output and some checks)
        
    Returns
    -------
    df : pd.DataFrame
        Preprocessed dataframe with NO NaN values (filled for SMOTE compatibility)
    """
    df = df.copy()
    
    if is_train:
        print("Preprocessing training data...")
    else:
        print("Preprocessing test data...")

    # =============================
    # Numeric columns
    # =============================
    for col in ["TransactionAmt", "TransactionDT"]:
        if col in df.columns:
            if df[col].isna().sum() > 0:
                fill_value = df[col].median() if df[col].notna().sum() > 0 else 0
                df[col] = df[col].fillna(fill_value)
                if is_train and df[col].isna().sum() > 0:
                    print(f"Warning: {col} still has missing values after median fill!")

    # =============================
    # Categorical groups
    # =============================
    categorical_cols = (
        [f"card{i}" for i in range(1, 7)] +
        ["addr1", "addr2", "ProductCD", "P_emaildomain", "R_emaildomain",
         "DeviceType", "DeviceInfo"] +
        [f"C{i}" for i in range(1, 15)] +
        ["dist1", "dist2"] +
        ["id_28", "id_29", "id_30", "id_31"]
    )

    for col in categorical_cols:
        if col in df.columns:
            df[f"{col}_isMissing"] = df[col].isna().astype(np.int8)
            df[col] = df[col].fillna("missing").astype("category")

    # =============================
    # FINAL NaN GUARD (SMOTE SAFE)
    # =============================
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].isna().sum() > 0:
                fill_value = df[col].median() if df[col].notna().sum() > 0 else 0
                df[col] = df[col].fillna(fill_value)
        else:
            if df[col].isna().sum() > 0:
                df[col] = df[col].fillna("missing")

    # Verify no NaN values remain (especially important for training)
    remaining_nans = df.isna().sum().sum()
    if remaining_nans > 0:
        if is_train:
            print(f"Warning: {remaining_nans} NaN values still remain after preprocessing!")
        # Force fill any remaining NaN
        df = df.fillna(0)
        if is_train:
            print("Force-filled remaining NaN values with 0/missing")

    # =============================
    # FEATURE TYPE CONVERSIONS (FAZ 1)
    # =============================
    df = convert_feature_types(df, verbose=is_train)
    
    df = reduce_memory_usage(df, verbose=is_train)
    
    if is_train:
        print(f"Preprocessing complete. Shape: {df.shape}")
        print(f"NaN values remaining: {df.isna().sum().sum()}")
    
    return df


# -------------------------------------------------------
# SMOTE (SAFE VERSION)
# -------------------------------------------------------
def apply_smote(
    X: pd.DataFrame,
    y: np.ndarray,
    sampling_strategy: float = 0.12,
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[pd.DataFrame, np.ndarray]:

    if not SMOTE_AVAILABLE:
        if verbose:
            print("SMOTE not available – skipping")
        return X, y

    X = X.copy()

    if verbose:
        print(f"  Converting categorical features to numeric for SMOTE...")
        print(f"  Original shape: {X.shape}")
        print(f"  Memory usage: {X.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # 🔒 Convert categorical → numeric (codes)
    # Use more memory-efficient conversion for high cardinality features
    for col in X.columns:
        if pd.api.types.is_categorical_dtype(X[col]):
            # For high cardinality features, use cat.codes but ensure int32 to save memory
            codes = X[col].cat.codes
            # Replace -1 (NaN) with 0 for SMOTE compatibility
            codes = codes.replace(-1, 0)
            # Convert to int32 to save memory (instead of int64)
            X[col] = codes.astype(np.int32)

    # Fill any remaining NaN values
    X = X.fillna(0)
    
    # Convert all columns to numeric types to reduce memory
    for col in X.columns:
        if X[col].dtype == 'object':
            # Convert object to numeric if possible
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype(np.float32)
        elif X[col].dtype == 'int64':
            # Downcast int64 to int32 to save memory
            X[col] = X[col].astype(np.int32)
        elif X[col].dtype == 'float64':
            # Downcast float64 to float32 to save memory
            X[col] = X[col].astype(np.float32)

    if verbose:
        print(f"  After conversion shape: {X.shape}")
        print(f"  Memory usage: {X.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    minority_count = int(y.sum())
    if minority_count < 2:
        raise ValueError("Not enough minority samples for SMOTE")

    # Optimize SMOTE parameters for large feature space
    # For high-dimensional data (483 features), use k_neighbors=1 to prevent memory issues
    # k_neighbors=1 is safer for high-dimensional data and reduces memory usage significantly
    n_features = X.shape[1]
    
    # Determine k_neighbors based on feature count and minority samples
    if n_features > 400:
        # Very high dimensional: use k_neighbors=1
        k_neighbors = 1
        if verbose:
            print(f"  ⚠ High-dimensional data ({n_features} features), using k_neighbors=1 for memory safety")
    elif n_features > 200:
        # High dimensional: use k_neighbors=2
        k_neighbors = min(2, minority_count - 1)
        if verbose:
            print(f"  ⚠ High-dimensional data ({n_features} features), using k_neighbors={k_neighbors}")
    else:
        # Normal dimensional: use k_neighbors=3
        k_neighbors = min(3, minority_count - 1)
    
    k_neighbors = max(1, min(k_neighbors, minority_count - 1))  # Ensure valid range
    
    if verbose:
        print(f"  SMOTE parameters: k_neighbors={k_neighbors}, sampling_strategy={sampling_strategy}")
        print(f"  Feature count: {n_features}, Minority samples: {minority_count}")

    if verbose:
        print(f"Before SMOTE: ratio={y.mean():.4f}, samples={len(y)}")

    # Try SMOTE with error handling for segmentation faults
    try:
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=random_state
        )
        X_res, y_res = smote.fit_resample(X, y)
    except (MemoryError, ValueError) as e:
        if verbose:
            print(f"  ⚠ Error during SMOTE: {e}")
            print(f"  Trying with k_neighbors=1 (safest for high-dimensional data)...")
        # Try with k_neighbors=1 as last resort (safest for high-dimensional data)
        try:
            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=1,
                random_state=random_state
            )
            X_res, y_res = smote.fit_resample(X, y)
        except Exception as e2:
            if verbose:
                print(f"  ❌ SMOTE failed even with k_neighbors=1: {e2}")
                print(f"  ⚠ Skipping SMOTE, returning original data")
            # If SMOTE still fails, return original data
            return X, y

    if verbose:
        print(f"After SMOTE: ratio={y_res.mean():.4f}, samples={len(y_res)}")

    return (
        pd.DataFrame(X_res, columns=X.columns),
        y_res
    )
