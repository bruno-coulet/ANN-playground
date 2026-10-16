"""
Module : cleaning_utils.py
Description : Utility functions for tabular data diagnostic, feature typing,
              data cleaning, and dataset exporting using Pandas and Scikit-Learn.

Functions :
    - missing_summary(df: pd.DataFrame) -> pd.DataFrame
    - special_columns(df: pd.DataFrame, max_modalities: int) -> dict[str, list[str]]
    - empty_columns(df: pd.DataFrame) -> list[str]
    - missing_like_columns(df: pd.DataFrame) -> list[str]
    - duplicate_rows(df: pd.DataFrame, subset: list[str] | None, keep: str) -> pd.DataFrame
    - count_duplicates(df: pd.DataFrame, subset: list[str] | None) -> int
    - drop_columns(df: pd.DataFrame, cols: list[str], dropped: list[str] | None) -> pd.DataFrame
    - drop_one_column(df: pd.DataFrame, col: str, dropped: list[str] | None) -> pd.DataFrame
    - convert_bool_to_uint8(df: pd.DataFrame, cols: list[str], keep_na: bool) -> pd.DataFrame
    - lower_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame
    - normalize_text_features(df: pd.DataFrame) -> pd.DataFrame
    - fit_transform_clean(X_train: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]
    - transform_clean(X_test: pd.DataFrame, stats: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame
    - export_train_test_parquet(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series | np.ndarray, y_test: pd.Series | np.ndarray, output_dir: Path | str, target_name: str, transform_y: str | None, drop_cols: list[str] | None) -> tuple[Path, Path]
    - preprocess_data(X_train: pd.DataFrame, X_test: pd.DataFrame | None, numeric_columns: list[str] | None, categorical_columns: list[str] | None, scaler: BaseEstimator) -> tuple[np.ndarray, np.ndarray | None, ColumnTransformer]
"""

import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute data types, missing value counts, and fill rates per column.

    Args:
        df: Input DataFrame to inspect.

    Returns:
        Summary DataFrame sorted by missing values in descending order.
    """
    return pd.DataFrame(
        {
            "type": df.dtypes,
            "missing_count": df.isna().sum(),
            "fill_rate_%": (df.count() / len(df) * 100).round(2),
        }
    ).sort_values(by="missing_count", ascending=False)


def special_columns(df: pd.DataFrame, max_modalities: int = 20) -> dict[str, list[str]]:
    """Identify and categorize DataFrame columns according to structural features.

    Args:
        df: Input DataFrame to analyze.
        max_modalities: Maximum unique count to define low vs high cardinality.

    Returns:
        Dictionary mapping structural categories to lists of column names.
    """
    string_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    return {
        "empty": df.columns[df.isna().all()].tolist(),
        "constant": df.columns[df.nunique(dropna=False) <= 1].tolist(),
        "numeric": df.select_dtypes(include=["number"]).columns.tolist(),
        "categorical": string_cols,
        "high_cardinality": [
            c for c in string_cols if df[c].nunique(dropna=True) > max_modalities
        ],
        "boolean": [
            col
            for col in df.columns
            if set(df[col].dropna().unique()).issubset({True, False})
        ],
    }


def empty_columns(df: pd.DataFrame) -> list[str]:
    """Retrieve column names that contain only null or missing values.

    Args:
        df: Input DataFrame.

    Returns:
        List of entirely empty column names.
    """
    return df.columns[df.isna().all()].tolist()


def missing_like_columns(df: pd.DataFrame) -> list[str]:
    """Detect columns holding pseudo-missing string representations.

    Args:
        df: Input DataFrame.

    Returns:
        List of column names with missing-like placeholders.
    """
    missing_vals = {np.nan, None, "", "na", "NA", "null", "NULL"}
    return [col for col in df.columns if df[col].isin(missing_vals).any()]


def duplicate_rows(
    df: pd.DataFrame, subset: list[str] | None = None, keep: str = "first"
) -> pd.DataFrame:
    """Filter duplicate rows based on subset keys.

    Args:
        df: Input DataFrame.
        subset: Column keys to evaluate duplicates on.
        keep: Strategy for marking duplicates ('first', 'last', False).

    Returns:
        DataFrame containing only duplicated entries.
    """
    return df[df.duplicated(subset=subset, keep=keep)]


def count_duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> int:
    """Count total duplicated observations in a DataFrame.

    Args:
        df: Input DataFrame.
        subset: Column keys to consider.

    Returns:
        Total count of duplicated records.
    """
    return int(df.duplicated(subset=subset, keep="first").sum())


def drop_columns(
    df: pd.DataFrame, cols: list[str], dropped: list[str] | None = None
) -> pd.DataFrame:
    """Drop specified columns and record dropped feature names.

    Args:
        df: Source DataFrame.
        cols: Target columns to delete.
        dropped: Optional tracking list to append dropped names.

    Returns:
        Cleaned DataFrame without target columns.
    """
    existing_cols = [c for c in cols if c in df.columns]
    if dropped is not None:
        dropped.extend(existing_cols)
    return df.drop(columns=existing_cols)


def drop_one_column(
    df: pd.DataFrame, col: str, dropped: list[str] | None = None
) -> pd.DataFrame:
    """Drop a single column with optional audit tracking.

    Args:
        df: Source DataFrame.
        col: Target column to delete.
        dropped: Optional tracking list.

    Returns:
        DataFrame without the specified column.
    """
    if col in df.columns and dropped is not None:
        dropped.append(col)
    return df.drop(columns=[col], errors="ignore")


def convert_bool_to_uint8(
    df: pd.DataFrame, cols: list[str], keep_na: bool = True
) -> pd.DataFrame:
    """Cast boolean columns into unsigned 8-bit integers.

    Args:
        df: Source DataFrame.
        cols: Target boolean columns.
        keep_na: If True, preserves nulls with pandas nullable boolean type.

    Returns:
        DataFrame with converted column types.
    """
    df_out = df.copy()
    for col in cols:
        if col in df_out.columns:
            if keep_na:
                df_out[col] = df_out[col].astype("boolean").astype("UInt8")
            else:
                df_out[col] = (
                    df_out[col].astype("boolean").fillna(False).astype("UInt8")
                )
    return df_out


def lower_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Transform text column contents into lower-case.

    Args:
        df: Source DataFrame.
        cols: Target text columns.

    Returns:
        Copy of DataFrame with lower-cased text columns.
    """
    df_out = df.copy()
    existing_cols = [c for c in cols if c in df_out.columns]
    string_cols = [c for c in existing_cols if pd.api.types.is_string_dtype(df_out[c])]
    for col in string_cols:
        df_out[col] = df_out[col].str.lower()
    return df_out


def normalize_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize text representations by removing accents and leading/trailing whitespace.

    Args:
        df: Source DataFrame.

    Returns:
        Copy of DataFrame with normalized text features.
    """
    df_out = df.copy()
    string_cols = df_out.select_dtypes(include=["object", "string"]).columns
    for col in string_cols:
        df_out[col] = df_out[col].astype(str).str.lower().str.strip()
        df_out[col] = df_out[col].apply(
            lambda x: (
                "".join(
                    c
                    for c in unicodedata.normalize("NFD", str(x))
                    if unicodedata.category(c) != "Mn"
                )
                if pd.notna(x)
                else x
            )
        )
        df_out[col] = df_out[col].replace({"nan": np.nan, "none": np.nan})
    return df_out


def fit_transform_clean(
    X_train: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean the training feature set and save fitted parameters to avoid data leakage.

    Args:
        X_train: Training feature DataFrame.
        config: Configuration dictionary specifying thresholds, modes, and replace maps.

    Returns:
        Tuple containing the cleaned training DataFrame and fitted statistics.
    """
    X = X_train.copy()
    stats: dict[str, Any] = {}

    threshold = config.get("drop_na_threshold", 1.0)
    high_na = X.columns[X.isna().mean() > threshold].tolist()
    stats["cols_to_drop"] = high_na
    X = X.drop(columns=high_na)

    bin_cols = [c for c in config.get("binary_cols", []) if c in X.columns]
    stats["modes"] = {c: X[c].mode()[0] for c in bin_cols if not X[c].mode().empty}
    for c, val in stats["modes"].items():
        X[c] = X[c].fillna(val)

    for col, replace_map in config.get("replace_maps", {}).items():
        if col in X.columns:
            X[col] = X[col].replace(replace_map)

    num_cols = [c for c in config.get("numeric_median_cols", []) if c in X.columns]
    stats["medians"] = {}
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        stats["medians"][c] = X[c].median()
        X[c] = X[c].fillna(stats["medians"][c])

    return X, stats


def transform_clean(
    X_test: pd.DataFrame, stats: dict[str, Any], config: dict[str, Any]
) -> pd.DataFrame:
    """Apply fitted training transformations to evaluation sets.

    Args:
        X_test: Test or inference feature DataFrame.
        stats: Computed training statistics from fit_transform_clean.
        config: Transformation configuration dictionary.

    Returns:
        Cleaned evaluation DataFrame.
    """
    X = X_test.copy()

    X = X.drop(columns=stats.get("cols_to_drop", []), errors="ignore")

    for c, mode_val in stats.get("modes", {}).items():
        if c in X.columns:
            X[c] = X[c].fillna(mode_val)

    for col, replace_map in config.get("replace_maps", {}).items():
        if col in X.columns:
            X[col] = X[col].replace(replace_map)

    for c, median_val in stats.get("medians", {}).items():
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
            X[c] = X[c].fillna(median_val)

    return X


def export_train_test_parquet(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    y_test: pd.Series | np.ndarray,
    output_dir: Path | str = "data/processed_data",
    target_name: str = "target",
    transform_y: str | None = None,
    drop_cols: list[str] | None = None,
) -> tuple[Path, Path]:
    """Export prepared training and test sets as compressed Parquet files.

    Concatenates features and target into single tables matching the structure
    expected by ANN Lab ('train_prepared.parquet' and 'test_prepared.parquet').

    Args:
        X_train: Training features DataFrame.
        X_test: Test features DataFrame.
        y_train: Training target labels or continuous values.
        y_test: Test target labels or continuous values.
        output_dir: Directory where artifacts will be saved.
        target_name: Column name assigned to the target variable.
        transform_y: Optional transform ('log1p' supported).
        drop_cols: Columns to exclude from the export.

    Returns:
        Tuple of (train_filepath, test_filepath).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    X_tr = X_train.drop(columns=drop_cols or [], errors="ignore").reset_index(drop=True)
    X_te = X_test.drop(columns=drop_cols or [], errors="ignore").reset_index(drop=True)

    y_tr_arr = y_train.values if isinstance(y_train, pd.Series) else np.asarray(y_train)
    y_te_arr = y_test.values if isinstance(y_test, pd.Series) else np.asarray(y_test)

    if transform_y == "log1p":
        y_tr_vals = np.log1p(y_tr_arr)
        y_te_vals = np.log1p(y_te_arr)
    else:
        y_tr_vals = y_tr_arr
        y_te_vals = y_te_arr

    X_tr[target_name] = y_tr_vals
    X_te[target_name] = y_te_vals

    train_path = out_path / "train_prepared.parquet"
    test_path = out_path / "test_prepared.parquet"

    X_tr.to_parquet(train_path, index=False)
    X_te.to_parquet(test_path, index=False)

    return train_path, test_path


def preprocess_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame | None = None,
    numeric_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    scaler: BaseEstimator | None = None,
) -> tuple[np.ndarray, np.ndarray | None, ColumnTransformer]:
    """Fit a ColumnTransformer on train data and transform both sets without leakage.

    Args:
        X_train: Training feature DataFrame used to fit scalers and encoders.
        X_test: Optional test feature DataFrame to transform.
        numeric_columns: Columns to scale. Auto-detected if None.
        categorical_columns: Categorical columns to one-hot encode. Auto-detected if None.
        scaler: Scikit-learn scaler instance (default: RobustScaler).

    Returns:
        Tuple of (processed_X_train, processed_X_test, fitted_column_transformer).

    Raises:
        ValueError: If no valid numerical or categorical features are provided.
    """
    if scaler is None:
        scaler = RobustScaler()

    if numeric_columns is None:
        numeric_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()

    if categorical_columns is None:
        categorical_columns = X_train.select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()

    transformers = []
    if numeric_columns:
        transformers.append(("num", clone(scaler), numeric_columns))
    if categorical_columns:
        transformers.append(
            (
                "cat",
                OneHotEncoder(
                    drop="first", handle_unknown="ignore", sparse_output=False
                ),
                categorical_columns,
            )
        )

    if not transformers:
        raise ValueError(
            "No valid numeric or categorical columns found for transformation."
        )

    column_transformer = ColumnTransformer(transformers=transformers, remainder="drop")

    X_train_processed = column_transformer.fit_transform(X_train)
    X_test_processed = (
        column_transformer.transform(X_test) if X_test is not None else None
    )

    return X_train_processed, X_test_processed, column_transformer
