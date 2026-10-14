"""
Module : cleaning_utils.py
Description : Fonctions utilitaires pour le diagnostic, le typage et le nettoyage
              de données tabulaires avec Pandas.

Functions :
    - missing_summary(df) -> pd.DataFrame
    - special_columns(df, max_modalities) -> dict[str, list[str]]
    - empty_columns(df) -> list[str]
    - missing_like_columns(df) -> list[str]
    - duplicate_rows(df, subset, keep) -> pd.DataFrame
    - count_duplicates(df, subset) -> int
    - drop_columns(df, cols, dropped) -> pd.DataFrame
    - drop_one_column(df, col, dropped) -> pd.DataFrame
    - convert_bool_to_uint8(df, cols, keep_na) -> pd.DataFrame
    - lower_columns(df, cols) -> pd.DataFrame
    - normalize_text_features(df) -> pd.DataFrame
    - fit_transform_clean(X_train, config) -> tuple[pd.DataFrame, dict[str, Any]]
    - transform_clean(X_test, stats, config) -> pd.DataFrame
    - export_train_test_feather(X_train, X_test, y_train, y_test, output_dir, target_name, transform_y, drop_cols) -> None
    - preprocess_data(X_train, X_test, numeric_columns, categorical_columns) -> tuple[np.ndarray, np.ndarray | None, ColumnTransformer]
"""

from typing import Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.base import BaseEstimator, clone



def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le type, le nombre de valeurs manquantes et le taux de remplissage par colonne.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à analyser.

    Returns
    -------
    pd.DataFrame
        Résumé ordonné par nombre décroissant de valeurs manquantes.
    """
    return pd.DataFrame(
        {
            "type": df.dtypes,
            "missing_count": df.isna().sum(),
            "fill_rate_%": (df.count() / len(df) * 100).round(2),
        }
    ).sort_values(by="missing_count", ascending=False)


def special_columns(
    df: pd.DataFrame, max_modalities: int = 20
) -> dict[str, list[str]]:
    """
    Identifie et catégorise les colonnes d'un DataFrame selon leurs spécificités structurelles.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à analyser.
    max_modalities : int, default=20
        Seuil de cardinalité pour distinguer les catégories à haute cardinalité
        (les colonnes textuelles qui possèdent un nombre élevé de valeurs uniques différentes)

    Returns
    -------
    dict[str, list[str]]
        Dictionnaire contenant les listes de colonnes par catégorie :
        'empty', 'constant', 'numeric', 'categorical', 'high_cardinality', 'boolean'.
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
            if set(df[col].dropna().unique()).issubset({True, False, 1, 0})
        ],
    }


def empty_columns(df: pd.DataFrame) -> list[str]:
    """
    Identifie et retourne les colonnes contenant exclusivement des valeurs manquantes (NaN).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à analyser.

    Returns
    -------
    list[str]
        Liste des noms des colonnes entièrement vides.
    """
    return df.columns[df.isna().all()].tolist()

def missing_like_columns(df: pd.DataFrame) -> list[str]:
    """
    Identifie les colonnes contenant des représentations implicites de valeurs nulles.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à analyser.

    Returns
    -------
    list[str]
        Liste des colonnes contenant des valeurs telles que '', 'na', 'null', etc.
    """
    missing_vals = {np.nan, None, "", "na", "NA", "null", "NULL"}
    return [col for col in df.columns if df[col].isin(missing_vals).any()]


def duplicate_rows(
    df: pd.DataFrame, subset: list[str] | None = None, keep: str = "first"
) -> pd.DataFrame:
    """
    Extrait les lignes identifiées comme doublons dans un DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à analyser.
    subset : list[str] | None, default=None
        Colonnes prises en compte pour le repérage des doublons.
    keep : str, default='first'
        Gestion des doublons ('first', 'last' ou False).

    Returns
    -------
    pd.DataFrame
        Sous-ensemble des lignes en doublon.
    """
    return df[df.duplicated(subset=subset, keep=keep)]


def count_duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> int:
    """
    Compte le nombre de lignes en doublon dans le DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à analyser.
    subset : list[str] | None, default=None
        Sous-ensemble de colonnes à considérer.

    Returns
    -------
    int
        Nombre total de lignes en double.
    """
    return int(df.duplicated(subset=subset, keep="first").sum())


def drop_columns(
    df: pd.DataFrame, cols: list[str], dropped: list[str] | None = None
) -> pd.DataFrame:
    """
    Supprime une liste de colonnes et enregistre les suppressions dans une liste de suivi.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame source.
    cols : list[str]
        Colonnes à supprimer.
    dropped : list[str] | None, default=None
        Liste optionnelle pour historiser les noms des colonnes supprimées.

    Returns
    -------
    pd.DataFrame
        Nouveau DataFrame sans les colonnes spécifiées.
    """
    existing_cols = [c for c in cols if c in df.columns]
    if dropped is not None:
        dropped.extend(existing_cols)
    return df.drop(columns=existing_cols)


def drop_one_column(
    df: pd.DataFrame, col: str, dropped: list[str] | None = None
) -> pd.DataFrame:
    """
    Supprime une seule colonne d'un DataFrame avec historisation optionnelle.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame source.
    col : str
        Nom de la colonne à supprimer.
    dropped : list[str] | None, default=None
        Liste de suivi recevant le nom de la colonne supprimée.

    Returns
    -------
    pd.DataFrame
        Nouveau DataFrame sans la colonne spécifiée.
    """
    if col in df.columns and dropped is not None:
        dropped.append(col)
    return df.drop(columns=[col], errors="ignore")


def convert_bool_to_uint8(
    df: pd.DataFrame, cols: list[str], keep_na: bool = True
) -> pd.DataFrame:
    """
    Convertit des colonnes booléennes en entiers non signés (UInt8).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame source.
    cols : list[str]
        Liste des colonnes booléennes à convertir.
    keep_na : bool, default=True
        Si True, conserve les NaN sous le type nullable UInt8 de Pandas.
        Si False, remplace les NaN par 0.

    Returns
    -------
    pd.DataFrame
        DataFrame avec les types de colonnes mis à jour.
    """
    df_out = df.copy()
    for col in cols:
        if col in df_out.columns:
            if keep_na:
                df_out[col] = df_out[col].astype("boolean").astype("UInt8")
            else:
                df_out[col] = df_out[col].astype("boolean").fillna(False).astype("UInt8")
    return df_out


def lower_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Passe en minuscules les valeurs des colonnes textuelles existantes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame source.
    cols : list[str]
        Colonnes cibles à transformer en minuscules.

    Returns
    -------
    pd.DataFrame
        Copie du DataFrame avec les chaînes converties.
    """
    df_out = df.copy()
    existing_cols = [c for c in cols if c in df_out.columns]
    string_cols = [
        c for c in existing_cols if pd.api.types.is_string_dtype(df_out[c])
    ]
    for col in string_cols:
        df_out[col] = df_out[col].str.lower()
    return df_out


def normalize_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les chaînes de texte de manière vectorisée (minuscules, retrait des accents).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à traiter.

    Returns
    -------
    pd.DataFrame
        DataFrame copié contenant les chaînes nettoyées et les NaN harmonisés.
    """
    df_out = df.copy()
    string_cols = df_out.select_dtypes(include=["object", "string"]).columns
    for col in string_cols:
        df_out[col] = df_out[col].astype(str).str.lower()
        df_out[col] = (
            df_out[col]
            .str.normalize("NFD")
            .str.replace(r"[\u0300-\u036f]", "", regex=True)
        )
        df_out[col] = df_out[col].replace({"nan": np.nan, "none": np.nan})
    return df_out


def fit_transform_clean(
    X_train: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Nettoie l'ensemble d'entraînement et enregistre les statistiques calculées.

    Permet de prévenir les fuites de données (Data Leakage) en calculant
    les médianes, modes et suppressions uniquement sur le train set.

    Parameters
    ----------
    X_train : pd.DataFrame
        Features de l'ensemble d'entraînement.
    config : dict[str, Any]
        Dictionnaire de configuration (seuil de NaN, colonnes binaires, remplacements, médianes).

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        - DataFrame d'entraînement nettoyé.
        - Dictionnaire de statistiques à passer à `transform_clean`.
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
    """
    Applique à un nouveau jeu de données les statistiques apprises lors du fit.

    Parameters
    ----------
    X_test : pd.DataFrame
        Features de test ou de validation.
    stats : dict[str, Any]
        Statistiques calculées sur le train set via `fit_transform_clean`.
    config : dict[str, Any]
        Dictionnaire de configuration pour réappliquer les transformations statiques.

    Returns
    -------
    pd.DataFrame
        DataFrame de test nettoyé de façon alignée avec l'entraînement.
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


def export_train_test_feather(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: str = "data_model",
    target_name: str = "target",
    transform_y: str | None = None,
    drop_cols: list[str] | None = None,
) -> None:
    """
    Exporte les jeux d'entraînement et de test au format binaire rapide Feather.

    Parameters
    ----------
    X_train : pd.DataFrame
        Features d'entraînement.
    X_test : pd.DataFrame
        Features de test.
    y_train : pd.Series
        Cible d'entraînement.
    y_test : pd.Series
        Cible de test.
    output_dir : str, default='data_model'
        Répertoire de destination.
    target_name : str, default='target'
        Nom attribué à la variable cible dans les fichiers exportés.
    transform_y : str | None, default=None
        Applique 'log1p' si spécifié.
    drop_cols : list[str] | None, default=None
        Colonnes additionnelles à exclure de l'export.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    X_tr = X_train.drop(columns=drop_cols or [], errors="ignore").reset_index(drop=True)
    X_te = X_test.drop(columns=drop_cols or [], errors="ignore").reset_index(drop=True)

    if transform_y == "log1p":
        y_tr = pd.Series(np.log1p(y_train.values), name=target_name).reset_index(drop=True)
        y_te = pd.Series(np.log1p(y_test.values), name=target_name).reset_index(drop=True)
    else:
        y_tr = pd.Series(y_train.values, name=target_name).reset_index(drop=True)
        y_te = pd.Series(y_test.values, name=target_name).reset_index(drop=True)

    X_tr.to_feather(out_path / "X_train.feather")
    X_te.to_feather(out_path / "X_test.feather")
    y_tr.to_frame().to_feather(out_path / "y_train.feather")
    y_te.to_frame().to_feather(out_path / "y_test.feather")


def preprocess_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame | None = None,
    numeric_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    scaler: BaseEstimator = RobustScaler(),
) -> tuple[np.ndarray, np.ndarray | None, ColumnTransformer]:
    """
    Ajuste un ColumnTransformer (scaling numérique et One-Hot Encoding) sans fuite de données.

    Parameters
    ----------
    X_train : pd.DataFrame
        Ensemble d'entraînement servant à ajuster les transformateurs.
    X_test : pd.DataFrame | None, default=None
        Ensemble de test ou d'inférence à transformer avec les statistiques du train set.
    numeric_columns : list[str] | None, default=None
        Colonnes numériques à normaliser. Si None, détectées automatiquement.
    categorical_columns : list[str] | None, default=None
        Colonnes catégorielles à encoder. Si None, détectées automatiquement.
    scaler : BaseEstimator, default=RobustScaler()
        Instance de scaler Scikit-Learn pour les colonnes numériques
        (ex. RobustScaler(), StandardScaler()). Clonée en interne.

    Returns
    -------
    tuple[np.ndarray, np.ndarray | None, ColumnTransformer]
        - Matrice numpy d'entraînement transformée.
        - Matrice numpy de test transformée (ou None si X_test n'est pas fourni).
        - Objet ColumnTransformer ajusté pour réutilisation ultérieure.

    Raises
    ------
    ValueError
        Si aucune colonne numérique ou catégorielle valide n'est trouvée.
    """
    if numeric_columns is None:
        numeric_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()

    if categorical_columns is None:
        categorical_columns = (
            X_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        )

    transformers = []
    if numeric_columns:
        transformers.append(("num", clone(scaler), numeric_columns))
    if categorical_columns:
        transformers.append(
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            )
        )

    if not transformers:
        raise ValueError("Aucune colonne numérique ou catégorielle valide trouvée pour la transformation.")

    column_transformer = ColumnTransformer(transformers=transformers, remainder="drop")

    # Fit uniquement sur X_train pour respecter l'étanchéité des données
    X_train_processed = column_transformer.fit_transform(X_train)
    X_test_processed = column_transformer.transform(X_test) if X_test is not None else None

    return X_train_processed, X_test_processed, column_transformer
