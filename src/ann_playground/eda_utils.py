"""
Module : process_utils.py
Description : Outils d'analyse exploratoire, sélection de features, visualisations
              et exportation de jeux de données.

Functions :
    - select_existing_features(features, columns) -> list[str]
    - target_correlations(X, y, n_top) -> pd.Series
    - correlated_features()(X, threshold) -> list[tuple[str, str, float]]
    - drop_redundant_features(X, y_or_target, threshold) -> pd.DataFrame
    - plot_numeric_histograms(X, bins, n_cols, figsize_per_col) -> None
    - plot_qualitative(X, top_n, n_cols, figsize_per_col, figsize, height_per_row) -> None
    - plot_missing_bar(X, top_n, figsize) -> None
    - plot_scatter_vs_target(X, y, cols, transform_y, figsize, alpha, s) -> None
    - plot_corr_heatmap(df, method, title, figsize, annot, fmt, vmin, vmax, cmap) -> None
    - scree_plot(pca, figsize) -> None
    - plot_correlation_circle(pca, components, feature_names) -> None
    - plot_features_correlations(X, threshold, figsize) -> None
    - plot_target_correlations(X, y, n_top, figsize) -> None
"""

from collections.abc import Iterable
import math
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def select_existing_features(
    features: Iterable[str], columns: Iterable[str]
) -> list[str]:
    """
    Filtre une liste de variables pour ne conserver que celles présentes dans un DataFrame.

    Parameters
    ----------
    features : Iterable[str]
        Variables cibles souhaitées.
    columns : Iterable[str]
        Colonnes réelles du DataFrame.

    Returns
    -------
    list[str]
        Liste filtrée conservant l'ordre initial.
    """
    col_set = set(columns)
    return [c for c in features if c in col_set]


def target_correlations(
    X: pd.DataFrame, y: pd.Series | np.ndarray, n_top: int = 10
) -> pd.Series:
    """
    Calcule la corrélation linéaire absolue entre les variables numériques et la cible.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame de variables explicatives.
    y : pd.Series | np.ndarray
        Variable cible numérique.
    n_top : int, default=10
        Nombre de variables les plus corrélées à retourner.

    Returns
    -------
    pd.Series
        Corrélations absolues triées par ordre décroissant.
    """
    X_numeric = X.select_dtypes(include=[np.number]).copy()
    target_values = y.values if isinstance(y, pd.Series) else y
    X_numeric["__target__"] = target_values

    correlations = (
        X_numeric.corr()["__target__"]
        .drop(labels=["__target__"])
        .abs()
        .sort_values(ascending=False)
    )
    return correlations.head(n_top)


def correlated_features(
    X: pd.DataFrame, threshold: float = 0.8
) -> list[tuple[str, str, float]]:
    """
    Identifie les paires de variables fortement corrélées entre elles.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame contenant les variables numériques.
    threshold : float, default=0.8
        Seuil de corrélation absolue minimale pour retenir une paire.

    Returns
    -------
    list[tuple[str, str, float]]
        Liste de tuples (feature_1, feature_2, valeur_correlation) triés par corrélation.
    """
    X_numeric = X.select_dtypes(include=[np.number])
    corr_matrix = X_numeric.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    collinear = [
        (col, row, float(upper.loc[row, col]))
        for col in upper.columns
        for row in upper.index
        if upper.loc[row, col] > threshold
    ]
    return sorted(collinear, key=lambda item: item[2], reverse=True)


def drop_redundant_features(
    X: pd.DataFrame,
    y_or_target: pd.Series | np.ndarray | str,
    threshold: float = 0.90,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Supprime les variables numériques redondantes en conservant, pour chaque paire
    fortement corrélée (|r| > threshold), la variable la plus corrélée à la cible.

    Les variables non numériques présentes dans X sont conservées sans modification.

    Parameters
    ----------
    X : pd.DataFrame
        Ensemble des variables explicatives (ou DataFrame complet contenant la cible).
    y_or_target : pd.Series | np.ndarray | str
        Nom de la colonne cible (si présente dans X) ou vecteur cible (Series / ndarray).
    threshold : float, default=0.90
        Seuil de corrélation linéaire absolue au-delà duquel deux variables
        sont jugées redondantes.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        - DataFrame nettoyé des colonnes redondantes.
        - Liste des noms des variables supprimées (triée par ordre alphabétique).
    """
    df_full = X.copy()

    if isinstance(y_or_target, str):
        target_col = y_or_target
        if target_col not in df_full.columns:
            raise ValueError(f"Target '{target_col}' absente du DataFrame fourni.")
    else:
        target_series = (
            y_or_target.copy()
            if isinstance(y_or_target, pd.Series)
            else pd.Series(y_or_target, index=df_full.index, name="target")
        )
        target_col = target_series.name or "target"
        df_full[target_col] = target_series.values

    # 1. Matrice de corrélation absolue entre features numériques
    features_df = df_full.drop(columns=[target_col])
    corr_matrix = features_df.corr(numeric_only=True).abs()

    # 2. Corrélation absolue de chaque variable numérique avec la cible
    target_corr = (
        df_full.corr(numeric_only=True).abs()[target_col].drop(labels=[target_col])
    )

    valid_cols = corr_matrix.columns.intersection(target_corr.index)
    corr_matrix = corr_matrix.loc[valid_cols, valid_cols]
    target_corr = target_corr.loc[valid_cols]

    to_drop: set[str] = set()
    cols = corr_matrix.columns

    # 3. Parcours des paires au-dessus du seuil et arbitrage
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col_a = cols[i]
            col_b = cols[j]
            if corr_matrix.loc[col_a, col_b] > threshold:
                if target_corr[col_a] >= target_corr[col_b]:
                    to_drop.add(col_b)
                else:
                    to_drop.add(col_a)

    dropped_list = sorted(to_drop)
    reduced_df = features_df.drop(columns=dropped_list, errors="ignore")

    if isinstance(y_or_target, str):
        reduced_df[target_col] = df_full[target_col]

    return reduced_df, dropped_list


def plot_numeric_histograms(
    X: pd.DataFrame,
    bins: int = 40,
    n_cols: int = 3,
    figsize_per_col: tuple[int, int] = (5, 3),
) -> None:
    """
    Génère une grille d'histogrammes pour l'ensemble des colonnes numériques.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame source.
    bins : int, default=40
        Nombre de classes d'intervalles.
    n_cols : int, default=3
        Nombre de subplots par ligne.
    figsize_per_col : tuple[int, int], default=(5, 3)
        Dimensions allouées à chaque subplot.
    """
    num_cols = X.select_dtypes(include=["number"]).columns
    if len(num_cols) == 0:
        return
    n_rows = math.ceil(len(num_cols) / n_cols)
    plt.figure(figsize=(n_cols * figsize_per_col[0], n_rows * figsize_per_col[1]))
    for i, col in enumerate(num_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(X[col].dropna(), bins=bins, kde=True)
        plt.xlabel(col)
        plt.ylabel("Fréquence")
        plt.grid(True, alpha=0.3)
        plt.title(col)
    plt.tight_layout()
    plt.show()


def plot_qualitative(
    X: pd.DataFrame | pd.Series,
    top_n: int = 20,
    n_cols: int = 2,
    figsize_per_col: tuple[int, int] = (6, 4),
    figsize: tuple[int, int] | None = None,
    height_per_row: int = 4,
) -> None:
    """
    Affiche la distribution des variables catégorielles sous forme de barres horizontales.

    Parameters
    ----------
    X : pd.DataFrame | pd.Series
        Variables qualitatives.
    top_n : int, default=20
        Nombre maximum de modalités affichées par variable.
    n_cols : int, default=2
        Nombre de colonnes de la grille.
    figsize_per_col : tuple[int, int], default=(6, 4)
        Taille d'un graphique individuel si figsize est None.
    figsize : tuple[int, int] | None, default=None
        Taille explicite de la figure globale.
    height_per_row : int, default=4
        Hauteur par ligne utilisée pour le calcul automatique.
    """
    if isinstance(X, pd.Series):
        X = X.to_frame(name=X.name or "Category")

    cat_cols = X.select_dtypes(include=["object", "category", "string", "bool"]).columns
    if len(cat_cols) == 0:
        return

    n_rows = math.ceil(len(cat_cols) / n_cols)
    fig_size = figsize or (n_cols * figsize_per_col[0], n_rows * height_per_row)

    plt.figure(figsize=fig_size)
    for i, col in enumerate(cat_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        vc = X[col].astype("string").value_counts(dropna=False).head(top_n)
        sns.barplot(x=vc.values, y=vc.index, color="#439cc8")
        plt.title(col)
        plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_missing_bar(
    X: pd.DataFrame,
    top_n: int | None = None,
    figsize: tuple[int, int] = (8, 4),
) -> None:
    """
    Affiche le pourcentage de valeurs manquantes par colonne sous forme de barplot.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame à analyser.
    top_n : int | None, default=None
        Nombre de colonnes les plus incomplètes à afficher.
    figsize : tuple[int, int], default=(8, 4)
        Dimensions de la figure.
    """
    missing_pct = (X.isna().mean() * 100).sort_values(ascending=False)
    if top_n is not None:
        missing_pct = missing_pct.head(top_n)

    plt.figure(figsize=figsize)
    sns.barplot(x=missing_pct.values, y=missing_pct.index, color="#439cc8")
    plt.xlabel("% de valeurs manquantes")
    plt.ylabel("Colonnes")
    plt.title("Taux de valeurs manquantes par colonne")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_scatter_vs_target(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    cols: Iterable[str],
    transform_y: str | None = None,
    figsize: tuple[int, int] = (15, 10),
    alpha: float = 0.2,
    s: int = 10,
) -> None:
    """
    Affiche un nuage de points pour chaque feature sélectionnée face à la variable cible.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame contenant les variables explicatives.
    y : pd.Series | np.ndarray
        Variable cible.
    cols : Iterable[str]
        Variables explicatives à tracer en abscisse.
    transform_y : str | None, default=None
        Applique 'log1p' sur la cible si demandé.
    figsize : tuple[int, int], default=(15, 10)
        Dimensions globales de la figure.
    alpha : float, default=0.2
        Transparence des points pour gérer la superposition.
    s : int, default=10
        Taille des points dans le scatterplot.
    """
    y_raw = y.values if isinstance(y, pd.Series) else np.asarray(y)
    y_vals = np.log1p(y_raw) if transform_y == "log1p" else y_raw

    cols_list = list(cols)
    if not cols_list:
        return

    n_rows = math.ceil(len(cols_list) / 3)
    plt.figure(figsize=figsize)
    for i, col in enumerate(cols_list, 1):
        plt.subplot(n_rows, 3, i)
        mask = X[col].notna()
        sns.scatterplot(x=X.loc[mask, col], y=y_vals[mask], s=s, alpha=alpha)
        label_prefix = f"{transform_y} " if transform_y else ""
        plt.title(f"{label_prefix}target vs {col}")
    plt.tight_layout()
    plt.show()


def plot_corr_heatmap(
    df: pd.DataFrame,
    method: str = "pearson",
    title: str = "Heatmap des corrélations",
    figsize: tuple[int, int] = (12, 10),
    annot: bool = True,
    fmt: str = ".2f",
    vmin: float = -1.0,
    vmax: float = 1.0,
    cmap: str = "coolwarm",
) -> None:
    """
    Génère une heatmap de corrélation pour l'ensemble des variables numériques.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame analysé.
    method : str, default='pearson'
        Méthode de corrélation ('pearson', 'kendall', 'spearman').
    title : str, default='Heatmap des corrélations'
        Titre du graphique.
    figsize : tuple[int, int], default=(12, 10)
        Dimensions de l'image.
    annot : bool, default=True
        Affiche les valeurs chiffrées dans chaque cellule si True.
    fmt : str, default='.2f'
        Format d'arrondi des valeurs annotées.
    vmin : float, default=-1.0
        Valeur basse de l'échelle couleur.
    vmax : float, default=1.0
        Valeur haute de l'échelle couleur.
    cmap : str, default='coolwarm'
        Palette de couleurs Seaborn.
    """
    corr = df.select_dtypes(include=[np.number]).corr(method=method)
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=annot, fmt=fmt, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def scree_plot(pca: Any, figsize: tuple[int, int] = (10, 6)) -> None:
    """
    Affiche le graphique des éboulis des valeurs propres pour une PCA.

    Parameters
    ----------
    pca : Any
        Instance Scikit-Learn de PCA préalablement ajustée (`fit`).
    figsize : tuple[int, int], default=(10, 6)
        Dimensions de la figure.
    """
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    plt.figure(figsize=figsize)
    plt.bar(
        range(1, len(explained) + 1),
        explained,
        alpha=0.5,
        align="center",
        label="Variance expliquée par composante",
        color="#439cc8",
    )
    plt.plot(
        range(1, len(cumulative) + 1),
        cumulative,
        marker="o",
        linestyle="--",
        color="darkorange",
        label="Variance cumulée",
    )
    plt.xlabel("Composantes principales")
    plt.ylabel("Ratio de variance expliquée")
    plt.title("Éboulis des valeurs propres")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_correlation_circle(
    pca: Any, components: tuple[int, int], feature_names: list[str]
) -> None:
    """
    Dessine le cercle des corrélations sur deux axes d'une ACP.

    Parameters
    ----------
    pca : Any
        Instance ajustée de Scikit-Learn PCA.
    components : tuple[int, int]
        Indices des deux axes à comparer (ex : (0, 1)).
    feature_names : list[str]
        Liste des noms des variables initiales.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.add_artist(plt.Circle((0, 0), 1, color="gray", fill=False, linestyle="-", alpha=0.5))

    for i, feature in enumerate(feature_names):
        x = pca.components_[components[0], i]
        y = pca.components_[components[1], i]
        ax.annotate(
            "",
            xy=(x, y),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="blue", alpha=0.7),
        )
        ax.text(x * 1.05, y * 1.05, feature, color="black", ha="center", va="center")

    var_ratio = pca.explained_variance_ratio_
    ax.set_xlabel(f"Comp. {components[0] + 1} ({var_ratio[components[0]] * 100:.2f}%)")
    ax.set_ylabel(f"Comp. {components[1] + 1} ({var_ratio[components[1]] * 100:.2f}%)")
    ax.set_title("Correlation Circle")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="#555", linewidth=1)
    ax.axvline(0, color="#555", linewidth=1)
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_aspect("equal", adjustable="box")
    plt.show()


def plot_features_correlations(
    X: pd.DataFrame,
    threshold: float = 0.8,
    figsize: tuple[int, int] = (12, 10),
) -> None:
    """
    Affiche une carte thermique (heatmap) masquée illustrant la colinéarité entre variables.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame contenant les variables explicatives.
    threshold : float, default=0.8
        Seuil indicatif de colinéarité pour lecture graphique.
    figsize : tuple[int, int], default=(12, 10)
        Dimensions de la figure Matplotlib.
    """
    numeric_df = X.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return

    corr = numeric_df.corr()
    # Masque pour masquer la partie supérieure symétrique et la diagonale
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1.0,
        vmax=1.0,
        cbar_kws={"label": "Coefficient de corrélation"},
    )
    plt.title(f"Colinarity between numerical features (choosen threshold : {threshold})")
    plt.tight_layout()
    plt.show()


def plot_target_correlations(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    n_top: int = 15,
    figsize: tuple[int, int] = (10, 8),
) -> None:
    """
    Affiche un diagramme en barres des variables les plus corrélées à la cible.

    Parameters
    ----------
    X : pd.DataFrame
        Variables explicatives numériques.
    y : pd.Series | np.ndarray
        Variable cible (numérique ou encodée).
    n_top : int, default=15
        Nombre maximal de variables à représenter.
    figsize : tuple[int, int], default=(10, 8)
        Dimensions de la figure Matplotlib.
    """
    numeric_df = X.select_dtypes(include=[np.number]).copy()
    if numeric_df.empty:
        return

    target_values = y.values if isinstance(y, pd.Series) else np.asarray(y)
    target_name = y.name if isinstance(y, pd.Series) and y.name else "Target"

    numeric_df["__target__"] = target_values
    corrs = (
        numeric_df.corr()["__target__"]
        .drop(labels=["__target__"])
        .abs()
        .sort_values(ascending=False)
        .head(n_top)
    )

    plt.figure(figsize=figsize)
    sns.barplot(
        x=corrs.values,
        y=corrs.index,
        color="#439cc8",
    )
    plt.title(f"Top {len(corrs)} des variables les plus corrélées avec : {target_name}")
    plt.xlabel("Coefficient de corrélation linéaire |r|")
    plt.ylabel("Variables")
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
