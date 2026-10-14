"""
Module : model_utils.py
Description : Entraînement, recherche d'hyperparamètres et évaluation unifiée
              des modèles de régression et classification Scikit-Learn.

Functions :
    - evaluate_regression(algo, param_grid, X_train, y_train, X_test, y_test, search_type, scoring, cv, inverse_transform_y) -> dict[str, Any]
    - evaluate_classification(algo, param_grid, X_train, y_train, X_test, y_test, search_type, scoring, cv) -> dict[str, Any]
"""

from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin


def evaluate_regression(
    algo: Any,
    param_grid: dict[str, Any] | None,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    search_type: str = "grid",
    scoring: str = "r2",
    cv: int = 5,
    inverse_transform_y: str | None = None,
) -> dict[str, Any]:
    """
    Entraîne, optimise et évalue un estimateur de régression sur les métriques standard (R2, RMSE, MAE).

    Parameters
    ----------
    algo : Any
        Estimateur Scikit-Learn non entraîné.
    param_grid : dict[str, Any] | None
        Grille d'hyperparamètres à explorer. Si None ou vide, entraîne le modèle directement.
    X_train : pd.DataFrame
        Features d'entraînement.
    y_train : pd.Series
        Cible d'entraînement.
    X_test : pd.DataFrame
        Features de test.
    y_test : pd.Series
        Cible de test.
    search_type : str, default='grid'
        Stratégie d'exploration ('grid' ou 'random').
    scoring : str, default='r2'
        Métrique d'optimisation pour la validation croisée.
    cv : int, default=5
        Nombre de plis (folds) de validation croisée.
    inverse_transform_y : str | None, default=None
        Passe 'expm1' pour recalculer les scores dans l'échelle d'origine si la cible a été log-transformée.

    Returns
    -------
    dict[str, Any]
        Dictionnaire avec les clés 'best_model', 'best_params', 'r2', 'rmse', 'mae', 'cv_results'.
    """
    if param_grid is None or len(param_grid) == 0:
        best_model = algo
        best_model.fit(X_train, y_train)
        best_params = algo.get_params()
        cv_results = cross_val_score(best_model, X_train, y_train, cv=cv, scoring=scoring)
    else:
        if search_type == "grid":
            search = GridSearchCV(algo, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
        elif search_type == "random":
            search = RandomizedSearchCV(
                algo, param_grid, cv=cv, scoring=scoring, n_jobs=-1, random_state=42
            )
        else:
            raise ValueError("search_type doit être 'grid' ou 'random'.")

        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_
        cv_results = search.cv_results_

    y_pred = best_model.predict(X_test)

    if inverse_transform_y == "expm1":
        y_test_eval = np.expm1(y_test)
        y_pred_eval = np.expm1(y_pred)
        unit = " (Unités réelles)"
    else:
        y_test_eval = y_test
        y_pred_eval = y_pred
        unit = ""

    r2 = r2_score(y_test_eval, y_pred_eval)
    rmse = mean_squared_error(y_test_eval, y_pred_eval) ** 0.5
    mae = mean_absolute_error(y_test_eval, y_pred_eval)

    print(f"=== Régression : {algo.__class__.__name__} ===")
    print(f"Meilleurs paramètres : {best_params}")
    print(f"R2 (Test){unit} : {r2:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")

    return {
        "best_model": best_model,
        "best_params": best_params,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "cv_results": cv_results,
    }


def evaluate_classification(
    algo: Any,
    param_grid: dict[str, Any] | None,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    search_type: str = "grid",
    scoring: str = "accuracy",
    cv: int = 5,
) -> dict[str, Any]:
    """
    Entraîne, optimise et évalue un classifieur avec matrice de confusion et rapport détaillé.

    Parameters
    ----------
    algo : Any
        Classifieur Scikit-Learn non ajusté.
    param_grid : dict[str, Any] | None
        Grille d'hyperparamètres à optimiser (ou None).
    X_train : pd.DataFrame
        Features d'entraînement.
    y_train : pd.Series
        Labels cibles d'entraînement.
    X_test : pd.DataFrame
        Features de test.
    y_test : pd.Series
        Labels cibles de test.
    search_type : str, default='grid'
        Type de recherche ('grid' ou 'random').
    scoring : str, default='accuracy'
        Métrique cible d'évaluation.
    cv : int, default=5
        Nombre de folds pour la validation croisée.

    Returns
    -------
    dict[str, Any]
        Dictionnaire contenant 'best_model', 'best_params', 'accuracy',
        'classification_report', 'confusion_matrix', 'cv_results'.
    """
    if np.issubdtype(y_train.dtype, np.number):
        y_train_encoded = y_train
        y_test_encoded = y_test
        unique_categories = [str(c) for c in np.unique(y_train)]
    else:
        encoder = LabelEncoder()
        y_train_encoded = encoder.fit_transform(y_train)
        y_test_encoded = encoder.transform(y_test)
        unique_categories = [str(c) for c in encoder.classes_]

    if param_grid is None or len(param_grid) == 0:
        best_model = algo
        best_model.fit(X_train, y_train_encoded)
        best_params = algo.get_params()
        cv_results = cross_val_score(
            best_model, X_train, y_train_encoded, cv=cv, scoring=scoring
        )
    else:
        if search_type == "grid":
            search = GridSearchCV(algo, param_grid=param_grid, cv=cv, scoring=scoring)
        elif search_type == "random":
            search = RandomizedSearchCV(
                algo, param_distributions=param_grid, cv=cv, scoring=scoring, random_state=42
            )
        else:
            raise ValueError("search_type doit être 'grid' ou 'random'.")

        search.fit(X_train, y_train_encoded)
        best_model = search.best_estimator_
        best_params = search.best_params_
        cv_results = search.cv_results_

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test_encoded, y_pred)
    class_report = classification_report(
        y_test_encoded, y_pred, target_names=unique_categories, zero_division=0
    )
    conf_matrix = confusion_matrix(y_test_encoded, y_pred)

    print(f"=== Classification : {algo.__class__.__name__} ===")
    print(f"Meilleurs paramètres : {best_params}")
    print(f"Accuracy : {accuracy:.4f}\n")
    print("Rapport de classification :")
    print(class_report)

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=unique_categories,
        yticklabels=unique_categories,
    )
    plt.xlabel("Étiquettes prédites")
    plt.ylabel("Étiquettes réelles")
    plt.title(f"Matrice de confusion - {algo.__class__.__name__}")
    plt.tight_layout()
    plt.show()

    return {
        "best_model": best_model,
        "best_params": best_params,
        "accuracy": accuracy,
        "classification_report": class_report,
        "confusion_matrix": conf_matrix,
        "cv_results": cv_results,
    }
