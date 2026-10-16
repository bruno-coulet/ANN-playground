import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from ann_playground.model_utils import (
    evaluate_classification,
    evaluate_regression,
)


def test_evaluate_regression():
    x_train = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
        }
    )

    y_train = pd.Series([2, 4, 6, 8])

    x_test = pd.DataFrame(
        {
            "feature": [5, 6],
        }
    )

    y_test = pd.Series([10, 12])

    result = evaluate_regression(
        algo=LinearRegression(),
        param_grid=None,
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test,
        cv=2,
    )

    assert "r2" in result
    assert "rmse" in result
    assert "mae" in result


def test_evaluate_regression_grid_search():
    x_train = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6],
        }
    )

    y_train = pd.Series([2, 4, 6, 8, 10, 12])

    x_test = pd.DataFrame(
        {
            "x": [7, 8],
        }
    )

    y_test = pd.Series([14, 16])

    result = evaluate_regression(
        algo=LinearRegression(),
        param_grid={
            "fit_intercept": [True, False],
        },
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test,
        search_type="grid",
        cv=2,
    )

    assert "best_model" in result


def test_evaluate_regression_invalid_search_type():
    x = pd.DataFrame(
        {
            "x": [1, 2, 3, 4],
        }
    )

    y = pd.Series([1, 2, 3, 4])

    with pytest.raises(ValueError):
        evaluate_regression(
            algo=LinearRegression(),
            param_grid={
                "fit_intercept": [True],
            },
            X_train=x,
            y_train=y,
            X_test=x,
            y_test=y,
            search_type="invalid",
            cv=2,
        )


def test_evaluate_classification():
    x_train = pd.DataFrame(
        {
            "x1": [0, 0, 1, 1],
            "x2": [0, 1, 0, 1],
        }
    )

    y_train = pd.Series([0, 0, 1, 1])

    x_test = x_train.copy()
    y_test = y_train.copy()

    result = evaluate_classification(
        algo=LogisticRegression(),
        param_grid=None,
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test,
        cv=2,
    )

    assert "accuracy" in result
    assert "classification_report" in result
    assert "confusion_matrix" in result


def test_evaluate_classification_grid_search():
    x_train = pd.DataFrame(
        {
            "x1": [0, 0, 1, 1, 0, 1],
            "x2": [0, 1, 0, 1, 0, 1],
        }
    )

    y_train = pd.Series([0, 0, 1, 1, 0, 1])

    result = evaluate_classification(
        algo=LogisticRegression(),
        param_grid={
            "C": [0.1, 1.0],
        },
        X_train=x_train,
        y_train=y_train,
        X_test=x_train,
        y_test=y_train,
        search_type="grid",
        cv=2,
    )

    assert "best_model" in result


def test_evaluate_classification_invalid_search_type():
    x = pd.DataFrame(
        {
            "x": [0, 1, 0, 1],
        }
    )

    y = pd.Series([0, 1, 0, 1])

    with pytest.raises(ValueError):
        evaluate_classification(
            algo=LogisticRegression(),
            param_grid={
                "C": [1],
            },
            X_train=x,
            y_train=y,
            X_test=x,
            y_test=y,
            search_type="invalid",
            cv=2,
        )
