import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from ann_playground.model_utils import (
    evaluate_classification,
    evaluate_regression,
)


def test_evaluate_regression():
    X_train = pd.DataFrame({"feature": [1, 2, 3, 4]})

    y_train = pd.Series([2, 4, 6, 8])

    X_test = pd.DataFrame({"feature": [5, 6]})

    y_test = pd.Series([10, 12])

    result = evaluate_regression(
        algo=LinearRegression(),
        param_grid=None,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        cv=2,
    )

    assert "r2" in result
    assert "rmse" in result
    assert "mae" in result


def test_evaluate_classification():
    X_train = pd.DataFrame(
        {
            "x1": [0, 0, 1, 1],
            "x2": [0, 1, 0, 1],
        }
    )

    y_train = pd.Series([0, 0, 1, 1])

    X_test = X_train.copy()
    y_test = y_train.copy()

    result = evaluate_classification(
        algo=LogisticRegression(),
        param_grid=None,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        cv=2,
    )

    assert "accuracy" in result
    assert "classification_report" in result
    assert "confusion_matrix" in result
