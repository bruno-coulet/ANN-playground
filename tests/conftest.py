# tests/conftest.py

import numpy as np
import pandas as pd
import pytest

from ann_playground.models.numpy_mlp import NumpyMLP


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame(
        {
            "age": [25, 30, 35],
            "city": ["Paris", "Lyon", "Marseille"],
            "target": [0, 1, 0],
        }
    )


@pytest.fixture
def duplicate_dataframe():
    return pd.DataFrame(
        {
            "id": [1, 1, 2],
            "value": [10, 10, 20],
        }
    )


@pytest.fixture
def mlp():
    return NumpyMLP([4, 8, 3])


@pytest.fixture
def classification_data():
    np.random.seed(42)

    X = np.random.rand(100, 4)

    y = np.random.randint(0, 3, size=100)

    y_one_hot = np.eye(3)[y]

    return X, y_one_hot
