# tests/integration/test_preprocessing_pipeline.py

import pandas as pd

from ann_playground.cleaning_utils import preprocess_data


def test_preprocessing_pipeline():
    df = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "city": ["Paris", "Lyon", "Paris"],
        }
    )

    x_train, _, transformer = preprocess_data(df)

    assert x_train.shape[0] == 3

    assert transformer is not None
