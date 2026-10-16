# tests/unit/test_cleaning_utils.py

import pandas as pd

from ann_playground.cleaning_utils import (
    count_duplicates,
    empty_columns,
    normalize_text_features,
)


def test_empty_columns():
    df = pd.DataFrame(
        {
            "a": [1, 2],
            "b": [None, None],
        }
    )

    assert empty_columns(df) == ["b"]


def test_count_duplicates():
    df = pd.DataFrame({"id": [1, 1, 2]})

    assert count_duplicates(df) == 1


def test_normalize_text_features():
    df = pd.DataFrame({"city": ["Évry", "MARSEILLE "]})

    result = normalize_text_features(df)

    assert result.loc[0, "city"] == "evry"
    assert result.loc[1, "city"] == "marseille"
