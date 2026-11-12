import pandas as pd
import pytest

from ann_playground.eda_utils import (
    correlated_features,
    drop_redundant_features,
    select_existing_features,
    target_correlations,
)


def test_select_existing_features():
    features = ["a", "b", "c"]

    columns = ["a", "c", "d"]

    result = select_existing_features(
        features,
        columns,
    )

    assert result == ["a", "c"]


def test_target_correlations():
    X = pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5],
            "x2": [5, 4, 3, 2, 1],
        }
    )

    y = pd.Series([1, 2, 3, 4, 5])

    result = target_correlations(
        X,
        y,
        n_top=2,
    )

    assert len(result) == 2

    assert "x1" in result.index


def test_correlated_features():
    X = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],
            "c": [10, 8, 6, 4, 2],
        }
    )

    result = correlated_features(
        X,
        threshold=0.8,
    )

    assert len(result) > 0


def test_drop_redundant_features_with_target_column():
    df = pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5],
            "x2": [2, 4, 6, 8, 10],
            "target": [1, 2, 3, 4, 5],
        }
    )

    reduced_df, dropped = drop_redundant_features(
        df,
        y_or_target="target",
        threshold=0.8,
    )

    assert len(dropped) == 1

    assert "target" in reduced_df.columns


def test_drop_redundant_features_with_series_target():
    X = pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5],
            "x2": [2, 4, 6, 8, 10],
        }
    )

    y = pd.Series(
        [1, 2, 3, 4, 5],
        name="target",
    )

    reduced_df, dropped = drop_redundant_features(
        X,
        y_or_target=y,
        threshold=0.8,
    )

    assert isinstance(reduced_df, pd.DataFrame)

    assert len(dropped) == 1


def test_drop_redundant_features_invalid_target():
    df = pd.DataFrame(
        {
            "a": [1, 2],
            "b": [3, 4],
        }
    )

    with pytest.raises(ValueError):
        drop_redundant_features(
            df,
            y_or_target="missing_target",
        )
