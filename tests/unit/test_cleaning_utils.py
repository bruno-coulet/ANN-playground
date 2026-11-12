# tests/unit/test_cleaning_utils.py

import pandas as pd

from ann_playground.cleaning_utils import (
    convert_bool_to_uint8,
    count_duplicates,
    drop_columns,
    drop_one_column,
    empty_columns,
    fit_transform_clean,
    lower_columns,
    missing_like_columns,
    normalize_text_features,
    special_columns,
    transform_clean,
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


def test_drop_columns():
    df = pd.DataFrame(
        {
            "a": [1],
            "b": [2],
            "c": [3],
        }
    )

    result = drop_columns(
        df,
        ["b"],
    )

    assert "b" not in result.columns


def test_drop_one_column():
    df = pd.DataFrame(
        {
            "a": [1],
            "b": [2],
        }
    )

    result = drop_one_column(
        df,
        "b",
    )

    assert "b" not in result.columns


def test_lower_columns():
    df = pd.DataFrame(
        {
            "city": ["PARIS"],
        }
    )

    result = lower_columns(
        df,
        ["city"],
    )

    assert result["city"].iloc[0] == "paris"


def test_convert_bool_to_uint8():
    df = pd.DataFrame({"flag": [True, False]})

    result = convert_bool_to_uint8(
        df,
        ["flag"],
    )

    assert str(result["flag"].dtype) in (
        "UInt8",
        "uint8",
    )


def test_missing_like_columns():
    df = pd.DataFrame(
        {
            "a": ["na", "test"],
            "b": [1, 2],
        }
    )

    result = missing_like_columns(df)

    assert "a" in result


def test_special_columns():
    df = pd.DataFrame(
        {
            "num": [1, 2, 3],
            "cat": ["a", "b", "c"],
        }
    )

    result = special_columns(df)

    assert "num" in result["numeric"]
    assert "cat" in result["categorical"]


def test_fit_transform_clean():
    df = pd.DataFrame(
        {
            "age": [20, None, 40],
        }
    )

    config = {
        "numeric_median_cols": ["age"],
    }

    cleaned, stats = fit_transform_clean(
        df,
        config,
    )

    assert cleaned["age"].isna().sum() == 0

    assert "medians" in stats


def test_transform_clean():
    stats = {
        "medians": {
            "age": 30,
        }
    }

    config = {}

    df = pd.DataFrame(
        {
            "age": [None],
        }
    )

    result = transform_clean(
        df,
        stats,
        config,
    )

    assert result["age"].iloc[0] == 30
