from unittest.mock import patch

import pandas as pd

from ann_playground.eda_utils import (
    plot_missing_bar,
)


@patch("matplotlib.pyplot.show")
def test_plot_missing_bar(mock_show):
    df = pd.DataFrame(
        {
            "a": [1, None, 3],
        }
    )

    plot_missing_bar(df)

    mock_show.assert_called_once()
