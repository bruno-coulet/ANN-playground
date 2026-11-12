from unittest.mock import MagicMock

from ann_playground.models.keras_mlp import (
    StreamlitEpochCallback,
)


def test_callback_initialization():
    chart_placeholder = MagicMock()
    progress_bar = MagicMock()

    callback = StreamlitEpochCallback(
        total_epochs=10,
        chart_placeholder=chart_placeholder,
        progress_bar=progress_bar,
    )

    assert callback.total_epochs == 10
    assert callback.history_records == []


def test_callback_epoch_end():
    chart_placeholder = MagicMock()
    progress_bar = MagicMock()

    callback = StreamlitEpochCallback(
        total_epochs=10,
        chart_placeholder=chart_placeholder,
        progress_bar=progress_bar,
    )

    logs = {
        "loss": 0.5,
        "val_loss": 0.6,
        "accuracy": 0.8,
        "val_accuracy": 0.7,
    }

    callback.on_epoch_end(
        epoch=1,
        logs=logs,
    )

    assert len(callback.history_records) == 1

    progress_bar.progress.assert_called()

    chart_placeholder.plotly_chart.assert_called()


def test_callback_logs_none():
    chart_placeholder = MagicMock()
    progress_bar = MagicMock()

    callback = StreamlitEpochCallback(
        total_epochs=10,
        chart_placeholder=chart_placeholder,
        progress_bar=progress_bar,
    )

    callback.on_epoch_end(
        epoch=0,
        logs=None,
    )

    assert len(callback.history_records) == 0
