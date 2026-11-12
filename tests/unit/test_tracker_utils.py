from unittest.mock import MagicMock, patch

from ann_playground.tracker_utils import (
    get_tracking_uri,
    init_mlflow_experiment,
    log_keras_training,
    log_numpy_training,
)


def test_get_tracking_uri():
    uri = get_tracking_uri()

    assert isinstance(uri, str)


@patch("ann_playground.tracker_utils.mlflow.set_experiment")
@patch("ann_playground.tracker_utils.mlflow.set_tracking_uri")
@patch("ann_playground.tracker_utils.MlflowClient")
def test_init_mlflow_experiment_existing(
    mock_client,
    mock_set_tracking_uri,
    mock_set_experiment,
):
    experiment = MagicMock()

    experiment.experiment_id = "123"

    experiment.lifecycle_stage = "active"

    mock_client.return_value.get_experiment_by_name.return_value = experiment

    experiment_id = init_mlflow_experiment(experiment_name="test-exp")

    assert experiment_id == "123"


@patch("ann_playground.tracker_utils.mlflow.set_experiment")
@patch("ann_playground.tracker_utils.mlflow.set_tracking_uri")
@patch("ann_playground.tracker_utils.MlflowClient")
def test_init_mlflow_experiment_created(
    mock_client,
    mock_set_tracking_uri,
    mock_set_experiment,
):
    client = mock_client.return_value

    client.get_experiment_by_name.return_value = None

    client.create_experiment.return_value = "999"

    experiment_id = init_mlflow_experiment(experiment_name="new-exp")

    assert experiment_id == "999"


@patch("ann_playground.tracker_utils.mlflow.keras.log_model")
@patch("ann_playground.tracker_utils.mlflow.log_figure")
@patch("ann_playground.tracker_utils.mlflow.log_metrics")
@patch("ann_playground.tracker_utils.mlflow.log_params")
@patch("ann_playground.tracker_utils.mlflow.start_run")
@patch("ann_playground.tracker_utils.mlflow.set_tracking_uri")
def test_log_keras_training(
    mock_uri,
    mock_start_run,
    mock_log_params,
    mock_log_metrics,
    mock_log_figure,
    mock_log_model,
):
    mock_start_run.return_value.__enter__.return_value = None
    mock_start_run.return_value.__exit__.return_value = None

    history = {
        "loss": [0.5, 0.3],
        "accuracy": [0.7, 0.9],
    }

    log_keras_training(
        run_name="test",
        params={"lr": 0.01},
        model=MagicMock(),
        history=history,
        metrics_test={"accuracy": 0.95},
        fig_cm=MagicMock(),
    )

    mock_log_params.assert_called_once()

    assert mock_log_metrics.call_count >= 2

    mock_log_figure.assert_called_once()

    mock_log_model.assert_called_once()


@patch("ann_playground.tracker_utils.mlflow.log_figure")
@patch("ann_playground.tracker_utils.mlflow.log_metrics")
@patch("ann_playground.tracker_utils.mlflow.log_params")
@patch("ann_playground.tracker_utils.mlflow.start_run")
@patch("ann_playground.tracker_utils.mlflow.set_tracking_uri")
def test_log_numpy_training(
    mock_uri,
    mock_start_run,
    mock_log_params,
    mock_log_metrics,
    mock_log_figure,
):
    mock_start_run.return_value.__enter__.return_value = None
    mock_start_run.return_value.__exit__.return_value = None

    history = [
        {"loss": 0.5, "accuracy": 0.7},
        {"loss": 0.3, "accuracy": 0.9},
    ]

    log_numpy_training(
        run_name="numpy",
        params={"lr": 0.01},
        history=history,
        metrics_test={"accuracy": 0.95},
        fig_cm=MagicMock(),
    )

    mock_log_params.assert_called_once()

    assert mock_log_metrics.call_count >= 2

    mock_log_figure.assert_called_once()
