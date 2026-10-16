from unittest.mock import MagicMock, patch

from ann_playground.tracker_utils import (
    get_tracking_uri,
    init_mlflow_experiment,
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
