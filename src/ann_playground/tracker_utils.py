"""
Module : tracker_utils.py
Description : Utilities for logging deep learning experiments (Keras and NumPy MLP)
              using MLflow with either local SQLite or HTTP server backends.

Functions :
    - init_mlflow_experiment(experiment_name, tracking_uri) -> str
    - log_keras_training(run_name, params, model, history, metrics_test, fig_cm) -> None
    - log_numpy_training(run_name, params, history, metrics_test, fig_cm) -> None
"""

from pathlib import Path
from typing import Any, Dict, List
import keras
import mlflow
from mlflow.exceptions import RestException
import plotly.graph_objects as go

from ann_playground.config import MLFLOW_ARTIFACTS_DIR, MLFLOW_TRACKING_URI


def init_mlflow_experiment(
    experiment_name: str = "ANN_Playground",
    tracking_uri: str | None = None,
) -> str:
    """Initialize or retrieve an MLflow experiment without backend mismatch.

    Args:
        experiment_name: Name of the experiment.
        tracking_uri: Optional custom tracking URI. Defaults to config value.

    Returns:
        The experiment ID.
    """
    selected_uri = tracking_uri or MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(selected_uri)

    MLFLOW_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact_location = MLFLOW_ARTIFACTS_DIR.resolve().as_uri()

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is not None:
        return experiment.experiment_id

    try:
        if selected_uri.startswith("http"):
            return client.create_experiment(name=experiment_name)
        return client.create_experiment(
            name=experiment_name,
            artifact_location=artifact_location,
        )
    except RestException:
        exp = client.get_experiment_by_name(experiment_name)
        return exp.experiment_id if exp else "0"


def log_keras_training(
    run_name: str,
    params: Dict[str, Any],
    model: keras.Model,
    history: Dict[str, List[float]],
    metrics_test: Dict[str, float],
    fig_cm: go.Figure | None = None,
) -> None:
    """Log Keras training metrics, hyper-parameters, and artifacts to MLflow."""
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)

        epochs = len(history["loss"])
        for ep in range(epochs):
            step_metrics = {k: float(history[k][ep]) for k in history.keys()}
            mlflow.log_metrics(step_metrics, step=ep + 1)

        mlflow.log_metrics(metrics_test)

        if fig_cm is not None:
            cm_path = Path("confusion_matrix.html")
            fig_cm.write_html(str(cm_path))
            mlflow.log_artifact(str(cm_path))
            cm_path.unlink(missing_ok=True)


def log_numpy_training(
    run_name: str,
    params: Dict[str, Any],
    history: List[Dict[str, float]],
    metrics_test: Dict[str, float],
    fig_cm: go.Figure | None = None,
) -> None:
    """Log custom NumPy MLP execution metrics and parameters to MLflow."""
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)

        for ep, record in enumerate(history):
            mlflow.log_metrics(record, step=ep + 1)

        mlflow.log_metrics(metrics_test)

        if fig_cm is not None:
            cm_path = Path("confusion_matrix.html")
            fig_cm.write_html(str(cm_path))
            mlflow.log_artifact(str(cm_path))
            cm_path.unlink(missing_ok=True)
