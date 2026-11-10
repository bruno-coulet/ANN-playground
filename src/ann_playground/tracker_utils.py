"""
Module: tracker_utils.py
Description: MLflow tracking orchestration, experiment lifecycle management,
             and robust artifact persistence for Keras 3 and custom NumPy models.

Functions:
    - init_mlflow_experiment(experiment_name: str) -> str
    - log_keras_training(run_name, params, model, history, metrics_test, fig_cm) -> None
    - log_numpy_training(run_name, params, model, history, metrics_test, fig_cm) -> None
"""

import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import plotly.graph_objects as go

from ann_playground.config import MLFLOW_TRACKING_URI


class NumpyPyfuncWrapper(mlflow.pyfunc.PythonModel):
    """
    Wrapper class enabling the serialization of custom NumPy models
    using MLflow's standard pyfunc flavor.
    """

    def __init__(self, model_instance: Any) -> None:
        """
        Initialize the pyfunc wrapper.

        Parameters
        ----------
        model_instance : Any
            The instantiated and fitted NumpyMLP model.
        """
        self.model = model_instance

    def predict(
        self, context: mlflow.pyfunc.PythonModelContext, model_input: np.ndarray
    ) -> np.ndarray:
        """
        Execute forward pass inference.

        Parameters
        ----------
        context : mlflow.pyfunc.PythonModelContext
            Context provided by MLflow during model loading.
        model_input : np.ndarray
            Input feature matrix.

        Returns
        -------
        np.ndarray
            Predicted class indices or probabilities.
        """
        return self.model.predict(np.asarray(model_input))


def init_mlflow_experiment(experiment_name: str) -> str:
    """
    Configure the tracking client and initialize the target experiment.

    Parameters
    ----------
    experiment_name : str
        Name of the MLflow experiment.

    Returns
    -------
    str
        Unique identifier assigned to the experiment.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = exp.experiment_id

    mlflow.set_experiment(experiment_name)
    return experiment_id


def log_keras_training(
    run_name: str,
    params: dict[str, Any],
    model: Any,
    history: dict[str, Sequence[float]],
    metrics_test: dict[str, float],
    fig_cm: go.Figure,
) -> None:
    """
    Log hyperparameters, epoch curves, confusion matrix, and Keras 3 artifact.

    Parameters
    ----------
    run_name : str
        Label assigned to the current run.
    params : dict[str, Any]
        Model topology and optimization hyperparameters.
    model : Any
        Trained Keras model instance.
    history : dict[str, Sequence[float]]
        Training metric trajectories per epoch.
    metrics_test : dict[str, float]
        Evaluation metrics on the test partition.
    fig_cm : go.Figure
        Plotly confusion matrix figure.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics_test)

        loss_series = history.get("loss", [])
        epochs_count: int = len(loss_series)

        for epoch in range(epochs_count):
            step_metrics: dict[str, float] = {
                metric_name: float(values[epoch])
                for metric_name, values in history.items()
                if epoch < len(values)
            }
            mlflow.log_metrics(step_metrics, step=epoch)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path: Path = Path(tmp_dir)

            cm_path: Path = tmp_path / "confusion_matrix.html"
            fig_cm.write_html(str(cm_path))
            mlflow.log_artifact(str(cm_path))

            mlflow.keras.log_model(model, artifact_path="model")


def log_numpy_training(
    run_name: str,
    params: dict[str, Any],
    model: Any | None,
    history: Sequence[dict[str, float]],
    metrics_test: dict[str, float],
    fig_cm: go.Figure,
) -> None:
    """
    Log parameters, step metrics, confusion matrix, and serialized NumPy model.

    Parameters
    ----------
    run_name : str
        Label assigned to the current run.
    params : dict[str, Any]
        Model topology and optimization hyperparameters.
    model : Any | None
        Fitted NumpyMLP instance. If None, model serialization is skipped.
    history : Sequence[dict[str, float]]
        Step-wise training metrics recorded during gradient descent.
    metrics_test : dict[str, float]
        Evaluation metrics on the test partition.
    fig_cm : go.Figure
        Plotly confusion matrix figure.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics_test)

        for step_idx, step_dict in enumerate(history):
            mlflow.log_metrics(step_dict, step=step_idx)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path: Path = Path(tmp_dir)

            cm_path: Path = tmp_path / "confusion_matrix.html"
            fig_cm.write_html(str(cm_path))
            mlflow.log_artifact(str(cm_path))

            if model is not None:
                try:
                    mlflow.pyfunc.log_model(
                        artifact_path="model",
                        python_model=NumpyPyfuncWrapper(model),
                    )
                except mlflow.exceptions.MlflowException:
                    # Fallback to generic joblib artifact if pyfunc fails
                    model_path: Path = tmp_path / "numpy_mlp.joblib"
                    joblib.dump(model, model_path)
                    mlflow.log_artifact(str(model_path), artifact_path="model")
