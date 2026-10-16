from unittest.mock import MagicMock, patch

import keras
import numpy as np

from ann_playground.models.train_baseline import (
    build_mlp_classifier,
    run_pipeline,
)


def test_build_baseline_model():
    model = build_mlp_classifier(
        input_dim=8,
        num_classes=2,
    )

    assert isinstance(model, keras.Model)

    assert model.output_shape[-1] == 2


def test_baseline_predict():
    model = build_mlp_classifier(
        input_dim=8,
        num_classes=2,
    )

    x = np.random.rand(
        5,
        8,
    )

    predictions = model.predict(
        x,
        verbose=0,
    )

    assert predictions.shape == (5, 2)


@patch("ann_playground.models.train_baseline.ARTIFACTS_DIR")
@patch("ann_playground.models.train_baseline.build_mlp_classifier")
def test_run_pipeline(
    mock_build_model,
    mock_artifacts_dir,
):
    mock_model = MagicMock()

    mock_build_model.return_value = mock_model

    run_pipeline()

    mock_model.summary.assert_called_once()

    mock_model.fit.assert_called_once()

    mock_model.save.assert_called_once()

    mock_artifacts_dir.mkdir.assert_called_once()
