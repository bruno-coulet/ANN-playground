import keras
import numpy as np

from ann_playground.models.train_baseline import (
    build_mlp_classifier,
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

    X = np.random.rand(
        5,
        8,
    )

    pred = model.predict(
        X,
        verbose=0,
    )

    assert pred.shape == (5, 2)
