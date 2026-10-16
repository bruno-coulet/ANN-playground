# tests/unit/test_keras_mlp.py

import keras
import numpy as np

from ann_playground.models.keras_mlp import build_keras_mlp


def test_build_model():
    model = build_keras_mlp(
        input_dim=4,
        num_classes=3,
        hidden_layers=[16, 8],
        activation="relu",
        dropout_rate=0.0,
        use_batch_norm=False,
        learning_rate=0.001,
    )

    assert isinstance(model, keras.Model)


def test_predict_shape():
    model = build_keras_mlp(
        input_dim=4,
        num_classes=3,
        hidden_layers=[16],
        activation="relu",
        dropout_rate=0,
        use_batch_norm=False,
        learning_rate=0.001,
    )

    X = np.random.rand(5, 4)

    y = model.predict(X, verbose=0)

    assert y.shape == (5, 3)
