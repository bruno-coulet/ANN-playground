# tests/integration/test_numpy_pipeline.py

import numpy as np

from ann_playground.models.numpy_mlp import NumpyMLP


def test_numpy_end_to_end():
    X = np.random.rand(100, 4)

    y = np.random.randint(0, 3, size=100)

    y_one_hot = np.eye(3)[y]

    model = NumpyMLP([4, 8, 3])

    loss, acc = model.train_step(X, y_one_hot, lr=0.01)

    pred = model.predict(X)

    assert len(pred) == 100
