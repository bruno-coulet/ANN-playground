# tests/unit/test_numpy_mlp.py

import numpy as np

from ann_playground.models.numpy_mlp import NumpyMLP


def test_model_creation():
    model = NumpyMLP([4, 8, 3])

    assert len(model.layers) == 2


def test_forward_shape():
    model = NumpyMLP([4, 8, 3])

    X = np.random.rand(16, 4)

    y = model.forward(X)

    assert y.shape == (16, 3)


def test_softmax_sum_to_one():
    model = NumpyMLP([4, 8, 3])

    X = np.random.rand(10, 4)

    output = model.forward(X)

    sums = np.sum(output, axis=1)

    assert np.allclose(sums, np.ones(10), atol=1e-6)


def test_predict_shape():
    model = NumpyMLP([4, 8, 3])

    X = np.random.rand(5, 4)

    pred = model.predict(X)

    assert pred.shape == (5,)


def test_train_step():
    model = NumpyMLP([4, 8, 3])

    X = np.random.rand(32, 4)

    y = np.eye(3)[np.random.randint(0, 3, 32)]

    loss, acc = model.train_step(X, y, lr=0.01)

    assert isinstance(loss, float)
    assert isinstance(acc, float)


def test_evaluate():
    model = NumpyMLP([4, 8, 3])

    X = np.random.rand(32, 4)

    y = np.eye(3)[np.random.randint(0, 3, 32)]

    loss, acc = model.evaluate(X, y)

    assert isinstance(loss, float)
    assert isinstance(acc, float)
