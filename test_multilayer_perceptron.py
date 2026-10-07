"""Unit tests for MultilayerPerceptron. Run with:  pytest -v"""

import numpy as np
import pytest

from multilayer_perceptron import MultilayerPerceptron, relu, sigmoid


@pytest.fixture
def mlp():
    return MultilayerPerceptron(seed=0)


# ---------------------------- architecture ----------------------------- #
def test_default_architecture(mlp):
    assert mlp.layer_sizes == (3, 5, 5, 2, 2)
    assert mlp.activation_names == ("sigmoid", "sigmoid", "sigmoid", "relu")


def test_parameter_shapes(mlp):
    assert [w.shape for w in mlp.weights] == [(3, 5), (5, 5), (5, 2), (2, 2)]
    assert [b.shape for b in mlp.biases] == [(5,), (5,), (2,), (2,)]


def test_parameter_count(mlp):
    assert mlp.n_parameters == 20 + 30 + 12 + 6


# ----------------------------- activations ----------------------------- #
def test_sigmoid_known_values():
    assert sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)
    assert sigmoid(np.array([2.0]))[0] == pytest.approx(1 / (1 + np.exp(-2)))


def test_sigmoid_is_stable_for_extremes():
    # Overflow / invalid / divide would raise; harmless underflow to 0 is fine.
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        out = sigmoid(np.array([-1000.0, 1000.0]))
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(1.0)


def test_relu():
    assert np.array_equal(relu(np.array([-2.0, 0.0, 3.5])), [0.0, 0.0, 3.5])


# ---------------------------- forward pass ----------------------------- #
def test_single_and_batch_shapes(mlp):
    assert mlp.forward([1, 2, 3]).shape == (2,)
    assert mlp.forward(np.ones((7, 3))).shape == (7, 2)


def test_matches_manual_calculation(mlp):
    x = np.array([0.3, -0.7, 1.1])
    a = x
    for W, b, name in zip(mlp.weights, mlp.biases, mlp.activation_names):
        z = a @ W + b
        a = 1 / (1 + np.exp(-z)) if name == "sigmoid" else np.maximum(0, z)
    assert np.allclose(mlp.forward(x), a)


def test_matches_pure_python(mlp):
    for x in ([0, 0, 0], [1, -1, 1], [10, -10, 5]):
        assert np.allclose(mlp.forward(x), mlp.forward_pure_python(x))


def test_batch_equals_rowwise(mlp):
    X = np.random.default_rng(1).normal(size=(6, 3))
    assert np.allclose(mlp(X), np.vstack([mlp(r) for r in X]))


def test_output_non_negative(mlp):
    X = np.random.default_rng(2).normal(scale=5, size=(50, 3))
    assert np.all(mlp(X) >= 0)


def test_trace_contents(mlp):
    out, trace = mlp.forward([1, 2, 3], return_all=True)
    assert [t["layer"] for t in trace] == [1, 2, 3, 4]
    assert np.allclose(trace[-1]["a"], out)
    assert all(0 < v < 1 for t in trace[:3] for v in t["a"])


def test_seed_reproducible():
    a, b = MultilayerPerceptron(seed=7), MultilayerPerceptron(seed=7)
    assert np.allclose(a.forward([1, 2, 3]), b.forward([1, 2, 3]))


def test_different_seed_differs():
    a, b = MultilayerPerceptron(seed=1), MultilayerPerceptron(seed=2)
    assert not np.allclose(a.forward([1, 2, 3]), b.forward([1, 2, 3]))


def test_custom_weights():
    net = MultilayerPerceptron(
        layer_sizes=(2, 1),
        activations=("relu",),
        weights=[np.array([[2.0], [3.0]])],
        biases=[np.array([-1.0])],
    )
    assert net.forward([1, 1])[0] == pytest.approx(4.0)  # 2 + 3 - 1
    assert net.forward([0, 0])[0] == pytest.approx(0.0)  # relu(-1)


# ----------------------------- validation ------------------------------ #
def test_wrong_input_size(mlp):
    with pytest.raises(ValueError):
        mlp.forward([1, 2])


def test_nan_input_rejected(mlp):
    with pytest.raises(ValueError):
        mlp.forward([1, np.nan, 2])


def test_bad_activation_name():
    with pytest.raises(ValueError):
        MultilayerPerceptron(layer_sizes=(2, 2), activations=("tanh",))


def test_activation_count_mismatch():
    with pytest.raises(ValueError):
        MultilayerPerceptron(layer_sizes=(2, 2, 2), activations=("relu",))


def test_bad_weight_shape():
    with pytest.raises(ValueError):
        MultilayerPerceptron(
            layer_sizes=(2, 2),
            activations=("relu",),
            weights=[np.zeros((3, 2))],
            biases=[np.zeros(2)],
        )


def test_weights_without_biases():
    with pytest.raises(ValueError):
        MultilayerPerceptron(
            layer_sizes=(2, 2), activations=("relu",), weights=[np.zeros((2, 2))]
        )
