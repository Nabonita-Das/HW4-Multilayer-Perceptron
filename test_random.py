"""Randomized tests: random inputs, random weights and random architectures.

Every run uses a fixed master seed so failures are reproducible.
Run with:  pytest -v test_random.py
"""

import numpy as np
import pytest

from multilayer_perceptron import MultilayerPerceptron

MASTER_SEED = 12345


def _manual_forward(net, x):
    """Reference forward pass written directly from the math."""
    a = np.asarray(x, dtype=float)
    for W, b, name in zip(net.weights, net.biases, net.activation_names):
        z = a @ W + b
        a = 1 / (1 + np.exp(-z)) if name == "sigmoid" else np.maximum(0, z)
    return a


@pytest.mark.parametrize("trial", range(25))
def test_random_inputs_default_network(trial):
    """Random inputs of mixed scale: numpy == pure python == manual reference."""
    rng = np.random.default_rng(MASTER_SEED + trial)
    net = MultilayerPerceptron(seed=int(rng.integers(0, 10_000)))
    scale = 10.0 ** rng.integers(-3, 3)  # 0.001 ... 100
    x = rng.normal(scale=scale, size=3)

    out = net.forward(x)
    assert out.shape == (2,)
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0)  # ReLU output
    assert np.allclose(out, net.forward_pure_python(x))
    assert np.allclose(out, _manual_forward(net, x))


@pytest.mark.parametrize("trial", range(25))
def test_random_batches(trial):
    rng = np.random.default_rng(MASTER_SEED + 100 + trial)
    net = MultilayerPerceptron(seed=int(rng.integers(0, 10_000)))
    n = int(rng.integers(1, 40))
    X = rng.normal(scale=3.0, size=(n, 3))
    out = net.forward(X)
    assert out.shape == (n, 2)
    assert np.allclose(out, np.vstack([net.forward(r) for r in X]))


@pytest.mark.parametrize("trial", range(25))
def test_random_architectures_and_weights(trial):
    """Random depth/width/activations with random (not Xavier) weights."""
    rng = np.random.default_rng(MASTER_SEED + 200 + trial)
    depth = int(rng.integers(1, 6))
    sizes = [int(s) for s in rng.integers(1, 9, size=depth + 1)]
    acts = [str(a) for a in rng.choice(["sigmoid", "relu"], size=depth)]
    weights = [rng.normal(size=(a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
    biases = [rng.normal(size=b) for b in sizes[1:]]
    net = MultilayerPerceptron(sizes, acts, weights=weights, biases=biases)

    x = rng.normal(size=sizes[0])
    out = net.forward(x)
    assert out.shape == (sizes[-1],)
    assert np.allclose(out, _manual_forward(net, x))
    assert np.allclose(out, net.forward_pure_python(x))
    if acts[-1] == "sigmoid":
        assert np.all((out > 0) & (out < 1))
    else:
        assert np.all(out >= 0)


def test_random_extreme_inputs_never_overflow():
    rng = np.random.default_rng(MASTER_SEED)
    net = MultilayerPerceptron(seed=1)
    X = rng.choice([-1e6, -1e3, 0.0, 1e3, 1e6], size=(200, 3))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        out = net.forward(X)
    assert np.all(np.isfinite(out))


def test_random_inputs_are_not_mutated():
    rng = np.random.default_rng(MASTER_SEED)
    net = MultilayerPerceptron(seed=3)
    X = rng.normal(size=(10, 3))
    before = X.copy()
    net.forward(X)
    assert np.array_equal(X, before)


def test_forward_is_deterministic():
    rng = np.random.default_rng(MASTER_SEED)
    net = MultilayerPerceptron(seed=5)
    x = rng.normal(size=3)
    assert np.array_equal(net.forward(x), net.forward(x))
