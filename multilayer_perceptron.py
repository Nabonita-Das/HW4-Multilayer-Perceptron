"""
multilayer_perceptron.py
========================

A from-scratch implementation of a fully connected Multilayer Perceptron (MLP)
that performs **forward propagation** only.

Default architecture (HW#4)
---------------------------
    Input layer  : 3 neurons
    Hidden 1     : 5 neurons  (Sigmoid)
    Hidden 2     : 5 neurons  (Sigmoid)
    Hidden 3     : 2 neurons  (Sigmoid)
    Output (L4)  : 2 neurons  (ReLU)

Author : Nabonita Das
Course : HW#4 - Multilayer Perceptron Implementation
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Activation functions
# --------------------------------------------------------------------------- #


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid: 1 / (1 + exp(-z)).

    Splitting on the sign of ``z`` avoids overflow in ``exp`` for large
    negative or positive inputs.
    """
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def relu(z: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit: max(0, z)."""
    return np.maximum(0.0, np.asarray(z, dtype=float))


ACTIVATIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sigmoid": sigmoid,
    "relu": relu,
}

# --------------------------------------------------------------------------- #
# The network
# --------------------------------------------------------------------------- #


class MultilayerPerceptron:
    """Fully connected feed-forward neural network (forward pass only).

    Parameters
    ----------
    layer_sizes : sequence of int, default (3, 5, 5, 2, 2)
        Number of neurons in each layer, starting with the input layer.
    activations : sequence of str, default ("sigmoid", "sigmoid", "sigmoid", "relu")
        Activation for every layer *after* the input layer
        (so ``len(activations) == len(layer_sizes) - 1``).
    seed : int or None, default 42
        Seed for reproducible random weights. Ignored if ``weights`` is given.
    weights, biases : list of ndarray, optional
        Custom parameters. ``weights[i]`` must have shape
        ``(layer_sizes[i], layer_sizes[i + 1])`` and ``biases[i]`` shape
        ``(layer_sizes[i + 1],)``.

    Notes
    -----
    Each layer computes::

        z^(l) = a^(l-1) @ W^(l) + b^(l)
        a^(l) = activation_l(z^(l))
    """

    DEFAULT_LAYER_SIZES: Tuple[int, ...] = (3, 5, 5, 2, 2)
    DEFAULT_ACTIVATIONS: Tuple[str, ...] = ("sigmoid", "sigmoid", "sigmoid", "relu")

    def __init__(
        self,
        layer_sizes: Sequence[int] = DEFAULT_LAYER_SIZES,
        activations: Sequence[str] = DEFAULT_ACTIVATIONS,
        seed: Optional[int] = 42,
        weights: Optional[List[np.ndarray]] = None,
        biases: Optional[List[np.ndarray]] = None,
    ) -> None:
        self.layer_sizes: Tuple[int, ...] = tuple(int(n) for n in layer_sizes)
        self.activation_names: Tuple[str, ...] = tuple(a.lower() for a in activations)
        self._validate_architecture()

        self.activation_fns = [ACTIVATIONS[a] for a in self.activation_names]

        if (weights is None) != (biases is None):
            raise ValueError("Provide both `weights` and `biases`, or neither.")

        if weights is None:
            self.weights, self.biases = self._random_parameters(seed)
        else:
            self.weights = [np.asarray(w, dtype=float) for w in weights]
            self.biases = [np.asarray(b, dtype=float) for b in biases]
            self._validate_parameters()

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    def _validate_architecture(self) -> None:
        if len(self.layer_sizes) < 2:
            raise ValueError("Need at least an input and an output layer.")
        if any(n < 1 for n in self.layer_sizes):
            raise ValueError("Every layer must have at least one neuron.")
        if len(self.activation_names) != len(self.layer_sizes) - 1:
            raise ValueError(
                "Need exactly one activation per non-input layer: expected "
                f"{len(self.layer_sizes) - 1}, got {len(self.activation_names)}."
            )
        unknown = [a for a in self.activation_names if a not in ACTIVATIONS]
        if unknown:
            raise ValueError(
                f"Unknown activation(s) {unknown}. Choose from {sorted(ACTIVATIONS)}."
            )

    def _random_parameters(
        self, seed: Optional[int]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Xavier/Glorot-uniform weights and small non-zero biases."""
        rng = np.random.default_rng(seed)
        weights, biases = [], []
        for fan_in, fan_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            weights.append(rng.uniform(-limit, limit, size=(fan_in, fan_out)))
            biases.append(rng.uniform(-0.1, 0.1, size=fan_out))
        return weights, biases

    def _validate_parameters(self) -> None:
        n_layers = len(self.layer_sizes) - 1
        if len(self.weights) != n_layers or len(self.biases) != n_layers:
            raise ValueError(f"Expected {n_layers} weight matrices and bias vectors.")
        for i in range(n_layers):
            w_shape = (self.layer_sizes[i], self.layer_sizes[i + 1])
            b_shape = (self.layer_sizes[i + 1],)
            if self.weights[i].shape != w_shape:
                raise ValueError(
                    f"weights[{i}] has shape {self.weights[i].shape}, expected {w_shape}."
                )
            if self.biases[i].shape != b_shape:
                raise ValueError(
                    f"biases[{i}] has shape {self.biases[i].shape}, expected {b_shape}."
                )

    # ------------------------------------------------------------------ #
    # Forward propagation
    # ------------------------------------------------------------------ #
    def _prepare_input(self, x) -> Tuple[np.ndarray, bool]:
        """Return a 2-D float array (batch, features) and whether input was 1-D."""
        arr = np.asarray(x, dtype=float)
        single = arr.ndim == 1
        if arr.ndim not in (1, 2):
            raise ValueError("Input must be 1-D (one sample) or 2-D (batch).")
        if single:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self.layer_sizes[0]:
            raise ValueError(
                f"Expected {self.layer_sizes[0]} input features, got {arr.shape[1]}."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("Input contains NaN or infinity.")
        return arr, single

    def forward(self, x, return_all: bool = False):
        """Run forward propagation.

        Parameters
        ----------
        x : array-like, shape (3,) or (n_samples, 3)
        return_all : bool
            If True, also return a list with the pre-activation ``z`` and
            activation ``a`` of every layer (useful for inspection).

        Returns
        -------
        output : ndarray
            Shape (2,) for a single sample, (n_samples, 2) for a batch.
        trace : list of dict, only when ``return_all`` is True
            ``trace[l] = {"layer", "z", "a", "activation"}``.
        """
        a, single = self._prepare_input(x)
        trace = []
        for i, (W, b, act, name) in enumerate(
            zip(self.weights, self.biases, self.activation_fns, self.activation_names),
            start=1,
        ):
            z = a @ W + b
            a = act(z)
            if return_all:
                trace.append(
                    {
                        "layer": i,
                        "activation": name,
                        "z": z[0].copy() if single else z.copy(),
                        "a": a[0].copy() if single else a.copy(),
                    }
                )
        out = a[0] if single else a
        return (out, trace) if return_all else out

    # Convenience aliases -------------------------------------------------
    __call__ = forward
    predict = forward

    def forward_pure_python(self, x: Sequence[float]) -> List[float]:
        """Independent loop-based forward pass (no NumPy matrix ops).

        Written separately from :meth:`forward` so the two can be used to
        cross-verify each other. Accepts a single sample only.
        """
        import math

        if len(x) != self.layer_sizes[0]:
            raise ValueError(
                f"Expected {self.layer_sizes[0]} input features, got {len(x)}."
            )

        def _sig(v: float) -> float:
            if v >= 0:
                return 1.0 / (1.0 + math.exp(-v))
            e = math.exp(v)
            return e / (1.0 + e)

        a = [float(v) for v in x]
        for W, b, name in zip(self.weights, self.biases, self.activation_names):
            nxt = []
            for j in range(W.shape[1]):
                z = float(b[j])
                for i in range(W.shape[0]):
                    z += a[i] * float(W[i, j])
                nxt.append(_sig(z) if name == "sigmoid" else max(0.0, z))
            a = nxt
        return a

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def n_parameters(self) -> int:
        return int(sum(w.size + b.size for w, b in zip(self.weights, self.biases)))

    def summary(self) -> str:
        """Human-readable architecture table."""
        lines = [
            f"{'Layer':<12}{'Neurons':>8}{'Activation':>12}{'W shape':>10}{'b shape':>9}{'Params':>8}",
            "-" * 59,
            f"{'Input':<12}{self.layer_sizes[0]:>8}{'-':>12}{'-':>10}{'-':>9}{0:>8}",
        ]
        for i, (W, b) in enumerate(zip(self.weights, self.biases), start=1):
            kind = "Output" if i == len(self.weights) else f"Hidden {i}"
            lines.append(
                f"{kind + f' (L{i})':<12}{self.layer_sizes[i]:>8}"
                f"{self.activation_names[i - 1]:>12}"
                f"{str(W.shape):>10}{str(b.shape):>9}{W.size + b.size:>8}"
            )
        lines.append("-" * 59)
        lines.append(f"Total trainable parameters: {self.n_parameters}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        arch = "-".join(map(str, self.layer_sizes))
        return (
            f"MultilayerPerceptron(architecture={arch}, "
            f"activations={list(self.activation_names)}, params={self.n_parameters})"
        )
