"""
smoke_test.py - fast end-to-end sanity check ("does it basically work?").

Run:  python smoke_test.py
Exit code 0 = everything OK, 1 = something broke.
"""

import sys

import numpy as np

from multilayer_perceptron import MultilayerPerceptron


def check(label, condition):
    print(f"[{'OK' if condition else 'FAIL'}] {label}")
    return bool(condition)


def main() -> int:
    results = []

    # 1. Imports and construction
    net = MultilayerPerceptron()
    results.append(check("class imports and builds with defaults", net is not None))
    results.append(check("architecture is 3-5-5-2-2", net.layer_sizes == (3, 5, 5, 2, 2)))
    results.append(
        check("activations are sigmoid x3 + relu",
              net.activation_names == ("sigmoid", "sigmoid", "sigmoid", "relu"))
    )

    # 2. One forward pass
    y = net.forward([0.5, -1.0, 2.0])
    results.append(check("forward returns 2 finite values", y.shape == (2,) and np.all(np.isfinite(y))))

    # 3. Batch pass
    yb = net.forward(np.random.default_rng(0).normal(size=(10, 3)))
    results.append(check("batch of 10 returns shape (10, 2)", yb.shape == (10, 2)))

    # 4. Cross-check with independent implementation
    results.append(
        check("numpy result == pure-python result",
              np.allclose(y, net.forward_pure_python([0.5, -1.0, 2.0])))
    )

    # 5. Random smoke: 200 random inputs must all be finite and >= 0 (ReLU)
    X = np.random.default_rng(1).normal(scale=10, size=(200, 3))
    out = net.forward(X)
    results.append(check("200 random inputs: finite and non-negative",
                         np.all(np.isfinite(out)) and np.all(out >= 0)))

    # 6. Bad input is rejected
    try:
        net.forward([1.0, 2.0])
        rejected = False
    except ValueError:
        rejected = True
    results.append(check("wrong input size raises ValueError", rejected))

    passed = sum(results)
    print(f"\nSmoke test: {passed}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
