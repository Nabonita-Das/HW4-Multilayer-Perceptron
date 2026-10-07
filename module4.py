"""
module4.py
==========

Imports and tests the ``MultilayerPerceptron`` class from
``multilayer_perceptron.py`` using example inputs.

Run:
    python module4.py

Author : Nabonita Das
Course : HW#4 - Multilayer Perceptron Implementation
"""

import numpy as np

from multilayer_perceptron import MultilayerPerceptron


def section(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def main() -> None:
    # ------------------------------------------------------------------ #
    # 1. Build the network and show its architecture
    # ------------------------------------------------------------------ #
    mlp = MultilayerPerceptron()  # 3-5-5-2-2, seeded random weights
    section("1. Network architecture")
    print(mlp)
    print()
    print(mlp.summary())

    # ------------------------------------------------------------------ #
    # 2. Forward propagation on example inputs
    # ------------------------------------------------------------------ #
    section("2. Forward propagation on example inputs")
    examples = {
        "all zeros": [0.0, 0.0, 0.0],
        "all ones": [1.0, 1.0, 1.0],
        "mixed": [0.5, -1.0, 2.0],
        "negative": [-3.0, -2.0, -1.0],
        "large": [50.0, -50.0, 100.0],
    }
    for name, x in examples.items():
        y = mlp.forward(x)
        print(f"{name:<10} input={x!s:<20} -> output={np.round(y, 6)}")

    # ------------------------------------------------------------------ #
    # 3. Layer-by-layer trace for one input
    # ------------------------------------------------------------------ #
    section("3. Layer-by-layer trace for input [0.5, -1.0, 2.0]")
    out, trace = mlp.forward([0.5, -1.0, 2.0], return_all=True)
    for step in trace:
        print(f"Layer {step['layer']} ({step['activation']}):")
        print(f"   z = {np.round(step['z'], 5)}")
        print(f"   a = {np.round(step['a'], 5)}")
    print(f"Final output: {np.round(out, 6)}")

    # ------------------------------------------------------------------ #
    # 4. Batch processing
    # ------------------------------------------------------------------ #
    section("4. Batch forward propagation")
    batch = np.array([[0.1, 0.2, 0.3], [1.0, -1.0, 0.5], [-2.0, 2.0, -2.0]])
    batch_out = mlp(batch)
    print("Input batch shape :", batch.shape)
    print("Output batch shape:", batch_out.shape)
    print(np.round(batch_out, 6))

    # ------------------------------------------------------------------ #
    # 5. Self-checks (assert-based verification)
    # ------------------------------------------------------------------ #
    section("5. Verification checks")

    # 5a. Vectorised pass == independent pure-Python loop implementation
    for x in examples.values():
        assert np.allclose(mlp.forward(x), mlp.forward_pure_python(x)), x
    print("[PASS] NumPy forward == pure-Python forward on all examples")

    # 5b. Batch result == row-by-row results
    row_by_row = np.vstack([mlp.forward(r) for r in batch])
    assert np.allclose(batch_out, row_by_row)
    print("[PASS] Batch output == row-by-row output")

    # 5c. Output layer is ReLU -> never negative
    assert np.all(batch_out >= 0)
    print("[PASS] ReLU output layer produces non-negative values")

    # 5d. Hidden sigmoid layers stay strictly inside (0, 1)
    _, tr = mlp.forward([5.0, -5.0, 5.0], return_all=True)
    assert all(np.all((t["a"] > 0) & (t["a"] < 1)) for t in tr[:3])
    print("[PASS] Sigmoid layers 1-3 stay inside (0, 1)")

    # 5e. Same seed -> same network -> same output (reproducibility)
    other = MultilayerPerceptron(seed=42)
    assert np.allclose(mlp.forward([1, 2, 3]), other.forward([1, 2, 3]))
    print("[PASS] Same seed reproduces identical outputs")

    # 5f. Custom weights: all-zero weights/biases -> sigmoid(0)=0.5 everywhere
    sizes = MultilayerPerceptron.DEFAULT_LAYER_SIZES
    zero_w = [np.zeros((a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
    zero_b = [np.zeros(b) for b in sizes[1:]]
    zero_net = MultilayerPerceptron(weights=zero_w, biases=zero_b)
    # L1..L3 give 0.5; L4 z = 0 -> ReLU(0) = 0
    assert np.allclose(zero_net.forward([9, 9, 9]), [0.0, 0.0])
    print("[PASS] Custom zero weights behave as expected")

    # 5g. Bad input is rejected with a clear error
    try:
        mlp.forward([1.0, 2.0])
    except ValueError as err:
        print(f"[PASS] Wrong input size rejected -> {err}")
    else:
        raise AssertionError("Expected ValueError for wrong input size")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
