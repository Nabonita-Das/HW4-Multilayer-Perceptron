# HW#4 – Multilayer Perceptron Implementation

**Student:** Nabonita Das

A from-scratch Multilayer Perceptron (forward propagation only) implemented with NumPy, together with a notebook demonstration, automated tests, and an IEEE-style technical report.

## Architecture

```
Input (3) ──► Hidden 1 (5, Sigmoid) ──► Hidden 2 (5, Sigmoid) ──► Hidden 3 (2, Sigmoid) ──► Output (2, ReLU)
```

| Layer | Neurons | Activation | Weights | Biases |
|-------|--------:|------------|---------|--------|
| Input | 3 | – | – | – |
| Hidden 1 | 5 | Sigmoid | 3×5 | 5 |
| Hidden 2 | 5 | Sigmoid | 5×5 | 5 |
| Hidden 3 | 2 | Sigmoid | 5×2 | 2 |
| Output (L4) | 2 | ReLU | 2×2 | 2 |

Total parameters: **68**. Each layer computes `a = f(a_prev @ W + b)`.

## Required files

| File | Purpose |
|------|---------|
| `multilayer_perceptron.py` | Contains the `MultilayerPerceptron` class |
| `module4.py` | Imports and tests the class using example inputs |
| `module4.ipynb` | Demonstrates the implementation and shows the forward-propagation results |

## Additional files

| File | Purpose |
|------|---------|
| `test_multilayer_perceptron.py` | Unit tests (pytest) |
| `test_random.py` | Randomized tests: random inputs, weights and architectures |
| `smoke_test.py` | Fast end-to-end sanity check |
| `report/` | IEEE-style technical report (PDF and LaTeX source), figures, and the script that regenerates them |
| `requirements.txt` | Dependencies |

## How to run

```bash
pip install -r requirements.txt
python module4.py          # run the demo and self-checks
python smoke_test.py       # quick smoke test
pytest -v                  # run all tests
jupyter notebook module4.ipynb
```

Keep `multilayer_perceptron.py`, `module4.py` and `module4.ipynb` in the same folder.

## Usage

```python
from multilayer_perceptron import MultilayerPerceptron

mlp = MultilayerPerceptron()                 # 3-5-5-2-2, seeded random weights
mlp.forward([0.5, -1.0, 2.0])                # single sample -> shape (2,)
mlp.forward([[0, 0, 0], [1, 1, 1]])          # batch         -> shape (2, 2)
out, trace = mlp.forward([1, 2, 3], return_all=True)   # per-layer z and a
print(mlp.summary())
```

## Features

- Exact architecture required by the assignment, as defaults
- Configurable layer sizes and activations
- Numerically stable sigmoid (no overflow for large inputs)
- Single-sample and batch inputs
- Optional per-layer trace (`return_all=True`)
- Independent pure-Python implementation (`forward_pure_python`) used to cross-check the NumPy version
- Input and parameter validation with clear error messages
- Reproducible weights through a seed, or custom weights and biases
- 99 tests (unit, randomized and smoke)

## Report

The technical report is in `report/`:

- `report/Nabonita_Das_HW4_IEEE_Report.pdf` – the paper
- `report/report.tex` – LaTeX source
- `report/make_figures.py` – regenerates every figure from the actual class

```bash
cd report
python make_figures.py
```
