"""
make_figures.py
===============

Regenerates every figure and every number used in the IEEE report directly
from the ``MultilayerPerceptron`` class, so nothing in the paper is typed in
by hand.

Run from inside the ``report`` folder:

    python make_figures.py

Outputs (written next to this script):
    figures/*.pdf, figures/*.png      figures used by report.tex
    generated/*.tex                   tables and number macros used by report.tex

Author: Nabonita Das (HW#4)
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from multilayer_perceptron import MultilayerPerceptron, relu, sigmoid  # noqa: E402

FIG = HERE / "figures"
GEN = HERE / "generated"
FIG.mkdir(exist_ok=True)
GEN.mkdir(exist_ok=True)

# IEEE-like look: serif font, small text, black/grey palette that survives
# black-and-white printing.
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.1,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }
)

COL_W = 3.5    # IEEE single-column width in inches
FULL_W = 7.16  # IEEE double-column width in inches

GREY_IN, GREY_HID, GREY_OUT = "#d9d9d9", "#8c8c8c", "#262626"


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print("saved", name)


# --------------------------------------------------------------------------- #
# The two networks used throughout the paper
# --------------------------------------------------------------------------- #
default_net = MultilayerPerceptron()  # seed 42, Xavier uniform

sizes = MultilayerPerceptron.DEFAULT_LAYER_SIZES
_rng = np.random.default_rng(277)
_W = [_rng.normal(0, 3.0, size=(a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
_B = [_rng.normal(0, 0.5, size=b) for b in sizes[1:]]
_W[-1] = np.abs(_W[-1]) * 0.6
_B[-1] = np.abs(_B[-1]) * 0.2
demo_net = MultilayerPerceptron(weights=_W, biases=_B)

X0 = np.array([0.5, -1.0, 2.0])

# --------------------------------------------------------------------------- #
# Fig. 1  Network architecture
# --------------------------------------------------------------------------- #
def fig_architecture():
    fig, ax = plt.subplots(figsize=(FULL_W, 2.55))
    layers = list(sizes)
    xs = np.linspace(0.08, 0.92, len(layers))
    pos = []
    for x, n in zip(xs, layers):
        ys = 0.46 + (np.arange(n) - (n - 1) / 2) * -0.13
        pos.append(list(zip([x] * n, ys)))
    for l in range(len(layers) - 1):
        for (x1, y1) in pos[l]:
            for (x2, y2) in pos[l + 1]:
                ax.plot([x1, x2], [y1, y2], color="#9a9a9a", lw=0.4, zorder=1)
    fills = [GREY_IN, GREY_HID, GREY_HID, GREY_HID, GREY_OUT]
    for l, pts in enumerate(pos):
        for (x, y) in pts:
            ax.scatter(x, y, s=210, color=fills[l], edgecolor="k", linewidth=0.7, zorder=2)
    heads = ["Input layer\n3 neurons", "Hidden layer 1\n5 neurons", "Hidden layer 2\n5 neurons",
             "Hidden layer 3\n2 neurons", "Output layer (L4)\n2 neurons"]
    acts = ["", r"Sigmoid", r"Sigmoid", r"Sigmoid", r"ReLU"]
    for l in range(len(layers)):
        ax.text(xs[l], 0.97, heads[l], ha="center", va="top", fontsize=7.5, fontweight="bold")
        if acts[l]:
            ax.text(xs[l], 0.03, f"$f$ = {acts[l]}", ha="center", va="bottom", fontsize=7.5)
    shapes = [r"$W^{(1)}$: 3$\times$5", r"$W^{(2)}$: 5$\times$5", r"$W^{(3)}$: 5$\times$2", r"$W^{(4)}$: 2$\times$2"]
    for l in range(4):
        ax.text((xs[l] + xs[l + 1]) / 2, 0.80, shapes[l], ha="center", va="center", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#555555", lw=0.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    save(fig, "fig_architecture")


# --------------------------------------------------------------------------- #
# Fig. 2  Forward-pass data flow
# --------------------------------------------------------------------------- #
def box(ax, x, y, w, h, text, fc="white", fs=7, bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.02",
                       fc=fc, ec="k", lw=0.7)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, label=None, dy=0.075):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
                                 lw=0.8, color="k"))
    if label:
        ax.text((x1 + x2) / 2, y1 + dy, label, ha="center", va="bottom", fontsize=6.0)


def fig_flow():
    fig, ax = plt.subplots(figsize=(FULL_W, 1.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    bw, bh, y = 0.138, 0.50, 0.12
    x0 = 0.092
    gap = (1 - 2 * x0 - 4 * bw) / 3
    xs = [x0 + i * (bw + gap) for i in range(4)]
    labels = [
        "Layer 1\n$z^{(1)}=a^{(0)}W^{(1)}+b^{(1)}$\n$a^{(1)}=\\sigma(z^{(1)})$",
        "Layer 2\n$z^{(2)}=a^{(1)}W^{(2)}+b^{(2)}$\n$a^{(2)}=\\sigma(z^{(2)})$",
        "Layer 3\n$z^{(3)}=a^{(2)}W^{(3)}+b^{(3)}$\n$a^{(3)}=\\sigma(z^{(3)})$",
        "Layer 4\n$z^{(4)}=a^{(3)}W^{(4)}+b^{(4)}$\n$a^{(4)}=\\mathrm{ReLU}(z^{(4)})$",
    ]
    fcs = ["#efefef", "#efefef", "#efefef", "#cfcfcf"]
    for x, t, fc in zip(xs, labels, fcs):
        box(ax, x, y, bw, bh, t, fc=fc, fs=6.1)
    ym = y + bh / 2
    arrow(ax, 0.005, ym, xs[0], ym, "$a^{(0)}\\in\\mathbb{R}^{3}$", dy=0.05)
    for i, d in enumerate(["5", "5", "2"]):
        arrow(ax, xs[i] + bw, ym, xs[i + 1], ym, f"$a^{{({i+1})}}\\in\\mathbb{{R}}^{{{d}}}$", dy=0.05)
    arrow(ax, xs[3] + bw, ym, 0.995, ym, "$\\hat{y}\\in\\mathbb{R}^{2}$", dy=0.05)
    ax.text(0.5, 0.97, "Input vector $\\rightarrow$ four fully connected layers $\\rightarrow$ output vector",
            ha="center", va="top", fontsize=7.5)
    save(fig, "fig_flow")


# --------------------------------------------------------------------------- #
# Fig. 3  Software structure
# --------------------------------------------------------------------------- #
def fig_software():
    fig, ax = plt.subplots(figsize=(COL_W, 2.75))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # library box
    box(ax, 0.02, 0.06, 0.50, 0.86, "", fc="#efefef")
    ax.text(0.27, 0.87, "multilayer_perceptron.py", ha="center", va="center",
            fontsize=6.8, fontweight="bold")
    ax.text(0.27, 0.76, "sigmoid(z)    relu(z)", ha="center", va="center", fontsize=6.4)
    ax.text(0.27, 0.46,
            "class MultilayerPerceptron\n\n"
            "__init__(sizes, acts,\n    seed, weights, biases)\n"
            "forward(x, return_all)\n"
            "forward_pure_python(x)\n"
            "summary()\n"
            "n_parameters",
            ha="center", va="center", fontsize=6.2, linespacing=1.45)
    users = ["module4.py\n(demo + checks)", "module4.ipynb\n(notebook demo)",
             "test_multilayer_\nperceptron.py (21)", "test_random.py\n(78 tests)", "smoke_test.py\n(8 checks)"]
    h = 0.15
    ys = np.linspace(0.80, 0.04, len(users))
    for u, y in zip(users, ys):
        box(ax, 0.70, y, 0.28, h, u, fc="white", fs=6.0)
        arrow(ax, 0.70, y + h / 2, 0.52, y + h / 2, None)
    ax.text(0.61, 0.965, "import", ha="center", va="center", fontsize=6.8, style="italic")
    save(fig, "fig_software")


# --------------------------------------------------------------------------- #
# Fig. 4  Activation functions
# --------------------------------------------------------------------------- #
def fig_activations():
    z = np.linspace(-8, 8, 400)
    fig, axs = plt.subplots(1, 2, figsize=(COL_W, 1.55))
    axs[0].plot(z, sigmoid(z), color="k")
    axs[0].axhline(0.5, color="#999999", lw=0.5, ls=":")
    axs[0].set_title("Sigmoid (layers 1--3)")
    axs[0].set_xlabel("$z$")
    axs[0].set_ylabel("$f(z)$")
    axs[1].plot(z, relu(z), color="k", ls="--")
    axs[1].set_title("ReLU (layer 4)")
    axs[1].set_xlabel("$z$")
    for a in axs:
        a.grid(alpha=0.25, lw=0.4)
    fig.tight_layout(pad=0.4)
    save(fig, "fig_activations")


# --------------------------------------------------------------------------- #
# Fig. 5  Activations after each layer
# --------------------------------------------------------------------------- #
def fig_layer_activations():
    _, trace = default_net.forward(X0, return_all=True)
    fig, axs = plt.subplots(1, 4, figsize=(COL_W, 1.45), sharey=True)
    hatches = ["", "", "", "//"]
    for ax, step in zip(axs, trace):
        v = step["a"]
        ax.bar(range(1, len(v) + 1), v, color="#7a7a7a" if step["layer"] < 4 else "#d0d0d0",
               edgecolor="k", linewidth=0.5, hatch=hatches[step["layer"] - 1])
        ax.set_title(f"L{step['layer']} ({'ReLU' if step['layer'] == 4 else 'Sig.'})", fontsize=7)
        ax.set_xticks(range(1, len(v) + 1))
        ax.set_xlabel("neuron", fontsize=7)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25, lw=0.4)
    axs[0].set_ylabel("activation $a^{(l)}$")
    fig.tight_layout(pad=0.3)
    save(fig, "fig_layer_activations")


# --------------------------------------------------------------------------- #
# Fig. 6  Input sweep: default vs scaled weights
# --------------------------------------------------------------------------- #
def fig_sweep():
    s = np.linspace(-6, 6, 241)
    X = np.column_stack([s, np.full_like(s, 0.5), np.full_like(s, -0.5)])
    fig, axs = plt.subplots(2, 1, figsize=(COL_W, 3.0), sharex=True)
    for ax, net, title in zip(axs, [default_net, demo_net],
                              ["(a) Default weights (Xavier, seed 42)", "(b) Scaled demonstration weights"]):
        Y = net(X)
        ax.plot(s, Y[:, 0], color="k", label="output 1")
        ax.plot(s, Y[:, 1], color="k", ls="--", label="output 2")
        ax.set_title(title, fontsize=7.5)
        ax.set_ylabel("network output")
        ax.grid(alpha=0.25, lw=0.4)
        ax.legend(loc="best", frameon=False)
    axs[0].set_ylim(0, 0.5)
    axs[1].set_xlabel("input $x_1$   ($x_2=0.5$, $x_3=-0.5$)")
    fig.tight_layout(pad=0.4)
    save(fig, "fig_sweep")
    return default_net(X), demo_net(X)


# --------------------------------------------------------------------------- #
# Fig. 7  Output surface of the scaled network
# --------------------------------------------------------------------------- #
def fig_surface():
    g = np.linspace(-4, 4, 120)
    G1, G2 = np.meshgrid(g, g)
    grid = np.column_stack([G1.ravel(), G2.ravel(), np.zeros(G1.size)])
    Z = demo_net(grid)[:, 0].reshape(G1.shape)
    fig, ax = plt.subplots(figsize=(COL_W, 2.45))
    cs = ax.contourf(G1, G2, Z, levels=14, cmap="Greys")
    ax.contour(G1, G2, Z, levels=14, colors="k", linewidths=0.25)
    cb = fig.colorbar(cs, ax=ax, pad=0.02)
    cb.set_label("output 1", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    ax.set_xlabel("input $x_1$")
    ax.set_ylabel("input $x_2$")
    fig.tight_layout(pad=0.4)
    save(fig, "fig_surface")


# --------------------------------------------------------------------------- #
# Fig. 8  Output distributions over random inputs
# --------------------------------------------------------------------------- #
def fig_hist():
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(5000, 3))
    Yd, Ys = default_net(X), demo_net(X)
    fig, axs = plt.subplots(1, 2, figsize=(COL_W, 1.6))
    axs[0].hist(Yd.ravel(), bins=40, color="#7a7a7a", edgecolor="k", linewidth=0.3)
    axs[0].set_title("(a) Default weights", fontsize=7.5)
    axs[1].hist(Ys.ravel(), bins=40, color="#bdbdbd", edgecolor="k", linewidth=0.3)
    axs[1].set_title("(b) Scaled weights", fontsize=7.5)
    for a in axs:
        a.set_xlabel("output value")
        a.grid(alpha=0.25, lw=0.4)
    axs[0].set_ylabel("count")
    fig.tight_layout(pad=0.4)
    save(fig, "fig_hist")
    return Yd, Ys


# --------------------------------------------------------------------------- #
# Generated LaTeX tables and number macros
# --------------------------------------------------------------------------- #
def fmt(v, d=4):
    return " ".join(f"{x:.{d}f}" for x in np.atleast_1d(v))



def write_rows(path, rows):
    """Write table rows separated by \\\\ but WITHOUT a trailing row terminator.

    The final ``\\\\`` is placed in report.tex right after ``\\input`` because a
    row terminator that ends an \\input file confuses booktabs' ``\\bottomrule``.
    """
    clean = []
    for r in rows:
        r = r.rstrip()
        if r.endswith("\\\\"):
            r = r[:-2].rstrip()
        clean.append(r)
    path.write_text(" \\\\\n".join(clean) + "\n")


def write_tables(sweep_default, sweep_demo, hist_default, hist_demo):
    # Table: architecture
    rows = []
    rows.append(r"Input & 3 & -- & -- & -- & 0 \\")
    for i, (W, b) in enumerate(zip(default_net.weights, default_net.biases), start=1):
        name = "Output (L4)" if i == 4 else f"Hidden {i} (L{i})"
        act = "ReLU" if i == 4 else "Sigmoid"
        rows.append(rf"{name} & {sizes[i]} & {act} & ${W.shape[0]}\times{W.shape[1]}$ & {b.shape[0]} & {W.size + b.size} \\")
    write_rows(GEN / "table_arch.tex", rows)

    # Table: example inputs and outputs
    examples = [("All zeros", [0, 0, 0]), ("All ones", [1, 1, 1]), ("Mixed", [0.5, -1, 2]),
                ("Negative", [-3, -2, -1]), ("Large", [50, -50, 100])]
    rows = []
    for name, x in examples:
        y = default_net.forward(x)
        xs = r",\ ".join(f"{v:g}" for v in x)
        rows.append(rf"{name} & $({xs})$ & {y[0]:.5f} & {y[1]:.5f} & {demo_net.forward(x)[0]:.3f} & {demo_net.forward(x)[1]:.3f} \\")
    write_rows(GEN / "table_examples.tex", rows)

    # Table: layer-by-layer trace (default network, mixed input)
    out, trace = default_net.forward(X0, return_all=True)
    rows = []
    for t in trace:
        act = "ReLU" if t["activation"] == "relu" else "Sigmoid"
        z = ",\\ ".join(f"{v:.3f}" for v in t["z"])
        a = ",\\ ".join(f"{v:.3f}" for v in t["a"])
        rows.append(rf"{t['layer']} & {act} & ${z}$ & ${a}$ \\")
    write_rows(GEN / "table_trace.tex", rows)

    # Hand calculation of neuron 1 in layer 1
    W1, b1 = default_net.weights[0], default_net.biases[0]
    z_manual = float(X0 @ W1[:, 0] + b1[0])
    a_manual = 1 / (1 + np.exp(-z_manual))

    # Random-trial verification
    rng = np.random.default_rng(2026)
    n_trials, ok, max_diff = 500, 0, 0.0
    for _ in range(n_trials):
        net = MultilayerPerceptron(seed=int(rng.integers(0, 10_000)))
        v = rng.normal(scale=10.0 ** rng.integers(-2, 3), size=3)
        o = net.forward(v)
        p = np.array(net.forward_pure_python(v))
        max_diff = max(max_diff, float(np.max(np.abs(o - p))))
        ok += bool(np.all(np.isfinite(o)) and np.all(o >= 0) and np.allclose(o, p))

    # Extreme-input stability
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        ext = default_net.forward([1e6, -1e6, 1e6])


    # Provable output bound: a3 lies in (0,1)^2, so y_j < max(0, sum_i max(W4_ij, 0) + b4_j)
    def bound(net):
        W4, b4 = net.weights[-1], net.biases[-1]
        return np.maximum(0.0, np.maximum(W4, 0).sum(axis=0) + b4)

    bd, bs = bound(default_net), bound(demo_net)
    assert np.all(hist_default.max(axis=0) <= bd + 1e-12) and np.all(hist_demo.max(axis=0) <= bs + 1e-12)

    # Exact-zero ReLU demonstration (same construction as the notebook)
    Wz = [np.ones((a, b)) * 0.1 for a, b in zip(sizes[:-1], sizes[1:])]
    Bz = [np.zeros(b) for b in sizes[1:]]
    Bz[-1] = np.array([-5.0, 0.0])
    zero_out = MultilayerPerceptron(weights=Wz, biases=Bz).forward([1, 1, 1])

    s_d, s_s = sweep_default, sweep_demo
    macros = {
        "ZManual": f"{z_manual:.5f}",
        "AManual": f"{a_manual:.5f}",
        "ClassA": f"{trace[0]['a'][0]:.5f}",
        "OutOne": f"{out[0]:.5f}",
        "OutTwo": f"{out[1]:.5f}",
        "RandOk": str(ok),
        "RandN": str(n_trials),
        "MaxDiff": f"{max_diff:.1e}".replace("e-", r"\times 10^{-") + "}",
        "DefaultMin": f"{min(s_d.min(), hist_default.min()):.3f}",
        "DefaultMax": f"{max(s_d.max(), hist_default.max()):.3f}",
        "DemoMin": f"{s_s.min():.2f}",
        "DemoMax": f"{s_s.max():.2f}",
        "HistDefaultMin": f"{hist_default.min():.3f}",
        "HistDefaultMax": f"{hist_default.max():.3f}",
        "HistDemoMin": f"{hist_demo.min():.2f}",
        "HistDemoMax": f"{hist_demo.max():.2f}",
        "ExtOne": f"{ext[0]:.5f}",
        "ExtTwo": f"{ext[1]:.5f}",
        "Params": str(default_net.n_parameters),
        "WOne": f"{W1[0, 0]:.5f}",
        "WTwo": f"{W1[1, 0]:.5f}",
        "WThree": f"{W1[2, 0]:.5f}",
        "BOne": f"{b1[0]:.5f}",
        "BoundDefaultOne": f"{bd[0]:.3f}",
        "BoundDefaultTwo": f"{bd[1]:.3f}",
        "BoundDemoOne": f"{bs[0]:.2f}",
        "BoundDemoTwo": f"{bs[1]:.2f}",
        "ZeroOne": f"{zero_out[0]:.1f}",
        "ZeroTwo": f"{zero_out[1]:.5f}",
    }
    lines = [rf"\newcommand{{\num{k}}}{{{v}}}" for k, v in macros.items()]
    (GEN / "numbers.tex").write_text("\n".join(lines) + "\n")
    print("numbers:", macros)


if __name__ == "__main__":
    fig_architecture()
    fig_flow()
    fig_software()
    fig_activations()
    fig_layer_activations()
    sd, ss = fig_sweep()
    fig_surface()
    hd, hs = fig_hist()
    write_tables(sd, ss, hd, hs)
    print("done")
