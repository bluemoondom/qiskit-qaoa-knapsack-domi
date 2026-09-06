"""Optional plotting.

All of these are optional -- :func:`~.pipeline.solve` and the building blocks
call them only when ``make_plots=True`` / ``draw_circuit=True``, but they can
also be called by hand. Matplotlib is imported inside the functions so that the
package can be used without a GUI.
"""

from __future__ import annotations

import os

import numpy as np

from .config import PlotConfig, RunConfig
from .problem import as_problem
from .results import Distribution

__all__ = [
    "show",
    "draw_circuit_diagram",
    "plot_layout",
    "plot_bitstrings",
    "plot_tail",
    "plot_results",
]


def show(fig):
    """Display a figure both in a notebook and in a script.

    ``get_ipython`` is injected only into the notebook's namespace, so inside a
    module it does not exist -- it has to be imported from IPython, otherwise
    the script branch would always be taken in Jupyter.
    """
    import matplotlib.pyplot as plt

    try:
        from IPython import get_ipython

        shell = get_ipython()
    except Exception:  # noqa: BLE001 - IPython need not be installed
        shell = None

    if shell is not None:
        from IPython.display import display

        display(fig)
        plt.close(fig)
    else:
        plt.show()


def draw_circuit_diagram(
    circuit, config: RunConfig | None = None, *, fold: int | None = None
):
    """Circuit diagram.

    The logical circuit is dense (all-to-all rzz from the penalty), so it is
    wide even for small N; the transpiled one has hundreds of gates and is
    wider still.
    """
    cfg = config or RunConfig()
    fold = cfg.fold if fold is None else fold
    try:
        fig = circuit.draw("mpl", fold=fold, idle_wires=False)
        show(fig)
        return fig
    except Exception as e:  # noqa: BLE001 - drawing is optional
        print(f" circuit diagram skipped ({e!r})")
        return None


def plot_layout(isa, backend, config: RunConfig | None = None):
    """Device-layout picture (coupling map) with the used qubits highlighted.

    ``layout_view="physical"`` -> the highlighted qubits are labeled with the
    SAME physical numbers the log prints, so the map and the log agree.
    """
    cfg = config or RunConfig()
    try:
        from qiskit.visualization import plot_circuit_layout

        fig = plot_circuit_layout(isa, backend, view=cfg.layout_view)
        fig.suptitle(f"{backend.name}: used qubits ({cfg.layout_view} numbering)")
        show(fig)
        return fig
    except Exception as e:  # noqa: BLE001
        print(f" layout plot skipped ({e!r})")
        return None


def plot_bitstrings(
    dist: Distribution,
    *,
    backend_name: str = "",
    plots: PlotConfig | None = None,
    top_k: int | None = None,
):
    """Bar chart of the most probable bitstrings.

    Red = optimum, blue = within (1-good) of the optimum, grey = the rest.

    Args:
        dist: the distribution from :func:`~.results.build_distribution`.
        backend_name: name shown in the title.
        plots: :class:`~.config.PlotConfig` (``top_k``, ``save_dir``, ``dpi``).
        top_k: how many of the most probable bitstrings to draw. Overrides
            ``plots.top_k`` for this call only, without mutating the config.
            Bars with zero probability are dropped, so fewer bars may appear.
    """
    import matplotlib.pyplot as plt

    pc = plots or PlotConfig()
    n_top = pc.top_k if top_k is None else int(top_k)
    if n_top < 1:
        raise ValueError(f"top_k must be >= 1, got {n_top}")
    prob = dist.prob
    order = np.argsort(-prob)[:n_top]
    order = order[prob[order] > 0]  # do not draw zero bars
    if not len(order):
        print(" bitstring plot skipped (no non-zero samples)")
        return None

    N = int(np.log2(len(prob)))
    fig_w = float(np.clip(0.30 * len(order), 14, 50))
    fig1, ax = plt.subplots(figsize=(fig_w, 10))
    col = [
        "#c0392b" if i in dist.optimal_masks else ("#2b7bba" if dist.good[i] else "#bbb")
        for i in order
    ]
    ax.bar(range(len(order)), prob[order], color=col, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(
        [format(i, f"0{N}b") for i in order],
        rotation=90,
        fontsize=11,
        fontweight="bold",
        fontfamily="monospace",
    )
    for lbl, c in zip(ax.get_xticklabels(), col, strict=False):
        lbl.set_color(c if c != "#bbb" else "#444")
    ax.tick_params(axis="x", length=0, pad=6)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.set_ylabel("probability", fontsize=20, fontweight="bold")

    # best feasible solution among those shown, plus its rank
    feas_in_view = [(r, i) for r, i in enumerate(order) if dist.feas[i]]
    if feas_in_view:
        best_rank, bi = max(feas_in_view, key=lambda t: dist.tot_v[t[1]])
        ax.annotate(
            f"best in top {len(order)}: {int(dist.tot_v[bi])} = "
            f"{100*dist.tot_v[bi]/dist.optimum:.1f}% of optimum (rank {best_rank+1})",
            xy=(best_rank, prob[bi]),
            xytext=(best_rank, prob[order].max() * 0.92),
            fontsize=18,
            fontweight="bold",
            ha="left",
            color="#c0392b",
            arrowprops=dict(arrowstyle="->", lw=2.5, color="#c0392b"),
        )
    ax.set_title(
        f"Top {len(order)} sampled bitstrings on {backend_name}  (N={N})\n"
        f"red = optimum, blue = within {100*(1-dist.good_threshold):.0f}% of optimum, "
        "grey = rest",
        fontsize=22,
        fontweight="bold",
    )
    plt.tight_layout()
    if pc.save_dir:
        os.makedirs(pc.save_dir, exist_ok=True)
        fig1.savefig(
            os.path.join(pc.save_dir, pc.bitstrings_filename),
            dpi=pc.dpi,
            bbox_inches="tight",
        )
    show(fig1)
    return fig1


def plot_tail(dist: Distribution, *, plots: PlotConfig | None = None):
    """Tail distribution: P(sample at least this good) vs uniform random.

    The gap between the curves IS the advantage.
    """
    import matplotlib.pyplot as plt

    pc = plots or PlotConfig()
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    vals_sorted = np.unique(dist.tot_v[dist.feas])[::-1]
    p_feas = dist.prob[dist.feas].sum()
    if p_feas <= 0:
        print(" tail plot skipped (P(feasible) = 0)")
        plt.close(fig2)
        return None
    cum_q = [dist.prob[dist.feas & (dist.tot_v >= v)].sum() / p_feas for v in vals_sorted]
    cum_r = [(dist.feas & (dist.tot_v >= v)).sum() / dist.feas.sum() for v in vals_sorted]
    ax2.plot(
        vals_sorted / dist.optimum,
        cum_q,
        lw=3.5,
        label="quantum annealing (post-selected)",
    )
    ax2.plot(
        vals_sorted / dist.optimum,
        cum_r,
        "k:",
        lw=3.5,
        label="uniform random (feasible)",
    )
    ax2.set_xlabel("value / optimum", fontsize=20, fontweight="bold")
    ax2.set_ylabel("P(sample at least this good)", fontsize=20, fontweight="bold")
    ax2.tick_params(labelsize=16)
    ax2.set_yscale("log")
    ax2.invert_xaxis()
    ax2.set_title(
        "Tail distribution\n(the gap between the curves IS the advantage)",
        fontsize=24,
        fontweight="bold",
    )
    ax2.legend(fontsize=18)
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    if pc.save_dir:
        os.makedirs(pc.save_dir, exist_ok=True)
        fig2.savefig(
            os.path.join(pc.save_dir, pc.tail_filename), dpi=pc.dpi, bbox_inches="tight"
        )
    show(fig2)
    return fig2


def plot_results(
    counts: dict[str, int],
    values,
    weights=None,
    capacity=None,
    *,
    backend_name: str = "",
    plots: PlotConfig | None = None,
    optimum: int | None = None,
    top_k: int | None = None,
    verbose: bool = True,
):
    """Both at once: build the distribution and draw both plots.

    Useful when the counts came from elsewhere (e.g. a saved job).

    Args:
        top_k: how many bitstrings to draw; overrides ``plots.top_k``.
    """
    from .results import build_distribution

    problem = as_problem(values, weights, capacity)
    pc = plots or PlotConfig()
    dist = build_distribution(
        counts, problem, good=pc.good, optimum=optimum, verbose=verbose
    )
    plot_bitstrings(dist, backend_name=backend_name, plots=pc, top_k=top_k)
    plot_tail(dist, plots=pc)
    return dist
