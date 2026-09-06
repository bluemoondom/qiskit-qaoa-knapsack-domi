"""The whole pipeline in one function.

``solve()`` does the same as the notebook from the SETTINGS cell through to the
result plots, with the same log output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qiskit.circuit import QuantumCircuit

from .backends import select_backend
from .config import Penalties, RunConfig
from .problem import KnapsackProblem, as_problem
from .qubo import (
    IsingModel,
    Qubo,
    build_circuit,
    build_qubo,
    check_ground_state,
    qubo_to_ising,
)
from .results import (
    Distribution,
    KnapsackResult,
    build_distribution,
    build_result,
)
from .runner import run_circuit
from .transpiling import TranspileResult, transpile_best
from .tuning import tune_penalties

__all__ = ["Session", "solve", "build_annealing_circuit"]


@dataclass
class Session:
    """Everything produced during the run -- for further inspection afterwards."""

    problem: KnapsackProblem
    config: RunConfig
    penalties: Penalties
    qubo: Qubo | None = None
    ising: IsingModel | None = None
    circuit: QuantumCircuit | None = None
    backend: Any = None
    transpiled: TranspileResult | None = None
    counts: dict[str, int] = field(default_factory=dict, repr=False)
    result: KnapsackResult | None = None
    distribution: Distribution | None = None


def build_annealing_circuit(
    values,
    weights=None,
    capacity=None,
    *,
    penalties: Penalties | None = None,
    config: RunConfig | None = None,
) -> tuple[QuantumCircuit, Qubo, IsingModel]:
    """QUBO -> pruning -> ground-state check -> Ising -> circuit.

    Usable on its own when you want the circuit without running it on hardware.
    Draws the circuit diagram when ``draw_circuit=True``.
    """
    problem = as_problem(values, weights, capacity)
    cfg = config or RunConfig()
    p = penalties or Penalties()

    qubo = build_qubo(problem, penalties=p, prune_keep=cfg.prune_keep, verbose=cfg.verbose)
    check_ground_state(
        problem, penalties=p, max_exact_n=cfg.max_exact_n, verbose=cfg.verbose
    )
    ising = qubo_to_ising(qubo)
    qc = build_circuit(ising, p, verbose=cfg.verbose)

    if cfg.draw_circuit:
        from .plotting import draw_circuit_diagram

        draw_circuit_diagram(qc, cfg)

    return qc, qubo, ising


def solve(
    values,
    weights=None,
    capacity=None,
    *,
    config: RunConfig | None = None,
    penalties: Penalties | None = None,
    tune: bool | None = None,
    tune_kwargs: dict | None = None,
) -> Session:
    """Solve the knapsack by digitized quantum annealing on IBM hardware.

    Steps: tune the penalties -> QUBO (with pruning) -> Ising -> circuit ->
    noise-aware transpilation -> run on the Sampler -> decode and plot.

    Args:
        values, weights, capacity: the instance, or a :class:`KnapsackProblem`
            directly. N is derived from ``len(values)``.
        config: run settings (:class:`RunConfig`). None = defaults.
        penalties: ready-made ALPHA/LAM1/LAM2/STEPS/T. When given, tuning is
            skipped.
        tune: force tuning on/off. None = tune only when ``penalties`` is missing.
        tune_kwargs: arguments for :func:`~.tuning.tune_penalties`, e.g.
            ``{"schedule_grid": [(3, 6.0), (5, 10.0)], "max_rzz": 500}``.

    Returns:
        A :class:`Session` with everything produced (circuit, backend, counts,
        result).

    Example:
        >>> from qiskit_qaoa_knapsack_domi import KnapsackProblem, RunConfig, solve
        >>> problem = KnapsackProblem.random(15, seed=38)
        >>> session = solve(problem, config=RunConfig(dry_run=True))  # doctest: +SKIP
    """
    problem = as_problem(values, weights, capacity)
    cfg = config or RunConfig()

    if cfg.verbose:
        problem.log()

    # --- 1) penalties ------------------------------------------------------
    do_tune = tune if tune is not None else penalties is None
    if do_tune:
        tk = {"verbose": cfg.verbose, **(tune_kwargs or {})}
        penalties = tune_penalties(problem, **tk)
    elif penalties is None:
        penalties = Penalties()

    session = Session(problem=problem, config=cfg, penalties=penalties)

    # --- 2) circuit (drawing is handled by build_annealing_circuit) --------
    qc, qubo, ising = build_annealing_circuit(problem, penalties=penalties, config=cfg)
    session.qubo, session.ising, session.circuit = qubo, ising, qc

    # --- 3) backend + transpilation (layout and ISA diagram drawn there) ---
    backend = select_backend(cfg, num_qubits=qubo.n)
    session.backend = backend

    tr = transpile_best(qc, backend, cfg, num_items=problem.n)
    session.transpiled = tr

    # --- 4) run ------------------------------------------------------------
    counts = run_circuit(tr.circuit, backend, cfg, penalties)
    session.counts = counts

    # --- 5) decode + exact DP ---------------------------------------------
    optimum = problem.optimum
    result = build_result(
        counts,
        problem,
        backend=backend,
        transpiled=tr,
        n_qubits=qubo.n,
        optimum=optimum,
        verbose=cfg.verbose,
    )
    session.result = result

    # --- 6) result plots ---------------------------------------------------
    session.distribution = build_distribution(
        counts,
        problem,
        good=cfg.plots.good,
        optimum=optimum,
        verbose=cfg.verbose,
    )
    if cfg.make_plots:
        from .plotting import plot_bitstrings, plot_tail

        plot_bitstrings(session.distribution, backend_name=backend.name, plots=cfg.plots)
        plot_tail(session.distribution, plots=cfg.plots)

    return session
