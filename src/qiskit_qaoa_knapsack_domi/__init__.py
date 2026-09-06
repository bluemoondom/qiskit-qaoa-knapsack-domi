"""qiskit-qaoa-knapsack-domi.

Solving the 0/1 knapsack problem by digitized quantum annealing on IBM Quantum.

Only ``values``, ``weights`` and ``capacity`` are passed into the module; N is
derived from their count. Everything else is determined by
:func:`tune_penalties`.

Quick start::

    from qiskit_qaoa_knapsack_domi import KnapsackProblem, RunConfig, solve

    problem = KnapsackProblem.random(15, seed=38)
    session = solve(problem, config=RunConfig(dry_run=True, prune_keep=0.5))
    print(session.result.value, "/", session.result.optimum)

Step by step::

    from qiskit_qaoa_knapsack_domi import (
        KnapsackProblem, RunConfig, tune_penalties,
        build_annealing_circuit, select_backend, transpile_best,
        run_circuit, build_distribution, print_summary,
    )
"""

from .backends import error_score, select_backend, two_qubit_gates, twoq_count
from .config import Penalties, PlotConfig, RunConfig
from .pipeline import Session, build_annealing_circuit, solve
from .plotting import (
    draw_circuit_diagram,
    plot_bitstrings,
    plot_layout,
    plot_results,
    plot_tail,
)
from .problem import KnapsackProblem, as_problem, dp_optimum
from .qubo import (
    IsingModel,
    Qubo,
    build_circuit,
    build_qubo,
    check_ground_state,
    prune_couplings,
    qubo_to_ising,
)
from .results import (
    Distribution,
    KnapsackResult,
    best_feasible,
    build_distribution,
    build_result,
    decode_counts,
    print_summary,
)
from .runner import monitor_job, run_circuit
from .transpiling import TranspileResult, mapomatic_candidate, seed_scan, transpile_best
from .tuning import lp_dual, tune_penalties

__version__ = "0.3.0"

__all__ = [
    "__version__",
    # instance
    "KnapsackProblem",
    "as_problem",
    "dp_optimum",
    # configuration
    "RunConfig",
    "PlotConfig",
    "Penalties",
    # penalty tuning
    "tune_penalties",
    "lp_dual",
    # QUBO / circuit
    "Qubo",
    "IsingModel",
    "build_qubo",
    "prune_couplings",
    "check_ground_state",
    "qubo_to_ising",
    "build_circuit",
    "build_annealing_circuit",
    # backend + transpilation
    "select_backend",
    "two_qubit_gates",
    "error_score",
    "twoq_count",
    "seed_scan",
    "mapomatic_candidate",
    "transpile_best",
    "TranspileResult",
    # running
    "run_circuit",
    "monitor_job",
    # results
    "KnapsackResult",
    "Distribution",
    "decode_counts",
    "best_feasible",
    "build_distribution",
    "build_result",
    "print_summary",
    # plots (optional)
    "draw_circuit_diagram",
    "plot_layout",
    "plot_bitstrings",
    "plot_tail",
    "plot_results",
    # pipeline
    "solve",
    "Session",
]
