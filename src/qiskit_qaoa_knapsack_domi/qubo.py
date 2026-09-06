"""QUBO -> Ising -> digitized quantum annealing circuit.

This covers steps 1-3 from the notebook: building the QUBO from the penalties,
pruning the weakest ZZ couplings (``prune_couplings``), checking the ground
state, converting to Ising and building the circuit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from qiskit.circuit import QuantumCircuit

from .config import Penalties
from .problem import KnapsackProblem, as_problem

__all__ = [
    "Qubo",
    "IsingModel",
    "build_qubo",
    "prune_couplings",
    "check_ground_state",
    "qubo_to_ising",
    "build_circuit",
]


@dataclass
class Qubo:
    """QUBO cost function coefficients."""

    lin: dict[int, float]
    quad: dict[tuple[int, int], float]
    n: int
    raw_couplings: int = 0
    """How many ZZ couplings there were before pruning."""

    @property
    def kept_couplings(self) -> int:
        return len(self.quad)

    @property
    def keep_ratio(self) -> float:
        return len(self.quad) / self.raw_couplings if self.raw_couplings else 1.0


@dataclass
class IsingModel:
    """Ising model after normalizing to a maximum coefficient of 1."""

    h: dict[int, float]
    J: dict[tuple[int, int], float]
    n: int
    scale: float = 1.0
    extra: dict = field(default_factory=dict, repr=False)


def prune_couplings(
    raw: dict[tuple[int, int], float],
    prune_keep: float = 1.0,
    *,
    verbose: bool = True,
) -> dict[tuple[int, int], float]:
    """Keep only the ``prune_keep`` fraction of the strongest couplings.

    Couplings with a small |coefficient| cost just as many two-qubit gates as
    strong ones but barely affect the result. Pruning is the main lever on
    circuit depth.

    Args:
        raw: all quadratic coefficients {(i, j): coeff}.
        prune_keep: fraction of couplings to keep, 1.0 = prune nothing.
        verbose: print the "ZZ gates: stayed ..." line like the notebook.

    Returns:
        The pruned coupling dictionary.
    """
    if not 0.0 < prune_keep <= 1.0:
        raise ValueError(f"prune_keep must be in (0, 1], got {prune_keep}")
    if prune_keep >= 1.0 or not raw:
        quad = dict(raw)
    else:
        mags = sorted(abs(v) for v in raw.values())
        cut = mags[int((1 - prune_keep) * len(mags))]
        quad = {k: v for k, v in raw.items() if abs(v) >= cut}
    if verbose and raw:
        print(f"ZZ gates: stayed {len(quad)}/{len(raw)} ({100*len(quad)/len(raw):.0f}%)")
    return quad


def build_qubo(
    values,
    weights=None,
    capacity=None,
    *,
    penalties: Penalties | None = None,
    prune_keep: float = 1.0,
    verbose: bool = True,
) -> Qubo:
    """Build the QUBO by hand from ALPHA/LAM1/LAM2 (no slack variables).

    Cost function::

        E(x) = -ALPHA * sum v_i x_i
               + LAM1 * sum w_i x_i
               + LAM2 * (sum w_i x_i - CAPACITY)^2

    The quadratic term expands into linear contributions and ZZ couplings; the
    latter are then pruned according to ``prune_keep``.

    Args:
        values, weights, capacity: the instance (or a KnapsackProblem directly).
        penalties: output of :func:`~.tuning.tune_penalties`.
        prune_keep: PRUNE_KEEP -- fraction of ZZ couplings to keep.
        verbose: print the pruning statistics.
    """
    problem: KnapsackProblem = as_problem(values, weights, capacity)
    p = penalties or Penalties()
    n = problem.n
    N = problem.n
    CAPACITY = problem.capacity

    lin = {j: 0.0 for j in range(n)}
    for i in range(N):
        lin[i] += -p.ALPHA * problem.values[i]
        lin[i] += p.LAM1 * problem.weights[i]
        lin[i] += p.LAM2 * (problem.weights[i] ** 2 - 2 * CAPACITY * problem.weights[i])

    raw = {}
    for i in range(N):
        for j in range(i + 1, N):
            raw[(i, j)] = 2 * p.LAM2 * problem.weights[i] * problem.weights[j]

    quad = prune_couplings(raw, prune_keep, verbose=verbose)
    return Qubo(lin=lin, quad=quad, n=n, raw_couplings=len(raw))


def check_ground_state(
    values,
    weights=None,
    capacity=None,
    *,
    penalties: Penalties | None = None,
    max_exact_n: int = 24,
    verbose: bool = True,
    strict: bool = True,
) -> dict:
    """Verify that the global minimum of the cost function is a feasible solution.

    This walks all 2^N states, so it is skipped above ``max_exact_n``.

    Raises:
        ValueError: when the minimum is infeasible (weak penalty -> raise LAM1).
    """
    problem: KnapsackProblem = as_problem(values, weights, capacity)
    p = penalties or Penalties()
    if problem.n > max_exact_n:
        if verbose:
            print(
                f"ground state check skipped (N={problem.n} > max_exact_n="
                f"{max_exact_n}, would cost 2^N)"
            )
        return {"checked": False}

    tot_v, tot_w = problem.enumerate_states()
    E = -p.ALPHA * tot_v + p.LAM1 * tot_w + p.LAM2 * (tot_w - problem.capacity) ** 2
    gs = int(np.argmin(E))
    feasible = bool(tot_w[gs] <= problem.capacity)
    if verbose:
        print(
            f"ground state of the cost: value={tot_v[gs]} weight={tot_w[gs]} "
            f"(cap {problem.capacity}) feasible={feasible}"
        )
    if strict and not feasible:
        raise ValueError("penalty is weak - minimum is unacceptable, increase LAM1")
    return {
        "checked": True,
        "state": gs,
        "value": int(tot_v[gs]),
        "weight": int(tot_w[gs]),
        "feasible": feasible,
    }


def qubo_to_ising(qubo: Qubo) -> IsingModel:
    """Convert QUBO -> Ising by substituting x = (1 - Z)/2, then normalize."""
    n = qubo.n
    h = {j: 0.0 for j in range(n)}
    J: dict[tuple[int, int], float] = {}
    for j in range(n):
        h[j] += -qubo.lin[j] / 2
    for (j, l), q in qubo.quad.items():
        h[j] += -q / 4
        h[l] += -q / 4
        J[(j, l)] = q / 4
    scale = max(
        max(abs(v) for v in h.values()),
        max((abs(v) for v in J.values()), default=0.0),
    )
    if scale <= 0:
        raise ValueError("degenerate Ising: all coefficients are zero")
    h = {j: v / scale for j, v in h.items()}
    J = {k: v / scale for k, v in J.items()}
    if not all(0 <= j < n and 0 <= l < n for (j, l) in J):
        raise ValueError("coupling index out of qubit range")
    return IsingModel(h=h, J=J, n=n, scale=scale)


def build_circuit(
    ising: IsingModel,
    penalties: Penalties | None = None,
    *,
    steps: int | None = None,
    t: float | None = None,
    measure: bool = True,
    verbose: bool = True,
) -> QuantumCircuit:
    """Digitized quantum annealing circuit (no classical optimizer).

    Trotterization with a midpoint so that the mixer never drops to zero::

        s = (p + 0.5) / STEPS

    Args:
        ising: the normalized Ising model.
        penalties: STEPS and T are taken from here unless given explicitly.
        steps: STEPS -- number of annealing layers (more steps = more decoherence).
        t: T -- total annealing time.
        measure: append ``measure_all``.
        verbose: print the qubit count and depth.
    """
    p = penalties or Penalties()
    steps = int(steps if steps is not None else p.STEPS)
    t = float(t if t is not None else p.T)
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")

    n = ising.n
    dt = t / steps
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for step_i in range(steps):
        s = (step_i + 0.5) / steps  # midpoint - the mixer never drops to zero
        t_prob, t_mix = dt * s, dt * (1 - s)
        for j in range(n):
            if ising.h[j]:
                qc.rz(2 * t_prob * ising.h[j], j)
        for (j, l), Jjl in ising.J.items():
            if Jjl:
                qc.rzz(2 * t_prob * Jjl, j, l)
        for j in range(n):
            qc.rx(-2 * t_mix, j)
    if measure:
        qc.measure_all()
    if verbose:
        print(f"\nLogical qubits (incl. slack): {n} | raw depth: {qc.depth()}")
    return qc
