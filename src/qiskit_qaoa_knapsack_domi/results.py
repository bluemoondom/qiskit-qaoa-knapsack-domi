"""Decoding the measured bitstrings and the result tables."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .problem import KnapsackProblem, as_problem

__all__ = [
    "KnapsackResult",
    "Distribution",
    "decode_counts",
    "best_feasible",
    "build_result",
    "build_distribution",
    "print_summary",
]


@dataclass
class KnapsackResult:
    """The best feasible solution among the measured samples."""

    selection: list[int]
    """Indices of the selected items."""

    weight: int
    value: int
    optimum: int
    """The exact optimum from DP."""

    counts: dict[str, int] = field(default_factory=dict, repr=False)
    backend_name: str = ""
    method: str = ""
    n_qubits: int = 0
    twoq: int = 0
    shots: int = 0

    @property
    def ratio(self) -> float:
        """Ratio of the achieved value to the optimum (1.0 = optimum found)."""
        return self.value / self.optimum if self.optimum else 0.0

    @property
    def is_optimal(self) -> bool:
        return self.value == self.optimum


@dataclass
class Distribution:
    """Probability distribution over all 2^N combinations."""

    prob: np.ndarray
    """prob[i] = P(item combination i); bit j = item j."""

    tot_v: np.ndarray
    tot_w: np.ndarray
    feas: np.ndarray
    good: np.ndarray
    """Feasible and at the same time within (1-good_threshold) of the optimum."""

    optimal_masks: set[int]
    optimum: int
    good_threshold: float
    shots: int

    @property
    def p_feasible(self) -> float:
        return float(self.prob[self.feas].sum())

    @property
    def p_optimum(self) -> float:
        return float(self.prob[sorted(self.optimal_masks)].sum())

    @property
    def p_good(self) -> float:
        return float(self.prob[self.good].sum())

    def log(self, printer=print) -> None:
        """Print in the same format as the notebook."""
        printer(f"OPT = {self.optimum}   optimal bitstrings: {len(self.optimal_masks)}")
        printer(
            f"P(feasible) = {self.p_feasible:.4f}   "
            f"P(optimum)  = {self.p_optimum:.4f}   "
            f"P(within {100*(1-self.good_threshold):.0f}%) = {self.p_good:.4f}"
        )


def decode_counts(
    counts: dict[str, int],
    values,
    weights=None,
    capacity=None,
):
    """Decompose each bitstring into (item indices, weight, value).

    The bitstring follows the Qiskit convention (lowest qubit on the right).

    Yields:
        (bitstring, count, selection, weight, value)
    """
    problem: KnapsackProblem = as_problem(values, weights, capacity)
    n = problem.n
    for bitstring, c in counts.items():
        clean = bitstring.replace(" ", "")
        bits = [int(clean[len(clean) - 1 - j]) for j in range(n)]
        sel = [i for i in range(n) if bits[i] == 1]
        w, v = problem.evaluate(sel)
        yield bitstring, c, sel, w, v


def best_feasible(
    counts: dict[str, int],
    values,
    weights=None,
    capacity=None,
) -> tuple[list[int], int, int] | None:
    """The best feasible solution among the samples, or None."""
    problem: KnapsackProblem = as_problem(values, weights, capacity)
    best = None
    for _bs, _c, sel, w, v in decode_counts(counts, problem):
        if w <= problem.capacity and (best is None or v > best[2]):
            best = (sel, w, v)
    return best


def build_result(
    counts: dict[str, int],
    values,
    weights=None,
    capacity=None,
    *,
    backend=None,
    transpiled=None,
    n_qubits: int | None = None,
    optimum: int | None = None,
    verbose: bool = True,
) -> KnapsackResult:
    """Decode the counts into a :class:`KnapsackResult` and print the summary.

    This is what :func:`~.pipeline.solve` does after the run. Call it when
    going step by step so you get the same RESULT SUMMARY block; otherwise
    nothing prints it.

    Args:
        counts: the measured counts from :func:`~.runner.run_circuit`.
        values, weights, capacity: the instance, or a KnapsackProblem directly.
        backend: the backend it ran on; its name goes into the summary.
        transpiled: the :class:`~.transpiling.TranspileResult`; supplies the
            layout method, the two-qubit gate count and the qubit count.
        n_qubits: overrides the qubit count when no ``transpiled`` is given.
        optimum: the exact optimum; None = computed by DP.
        verbose: print the summary table (same format as the notebook).

    Returns:
        The best feasible solution among the samples. When there is none, the
        selection is empty and the summary says so.
    """
    problem: KnapsackProblem = as_problem(values, weights, capacity)
    best = best_feasible(counts, problem)
    sel, w, v = best if best else ([], 0, 0)

    result = KnapsackResult(
        selection=sel,
        weight=w,
        value=v,
        optimum=problem.optimum if optimum is None else optimum,
        counts=counts,
        backend_name=getattr(backend, "name", "") or "",
        method=getattr(transpiled, "method", "") or "",
        n_qubits=(
            n_qubits
            if n_qubits is not None
            # final_layout has one entry per LOGICAL qubit; transpiled.circuit
            # would give the whole chip (e.g. 156), which is not what we want
            else len(getattr(transpiled, "final_layout", None) or ()) or problem.n
        ),
        twoq=getattr(transpiled, "twoq", 0) or 0,
        shots=sum(counts.values()),
    )
    if verbose:
        print_summary(result, problem)
    return result


def build_distribution(
    counts: dict[str, int],
    values,
    weights=None,
    capacity=None,
    *,
    good: float = 0.90,
    optimum: int | None = None,
    verbose: bool = True,
) -> Distribution:
    """Build the probability distribution and the value/weight table for 2^N states.

    Args:
        counts: the measured counts.
        good: threshold for a "good" solution (0.90 = within 10 % of the optimum).
        optimum: the exact optimum; None = computed by DP.
        verbose: print P(feasible)/P(optimum)/P(within X%).
    """
    problem: KnapsackProblem = as_problem(values, weights, capacity)
    N = problem.n
    OPT = problem.optimum if optimum is None else optimum

    size = 1 << N
    shots_tot = sum(counts.values())
    prob = np.zeros(size)
    for bs, c in counts.items():
        # the lowest N bits are the items
        prob[int(bs.replace(" ", ""), 2) & (size - 1)] += c / shots_tot

    tot_v, tot_w = problem.enumerate_states()
    feas = tot_w <= problem.capacity
    dist = Distribution(
        prob=prob,
        tot_v=tot_v,
        tot_w=tot_w,
        feas=feas,
        good=feas & (tot_v >= good * OPT),
        optimal_masks=set(np.flatnonzero(feas & (tot_v == OPT)).tolist()),
        optimum=OPT,
        good_threshold=good,
        shots=shots_tot,
    )
    if verbose:
        dist.log()
    return dist


def print_summary(
    result: KnapsackResult,
    values,
    weights=None,
    capacity=None,
    printer=print,
) -> None:
    """Print the summary table in the same format as the notebook."""
    problem: KnapsackProblem = as_problem(values, weights, capacity)

    printer("\n" + "=" * 64)
    printer("                    KNAPSACK  -  RESULT SUMMARY")
    printer("=" * 64)
    printer(f" Backend : {result.backend_name:<24} Logical qubits : {result.n_qubits}")
    printer(" Method  : digitized quantum annealing (no optimizer)")
    printer(f" Layout  : {result.method}   |   2q gates : {result.twoq}")

    if not result.selection and result.value == 0:
        printer("-" * 64)
        printer(" No feasible solution among the samples (result dominated by noise).")
        printer("=" * 64)
        return

    printer("-" * 64)
    printer(" Selected items")
    printer("-" * 64)
    printer(f"   {'Item':>6} {'Value':>8} {'Weight':>8}")
    printer(f"   {'-'*6} {'-'*8} {'-'*8}")
    for i in result.selection:
        printer(f"   {i:>6} {problem.values[i]:>8} {problem.weights[i]:>8}")
    printer(f"   {'-'*6} {'-'*8} {'-'*8}")
    printer(
        f"   {'TOTAL':>6} {result.value:>8} {result.weight:>8}    "
        f"(capacity {problem.capacity})"
    )
    printer("-" * 64)
    printer(" Solution quality")
    printer("-" * 64)
    printer(f"   {'Method':<22} {'Value':>7} {'% of optimum':>14}")
    printer(f"   {'-'*22} {'-'*7} {'-'*14}")
    printer(
        f"   {'Quantum (hardware)':<22} {result.value:>7} {100*result.ratio:>13.1f}%"
    )
    printer(f"   {'Exact (dynamic prog.)':<22} {result.optimum:>7} {'-':>14}")
    printer("=" * 64)
