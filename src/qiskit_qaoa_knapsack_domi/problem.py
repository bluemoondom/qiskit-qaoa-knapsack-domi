"""Knapsack problem instance.

Only ``values``, ``weights`` and ``capacity`` are passed into the module;
N is derived from their count (``len(values)``).
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["KnapsackProblem", "dp_optimum", "as_problem"]


def dp_optimum(values: Sequence[int], weights: Sequence[int], capacity: int) -> int:
    """Exact 0/1 knapsack optimum by dynamic programming. O(N * capacity)."""
    cap = int(capacity)
    dp = [0] * (cap + 1)
    for v, w in zip(values, weights, strict=True):
        w = int(w)
        v = int(v)
        for x in range(cap, w - 1, -1):
            dp[x] = max(dp[x], dp[x - w] + v)
    return dp[cap]


@dataclass(frozen=True)
class KnapsackProblem:
    """A knapsack instance.

    Args:
        values: item values.
        weights: item weights (same count as values).
        capacity: knapsack capacity.

    Example:
        >>> p = KnapsackProblem([10, 20, 30], [5, 4, 6], capacity=9)
        >>> p.n
        3
        >>> p.optimum
        50
    """

    values: tuple[int, ...]
    weights: tuple[int, ...]
    capacity: int

    def __init__(
        self,
        values: Iterable[int],
        weights: Iterable[int],
        capacity: int | None = None,
        *,
        capacity_divisor: int = 3,
    ) -> None:
        v = tuple(int(x) for x in values)
        w = tuple(int(x) for x in weights)
        if len(v) != len(w):
            raise ValueError(
                f"values and weights must have the same length ({len(v)} != {len(w)})"
            )
        if not v:
            raise ValueError("empty instance: values is empty")
        if any(x <= 0 for x in w):
            raise ValueError("all weights must be positive")
        if capacity is None:
            capacity = sum(w) // capacity_divisor
        capacity = int(capacity)
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if capacity >= sum(w):
            raise ValueError(
                f"capacity={capacity} >= total weight {sum(w)}: the instance is "
                "trivial (everything fits)"
            )
        object.__setattr__(self, "values", v)
        object.__setattr__(self, "weights", w)
        object.__setattr__(self, "capacity", capacity)

    # --- derived quantities ------------------------------------------------
    @property
    def n(self) -> int:
        """N -- number of items = number of logical qubits."""
        return len(self.values)

    # alias so that problem.N reads the same as in the notebook
    @property
    def N(self) -> int:  # noqa: N802 - upper-case on purpose, matches the notebook
        return len(self.values)

    @property
    def CAPACITY(self) -> int:  # noqa: N802
        return self.capacity

    @property
    def optimum(self) -> int:
        """Exact optimum (DP)."""
        return dp_optimum(self.values, self.weights, self.capacity)

    @property
    def num_rzz_full(self) -> int:
        """Number of RZZ per layer with full all-to-all coupling."""
        return self.n * (self.n - 1) // 2

    # --- constructors ------------------------------------------------------
    @classmethod
    def random(
        cls,
        n: int,
        *,
        seed: int | None = None,
        value_range: tuple[int, int] = (5, 60),
        weight_range: tuple[int, int] = (1, 20),
        capacity: int | None = None,
        capacity_divisor: int = 3,
    ) -> KnapsackProblem:
        """Random instance.

        Generates exactly the same sequence as this code::

            random.seed(seed)
            values  = [random.randint(5, 60) for _ in range(n)]
            weights = [random.randint(1, 20) for _ in range(n)]

        Args:
            n: number of items (N).
            seed: generator seed; the same seed gives the same instance.
            value_range: value range, both bounds inclusive.
            weight_range: weight range, both bounds inclusive.
            capacity: capacity set by hand. When omitted,
                ``sum(weights) // capacity_divisor`` is used.
            capacity_divisor: divisor for the automatic capacity (3 = a third).
        """
        rng = random.Random(seed)
        values = [rng.randint(*value_range) for _ in range(n)]
        weights = [rng.randint(*weight_range) for _ in range(n)]
        return cls(values, weights, capacity, capacity_divisor=capacity_divisor)

    # --- helpers -----------------------------------------------------------
    def evaluate(self, selection: Iterable[int]) -> tuple[int, int]:
        """(total weight, total value) for a list of item indices."""
        sel = list(selection)
        return (
            sum(self.weights[i] for i in sel),
            sum(self.values[i] for i in sel),
        )

    def is_feasible(self, selection: Iterable[int]) -> bool:
        return self.evaluate(selection)[0] <= self.capacity

    def enumerate_states(self) -> tuple[np.ndarray, np.ndarray]:
        """Value and weight of all 2^N combinations. Note: exponential."""
        v = np.asarray(self.values)
        w = np.asarray(self.weights)
        states = np.arange(1 << self.n)
        bits = (states[:, None] >> np.arange(self.n)) & 1
        return bits @ v, bits @ w

    def log(self, printer=print) -> None:
        """Print the instance in the same format as the notebook."""
        printer(f"values : {list(self.values)}")
        printer(f"weights: {list(self.weights)}")
        printer(f"capacity: {self.capacity}")


def as_problem(values, weights=None, capacity=None, **kwargs) -> KnapsackProblem:
    """Accept either a ready :class:`KnapsackProblem` or the three arrays.

    This lets every public function be called both ways::

        tune_penalties(problem)
        tune_penalties(values, weights, capacity)
    """
    if isinstance(values, KnapsackProblem):
        if weights is not None or capacity is not None:
            raise TypeError(
                "when the first argument is a KnapsackProblem, "
                "weights/capacity must not be given"
            )
        return values
    if weights is None:
        raise TypeError("missing weights")
    return KnapsackProblem(values, weights, capacity, **kwargs)
