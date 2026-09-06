"""Penalty tuning.

This module contains a single main function, :func:`tune_penalties` -- in the
notebook it was called ``tune_penalties2``; the slower original version is not
carried over. The function determines ALPHA, LAM1, LAM2, STEPS, T and the
recommended SHOTS.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .config import Penalties
from .problem import KnapsackProblem, as_problem

__all__ = ["tune_penalties", "lp_dual"]


def lp_dual(values, weights=None, capacity=None) -> float:
    """Shadow price of capacity = density of the break item (LP relaxation).

    O(N log N).
    """
    p = as_problem(values, weights, capacity)
    v = np.asarray(p.values, float)
    w = np.asarray(p.weights, float)
    cum = 0.0
    for i in np.argsort(-(v / w)):
        if cum + w[i] <= p.capacity:
            cum += w[i]
        else:
            return float(v[i] / w[i])
    return 0.0


def tune_penalties(
    values,
    weights=None,
    capacity=None,
    *,
    STEPS: int = 5,
    T: float = 6.0,
    schedule_grid: Sequence[tuple[int, float]] | None = None,
    l1_range: tuple[float, float] | None = None,
    l2_range: tuple[float, float] = (0.1, 5.0),
    n_l2: int = 28,
    max_rzz: int | None = None,
    step: float = 0.5,
    n_coarse: int = 25,
    n_final: int = 8,
    alpha_out: float = 2.0,
    gate_error: float | None = None,
    twoq_overhead: float = 2.9,
    plateau: float = 0.90,
    dtype=np.complex64,
    auto_window: tuple[float, float] = (0.7, 1.7),
    verbose: bool = True,
) -> Penalties:
    """Find ALPHA/LAM1/LAM2 (and optionally STEPS/T) for the knapsack QUBO.

    Scoring is ALWAYS P(optimum) from a statevector simulation of the very
    circuit that will run. No proxy quantity.

    Three speedups over the naive version:
      1) the hard condition is tested on a Pareto reduction (a few hundred
         candidates instead of 2^N) - E_cls depends on a state only through
         the pair (V, W)
      2) the Ising energy is derived affinely from E_cls; the bits matrix,
         the Z matrix and the loop over N(N-1)/2 pairs all disappear
      3) in-place mixer + complex64

    Procedure:
      A) coarse grid: discard pairs where the global minimum of the cost
         function is NOT the DP optimum; simulate the rest with the cheapest
         schedule
      B) refine around the n_coarse best points with step/5
      C) final: all schedules for the n_final best candidates only, plus the
         plateau (parameter range down to `plateau` times the best P(opt))

    Args:
        values: item values, or a :class:`KnapsackProblem` directly.
        weights: item weights (when no KnapsackProblem was given).
        capacity: knapsack capacity (when no KnapsackProblem was given).
        STEPS, T: used when ``schedule_grid`` is not given.
        schedule_grid: list of (STEPS, T) pairs to search,
            e.g. ``[(3, 6.0), (5, 10.0), (7, 15.0)]``.
        l1_range: LAM1 range in units of ALPHA=``alpha_out``. ``None`` turns on
            an automatic window around the LP dual (the shadow price). Measured
            on N=8..15: the winning LAM1/ALPHA always fell within 1.09-1.38x
            the LP dual.
        l2_range: LAM2 range; traversed LOGARITHMICALLY (``n_l2`` points),
            because the optimum often sits just above the validity floor.
        max_rzz: cap on the number of logical RZZ; schedules above it are
            discarded.
        gate_error: error per two-qubit gate. When given, P(opt) is scored
            AFTER the attenuation F = exp(-gate_error * twoq_overhead * RZZ),
            so a depth the hardware cannot sustain is not selected.
        plateau: threshold for reporting the plateau (0.90 = down to 90 % of
            the best score).
        verbose: print the table and summary (same format as the notebook).

    Returns:
        :class:`~qiskit_qaoa_knapsack_domi.config.Penalties` -- ALPHA, LAM1,
        LAM2, STEPS, T plus diagnostics. The object can also be indexed like a
        dictionary (``cfg["LAM1"]``), so notebook code works unchanged.

    Raises:
        ValueError: N is too large, or no schedule fits within ``max_rzz``.
        RuntimeError: no pair in the given ranges makes the minimum of the cost
            function the DP optimum.
    """
    problem: KnapsackProblem = as_problem(values, weights, capacity)
    v = np.asarray(problem.values, float)
    w = np.asarray(problem.weights, float)
    N = problem.n
    C = float(problem.capacity)

    if N > 28:
        raise ValueError(
            f"N={N}: the statevector simulation is 2^N amplitudes; above ~28 "
            "qubits penalties cannot be tuned this way"
        )

    vi = v.astype(np.int64)
    wi = w.astype(np.int64)
    Wtot, Vtot = int(wi.sum()), int(vi.sum())
    U = 1.0 / (1 << N)

    # ---- 1) reach table + Pareto reduction (cheap, independent of 2^N) ----
    reach = np.zeros((Wtot + 1, Vtot + 1), bool)
    reach[0, 0] = True
    for i in range(N):
        reach[wi[i] :, vi[i] :] |= reach[: Wtot + 1 - wi[i], : Vtot + 1 - vi[i]]

    dp = [0] * (int(C) + 1)
    for i in range(N):
        for x in range(int(C), int(wi[i]) - 1, -1):
            dp[x] = max(dp[x], dp[x - int(wi[i])] + int(vi[i]))
    OPT = dp[int(C)]

    cands, opts = [], []
    for W_ in range(Wtot + 1):
        idx = np.flatnonzero(reach[W_])
        if not len(idx):
            continue
        if W_ <= C and idx[-1] == OPT:
            opts.append((OPT, W_))
            if len(idx) > 1:
                cands.append((idx[-2], W_))  # best NON-optimal at that weight
        else:
            cands.append((idx[-1], W_))  # for a given W the largest V suffices
    cands = np.array(cands, float)
    opts = np.array(opts, float)
    cV, cW = cands[:, 0], cands[:, 1]
    oV, oW = opts[:, 0], opts[:, 1]

    def valid(l1: float, l2: float) -> bool:
        """Hard condition: minimum of the cost function = DP optimum.

        O(number of weights).
        """
        e_opt = (-oV + l1 * oW + l2 * (oW - C) ** 2).max()
        return bool((-cV + l1 * cW + l2 * (cW - C) ** 2).min() > e_opt)

    # ---- state arrays: only V and W, no bits matrix ----------------------
    V = np.zeros(1)
    W = np.zeros(1)
    for i in range(N):
        V = np.concatenate([V, V + v[i]])
        W = np.concatenate([W, W + w[i]])
    feas = W <= C
    pen = (W - C) ** 2
    opt_idx = np.flatnonzero(feas & (V == OPT))
    W_sum = w.sum()
    ws = np.sort(w)[::-1]
    sq = W_sum**2 - (w**2).sum()  # = 2 * sum_{i<j} w_i w_j

    # ---- 2) Ising derived affinely from E_cls ---------------------------
    def ising(l1: float, l2: float):
        lin = -v + l1 * w + l2 * (w**2 - 2 * C * w)
        h = -lin / 2.0 - (l2 / 2.0) * w * (W_sum - w)
        s = max(np.abs(h).max(), (l2 / 2.0) * ws[0] * ws[1])
        const = 0.5 * lin.sum() + 0.25 * l2 * sq
        E = (-V + l1 * W + l2 * pen - l2 * C**2 - const) / s
        return E.astype(np.float32), s

    # ---- 3) simulation: in-place mixer -----------------------------------
    def simulate(E, nsteps: int, ttot: float):
        dt = ttot / nsteps
        psi = np.full(1 << N, 2.0 ** (-N / 2), dtype=dtype)
        ph = np.empty_like(psi)
        for p in range(nsteps):
            s = (p + 0.5) / nsteps  # midpoint: the mixer never vanishes
            np.exp((-1j * dt * s) * E, out=ph, casting="unsafe")
            psi *= ph
            be = dt * (1.0 - s)
            c, sn = np.cos(be), 1j * np.sin(be)
            for q in range(N):
                t = psi.reshape(-1, 2, 1 << q)
                a = t[:, 0, :].copy()
                b = t[:, 1, :]
                t[:, 0, :] = c * a + sn * b
                t[:, 1, :] = sn * a + c * b
        return np.abs(psi) ** 2

    # ---- schedules -------------------------------------------------------
    scheds = list(schedule_grid) if schedule_grid else [(STEPS, T)]
    rzz_layer = N * (N - 1) // 2
    if max_rzz is not None:
        scheds = [s for s in scheds if s[0] * rzz_layer <= max_rzz]
        if not scheds:
            raise ValueError(
                f"nothing fits within max_rzz={max_rzz} ({rzz_layer} RZZ/layer)"
            )
    cheap = min(scheds, key=lambda s: s[0])  # phases A/B run the cheapest schedule

    # ---- window for LAM1 -------------------------------------------------
    dual = lp_dual(problem)
    if l1_range is None:
        l1_range = (
            auto_window[0] * dual * alpha_out,
            auto_window[1] * dual * alpha_out,
        )
        if verbose:
            print(
                f"      LP dual = {dual:.3f}  ->  LAM1 window = "
                f"{l1_range[0]:.2f}..{l1_range[1]:.2f} (at ALPHA={alpha_out})"
            )

    lo1, hi1 = l1_range[0] / alpha_out, l1_range[1] / alpha_out
    lo2, hi2 = max(l2_range[0], 1e-3) / alpha_out, l2_range[1] / alpha_out
    h1 = step / alpha_out
    L2GRID = np.geomspace(lo2, hi2, n_l2)

    # ---- A) coarse grid --------------------------------------------------
    coarse = []
    for l1 in np.arange(lo1, hi1 + 1e-9, h1):
        for l2 in L2GRID:
            if not valid(l1, l2):
                continue
            E, _ = ising(l1, l2)
            coarse.append((float(simulate(E, *cheap)[opt_idx].sum()), l1, l2))
    if not coarse:
        raise RuntimeError(
            f"In the range LAM1<={l1_range[1]:.2f}, LAM2<={l2_range[1]} there is no "
            "pair where the minimum of the cost function is the DP optimum. Widen "
            "the ranges or switch to slack variables."
        )
    coarse.sort(reverse=True)

    # ---- B) refinement ---------------------------------------------------
    lr = (hi2 / lo2) ** (1.0 / (n_l2 - 1))  # log grid step
    fine, seen = list(coarse), set()
    for _, c1, c2 in coarse[:n_coarse]:
        for l1 in np.arange(max(0.0, c1 - h1), c1 + h1 + 1e-9, h1 / 5):
            for l2 in np.geomspace(c2 / lr, c2 * lr, 9):
                key = (round(l1, 4), round(l2, 4))
                if key in seen or not valid(l1, l2):
                    continue
                seen.add(key)
                E, _ = ising(l1, l2)
                fine.append((float(simulate(E, *cheap)[opt_idx].sum()), l1, l2))
    fine.sort(reverse=True)

    # ---- C) final: all schedules for the best candidates only ------------
    out = []
    for _, l1, l2 in fine[:n_final]:
        E, _ = ising(l1, l2)
        for ns, tt in scheds:
            p = simulate(E, ns, tt)
            F = (
                np.exp(-gate_error * twoq_overhead * rzz_layer * ns)
                if gate_error
                else 1.0
            )
            ideal = float(p[opt_idx].sum())
            out.append(
                dict(
                    score=F * ideal + (1 - F) * U * len(opt_idx),
                    p_opt=ideal,
                    F=float(F),
                    LAM1=l1 * alpha_out,
                    LAM2=l2 * alpha_out,
                    STEPS=ns,
                    T=tt,
                    rzz=ns * rzz_layer,
                    p_feas=float(p[feas].sum()),
                    rank=int(
                        np.argsort(-p).tolist().index(int(opt_idx[np.argmax(p[opt_idx])]))
                    )
                    + 1,
                )
            )
    out.sort(key=lambda r: -r["score"])
    best = out[0]
    flat = [r for r in out if r["score"] >= plateau * best["score"]]

    if verbose:
        print(f"TUNE  N={N}  capacity={int(C)}  DP optimum={OPT}  uniform={100*U:.4f} %")
        print(
            f"      {len(fine)} points simulated, {len(cands)} candidates in the "
            "Pareto reduction"
        )
        print(
            f"  {'LAM1':>7} {'LAM2':>7} {'STEPS':>5} {'T':>6} {'P(opt)':>9} "
            f"{'x uniform':>10} {'rank':>5} {'P(feas)':>8}"
        )
        for r in out[:8]:
            print(
                f"  {r['LAM1']:>7.2f} {r['LAM2']:>7.3f} {r['STEPS']:>5} {r['T']:>6.1f} "
                f"{100*r['p_opt']:>8.3f}% {r['p_opt']/U:>9.1f}x {r['rank']:>5} "
                f"{100*r['p_feas']:>7.1f}%"
            )
        print(
            f"\n  ==> ALPHA = {alpha_out}   LAM1 = {best['LAM1']:.2f}   "
            f"LAM2 = {best['LAM2']:.3f}   STEPS = {best['STEPS']}   T = {best['T']}"
        )
        print(
            f"      P(opt) = {100*best['p_opt']:.3f} % = {best['p_opt']/U:.1f}x uniform, "
            f"rank {best['rank']}, {best['rzz']} logical RZZ"
        )
        if gate_error:
            print(f"      after attenuation F={best['F']:.3f}: {best['score']/U:.1f}x uniform")
        print(
            f"      plateau: LAM1 {min(r['LAM1'] for r in flat):.2f}-"
            f"{max(r['LAM1'] for r in flat):.2f}, "
            f"LAM2 {min(r['LAM2'] for r in flat):.3f}-"
            f"{max(r['LAM2'] for r in flat):.3f}"
        )
        print(
            f"      recommended SHOTS >= "
            f"{max(10*(1<<N), int(np.ceil(10/best['p_opt'])))}"
        )

    return Penalties(
        ALPHA=alpha_out,
        LAM1=best["LAM1"],
        LAM2=best["LAM2"],
        STEPS=best["STEPS"],
        T=best["T"],
        p_opt=best["p_opt"],
        p_feas=best["p_feas"],
        rank=best["rank"],
        score=best["score"],
        F=best["F"],
        rzz=best["rzz"],
        optimum=OPT,
        x_uniform=best["p_opt"] / U,
        lp_dual=dual,
        shots_recommended=int(np.ceil(10 / best["p_opt"])),
        plateau_LAM1=(min(r["LAM1"] for r in flat), max(r["LAM1"] for r in flat)),
        plateau_LAM2=(min(r["LAM2"] for r in flat), max(r["LAM2"] for r in flat)),
        all_results=out,
    )
