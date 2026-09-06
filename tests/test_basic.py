"""Tests that need neither an account nor a real backend."""

import numpy as np
import pytest

from qiskit_qaoa_knapsack_domi import (
    KnapsackProblem,
    Penalties,
    RunConfig,
    build_annealing_circuit,
    build_distribution,
    build_qubo,
    dp_optimum,
    lp_dual,
    prune_couplings,
    qubo_to_ising,
    tune_penalties,
)


def test_problem_derives_n_from_values():
    p = KnapsackProblem([10, 20, 30], [5, 4, 6], capacity=9)
    assert p.n == 3 and p.N == 3
    assert p.num_rzz_full == 3


def test_problem_validates_lengths():
    with pytest.raises(ValueError):
        KnapsackProblem([1, 2, 3], [1, 2])


def test_default_capacity_is_sum_over_three():
    p = KnapsackProblem([1, 2, 3], [3, 3, 3])
    assert p.capacity == 3


def test_random_is_reproducible():
    a = KnapsackProblem.random(15, seed=38)
    b = KnapsackProblem.random(15, seed=38)
    assert a.values == b.values and a.weights == b.weights


def test_dp_optimum_matches_bruteforce():
    p = KnapsackProblem.random(12, seed=7)
    tot_v, tot_w = p.enumerate_states()
    assert p.optimum == int(tot_v[tot_w <= p.capacity].max())
    assert dp_optimum(p.values, p.weights, p.capacity) == p.optimum


def test_lp_dual_is_upper_bound_on_density():
    p = KnapsackProblem.random(10, seed=3)
    d = lp_dual(p)
    assert 0 < d <= max(v / w for v, w in zip(p.values, p.weights, strict=True))


def test_prune_keeps_roughly_the_requested_fraction():
    raw = {(i, j): float(i + j + 1) for i in range(6) for j in range(i + 1, 6)}
    kept = prune_couplings(raw, 0.5, verbose=False)
    assert 0.4 * len(raw) <= len(kept) <= 0.6 * len(raw) + 1
    assert min(abs(v) for v in kept.values()) >= 0


def test_prune_keep_one_keeps_everything():
    raw = {(0, 1): 1.0, (0, 2): 2.0}
    assert prune_couplings(raw, 1.0, verbose=False) == raw


def test_prune_keep_rejects_bad_values():
    with pytest.raises(ValueError):
        prune_couplings({(0, 1): 1.0}, 0.0)


def test_qubo_to_ising_reproduces_cost_ordering():
    p = KnapsackProblem.random(8, seed=11)
    pen = Penalties(ALPHA=2.0, LAM1=6.2, LAM2=0.179)
    qubo = build_qubo(p, penalties=pen, prune_keep=1.0, verbose=False)
    ising = qubo_to_ising(qubo)

    states = np.arange(1 << p.n)
    bits = (states[:, None] >> np.arange(p.n)) & 1
    e_qubo = bits @ np.array([qubo.lin[i] for i in range(p.n)])
    for (i, j), q in qubo.quad.items():
        e_qubo = e_qubo + q * bits[:, i] * bits[:, j]

    z = 1 - 2 * bits
    e_ising = z @ np.array([ising.h[i] for i in range(p.n)])
    for (i, j), jj in ising.J.items():
        e_ising = e_ising + jj * z[:, i] * z[:, j]

    # Ising is an affine image of the QUBO: E_ising * scale = E_qubo - const
    diff = e_ising * ising.scale - e_qubo
    assert np.allclose(diff, diff[0])


def _tuned(problem, steps=3, t=6.0):
    """Quickly tuned penalties for a small instance."""
    return tune_penalties(
        problem,
        schedule_grid=[(steps, t)],
        l1_range=(0.0, 25.0),  # the auto window is calibrated on N=8..15
        l2_range=(0.1, 30.0),  # small/tight instances need a higher LAM2
        n_l2=16,
        n_coarse=2,
        verbose=False,
    )


def test_build_circuit_shape():
    p = KnapsackProblem.random(6, seed=5)
    pen = _tuned(p, steps=3)
    qc, qubo, ising = build_annealing_circuit(
        p, penalties=pen, config=RunConfig(prune_keep=1.0, verbose=False, draw_circuit=False)
    )
    assert qc.num_qubits == 6
    ops = qc.count_ops()
    assert ops["rzz"] == pen.STEPS * p.num_rzz_full
    assert ops["rx"] == pen.STEPS * 6
    assert "measure" in ops


def test_pruning_reduces_rzz_count():
    p = KnapsackProblem.random(8, seed=5)
    pen = _tuned(p, steps=2)
    full, _, _ = build_annealing_circuit(
        p, penalties=pen, config=RunConfig(prune_keep=1.0, verbose=False, draw_circuit=False)
    )
    half, _, _ = build_annealing_circuit(
        p, penalties=pen, config=RunConfig(prune_keep=0.5, verbose=False, draw_circuit=False)
    )
    assert half.count_ops()["rzz"] < full.count_ops()["rzz"]


def test_weak_penalty_is_detected():
    p = KnapsackProblem.random(8, seed=5)
    with pytest.raises(ValueError, match="penalty is weak"):
        build_annealing_circuit(
            p,
            penalties=Penalties(ALPHA=2.0, LAM1=0.0, LAM2=0.0),
            config=RunConfig(verbose=False, draw_circuit=False),
        )


def test_tune_penalties_finds_valid_parameters():
    p = KnapsackProblem.random(9, seed=21)
    cfg = tune_penalties(
        p,
        schedule_grid=[(3, 6.0)],
        max_rzz=500,
        l1_range=(2.0, 25.0),
        l2_range=(0.1, 5.0),
        n_l2=10,
        n_coarse=3,
        verbose=False,
    )
    assert cfg.ALPHA == 2.0
    assert cfg.LAM1 > 0 and cfg.LAM2 > 0
    assert cfg.optimum == p.optimum
    assert cfg.p_opt > 1 / (1 << p.n)  # better than uniform random
    # the notebook's dict API
    assert cfg["LAM1"] == cfg.LAM1
    assert cfg["shots_for_10_hits"] == cfg.shots_recommended


def test_tune_penalties_accepts_raw_arrays():
    p = KnapsackProblem.random(8, seed=4)
    cfg = tune_penalties(
        list(p.values),
        list(p.weights),
        p.capacity,
        schedule_grid=[(3, 6.0)],
        n_l2=8,
        n_coarse=2,
        verbose=False,
    )
    assert cfg.optimum == p.optimum


def test_tuned_penalties_yield_feasible_ground_state():
    p = KnapsackProblem.random(9, seed=21)
    cfg = tune_penalties(p, schedule_grid=[(3, 6.0)], n_l2=10, n_coarse=3, verbose=False)
    # must not trip the ground-state check
    build_annealing_circuit(
        p, penalties=cfg, config=RunConfig(verbose=False, draw_circuit=False)
    )


def test_distribution_from_counts():
    p = KnapsackProblem([10, 20, 25], [5, 4, 6], capacity=9)
    # 011 = items 0 and 1 -> weight 9, value 30 = optimum
    # 100 = item 2        -> weight 6, value 25, feasible but not optimal
    counts = {"011": 80, "100": 20}
    dist = build_distribution(counts, p, verbose=False)
    assert dist.optimum == 30
    assert dist.p_optimum == pytest.approx(0.8)
    assert dist.p_feasible == pytest.approx(1.0)


def test_runconfig_shots_resolution():
    assert RunConfig(dry_run=True).resolve_shots() == 512
    assert RunConfig(shots=1234).resolve_shots() == 1234
    pen = Penalties(shots_recommended=999)
    assert RunConfig(shots="auto").resolve_shots(pen) == 999
    with pytest.raises(ValueError):
        RunConfig(shots="auto").resolve_shots()


def test_runconfig_validates():
    with pytest.raises(ValueError):
        RunConfig(prune_keep=0)
    with pytest.raises(ValueError):
        RunConfig(layout_view="nonsense")


def test_random_matches_notebook_snippet():
    """random() must reproduce exactly what the notebook code did."""
    import random as _random

    N = 15
    _random.seed(38)
    values = [_random.randint(5, 60) for _ in range(N)]
    weights = [_random.randint(1, 20) for _ in range(N)]
    CAPACITY = sum(weights) // 3

    p = KnapsackProblem.random(N, seed=38)
    assert list(p.values) == values
    assert list(p.weights) == weights
    assert p.capacity == CAPACITY


def test_random_accepts_manual_capacity():
    auto = KnapsackProblem.random(15, seed=38)
    manual = KnapsackProblem.random(15, seed=38, capacity=100)
    assert manual.capacity == 100
    assert manual.values == auto.values and manual.weights == auto.weights


def test_random_capacity_divisor():
    p = KnapsackProblem.random(15, seed=38, capacity_divisor=4)
    assert p.capacity == sum(p.weights) // 4


def test_manual_capacity_changes_optimum():
    tight = KnapsackProblem.random(12, seed=38, capacity=30)
    loose = KnapsackProblem.random(12, seed=38, capacity=80)
    assert loose.optimum > tight.optimum


def test_explicit_values_weights_capacity():
    p = KnapsackProblem(values=[10, 20, 30, 40], weights=[5, 4, 6, 3], capacity=10)
    assert p.n == 4 and p.capacity == 10
    assert p.optimum == 70  # items 2 and 3: weight 9, value 70


# --- plotting: the flags must behave the same in solve() and step by step ---


def _count_draws(monkeypatch):
    """Count calls to the drawing functions across modules."""
    calls = {"circuit": 0, "layout": 0}
    from qiskit_qaoa_knapsack_domi import pipeline, plotting, transpiling

    def fake_draw(*a, **k):
        calls["circuit"] += 1

    def fake_layout(*a, **k):
        calls["layout"] += 1

    for mod in (plotting, pipeline, transpiling):
        monkeypatch.setattr(mod, "draw_circuit_diagram", fake_draw, raising=False)
        monkeypatch.setattr(mod, "plot_layout", fake_layout, raising=False)
    monkeypatch.setattr(plotting, "draw_circuit_diagram", fake_draw)
    monkeypatch.setattr(plotting, "plot_layout", fake_layout)
    return calls


def test_build_annealing_circuit_honours_draw_flag(monkeypatch):
    calls = _count_draws(monkeypatch)
    p = KnapsackProblem.random(6, seed=5)
    pen = _tuned(p, steps=2)

    build_annealing_circuit(
        p, penalties=pen, config=RunConfig(draw_circuit=False, verbose=False)
    )
    assert calls["circuit"] == 0

    build_annealing_circuit(
        p, penalties=pen, config=RunConfig(draw_circuit=True, verbose=False)
    )
    assert calls["circuit"] == 1


def test_transpile_best_honours_plot_flags(monkeypatch):
    from qiskit_qaoa_knapsack_domi import select_backend, transpile_best

    calls = _count_draws(monkeypatch)
    p = KnapsackProblem.random(5, seed=5)
    pen = _tuned(p, steps=1)
    cfg_off = RunConfig(
        dry_run=True,
        draw_circuit=False,
        make_plots=False,
        num_transpiles=1,
        verbose=False,
    )
    qc, qubo, _ = build_annealing_circuit(p, penalties=pen, config=cfg_off)
    backend = select_backend(cfg_off, num_qubits=qubo.n)

    transpile_best(qc, backend, cfg_off, num_items=p.n)
    assert calls == {"circuit": 0, "layout": 0}

    cfg_on = RunConfig(
        dry_run=True,
        draw_circuit=True,
        make_plots=True,
        num_transpiles=1,
        verbose=False,
    )
    transpile_best(qc, backend, cfg_on, num_items=p.n)
    assert calls == {"circuit": 1, "layout": 1}


def test_solve_does_not_draw_twice(monkeypatch):
    from qiskit_qaoa_knapsack_domi import solve

    calls = _count_draws(monkeypatch)
    p = KnapsackProblem.random(5, seed=5)
    cfg = RunConfig(
        dry_run=True,
        draw_circuit=True,
        make_plots=False,
        num_transpiles=1,
        verbose=False,
    )
    solve(
        p,
        config=cfg,
        tune_kwargs=dict(
            schedule_grid=[(1, 2.0)],
            l1_range=(0.0, 25.0),
            l2_range=(0.1, 30.0),
            n_l2=8,
            n_coarse=1,
            verbose=False,
        ),
    )
    # the logical circuit and the ISA circuit, each exactly once
    assert calls["circuit"] == 2


# --- plot_bitstrings: controlling how many bars are drawn -----------------

def _fake_dist(n_states=64, n_nonzero=40):
    """A distribution with a known number of non-zero bars."""
    import numpy as np

    from qiskit_qaoa_knapsack_domi.results import Distribution

    prob = np.zeros(n_states)
    prob[:n_nonzero] = np.linspace(1.0, 0.1, n_nonzero)
    prob /= prob.sum()
    tot_v = np.arange(n_states)
    tot_w = np.zeros(n_states)
    feas = np.ones(n_states, bool)
    return Distribution(
        prob=prob,
        tot_v=tot_v,
        tot_w=tot_w,
        feas=feas,
        good=feas & (tot_v >= 0.9 * (n_states - 1)),
        optimal_masks={n_states - 1},
        optimum=n_states - 1,
        good_threshold=0.9,
        shots=1000,
    )


def test_plot_bitstrings_top_k_limits_bars():
    import matplotlib

    matplotlib.use("Agg")
    from qiskit_qaoa_knapsack_domi import PlotConfig, plot_bitstrings

    dist = _fake_dist(n_nonzero=40)
    fig = plot_bitstrings(dist, top_k=10, plots=PlotConfig(save_dir=None))
    assert len(fig.axes[0].patches) == 10


def test_plot_bitstrings_top_k_overrides_plotconfig_without_mutating_it():
    import matplotlib

    matplotlib.use("Agg")
    from qiskit_qaoa_knapsack_domi import PlotConfig, plot_bitstrings

    dist = _fake_dist(n_nonzero=40)
    pc = PlotConfig(top_k=30)
    fig = plot_bitstrings(dist, top_k=5, plots=pc)
    assert len(fig.axes[0].patches) == 5
    assert pc.top_k == 30  # the config must stay untouched


def test_plot_bitstrings_uses_plotconfig_when_no_override():
    import matplotlib

    matplotlib.use("Agg")
    from qiskit_qaoa_knapsack_domi import PlotConfig, plot_bitstrings

    dist = _fake_dist(n_nonzero=40)
    fig = plot_bitstrings(dist, plots=PlotConfig(top_k=7))
    assert len(fig.axes[0].patches) == 7


def test_plot_bitstrings_top_k_above_nonzero_count_is_clipped():
    import matplotlib

    matplotlib.use("Agg")
    from qiskit_qaoa_knapsack_domi import plot_bitstrings

    dist = _fake_dist(n_states=64, n_nonzero=12)
    fig = plot_bitstrings(dist, top_k=500)
    assert len(fig.axes[0].patches) == 12  # zero bars are dropped


def test_plot_bitstrings_rejects_bad_top_k():
    import matplotlib
    import pytest as _pytest

    matplotlib.use("Agg")
    from qiskit_qaoa_knapsack_domi import plot_bitstrings

    with _pytest.raises(ValueError):
        plot_bitstrings(_fake_dist(), top_k=0)


# --- build_result: the RESULT SUMMARY block outside solve() ---------------

def test_build_result_decodes_and_reports(capsys):
    from qiskit_qaoa_knapsack_domi import build_result

    p = KnapsackProblem([10, 20, 25], [5, 4, 6], capacity=9)
    counts = {"011": 80, "100": 20}
    r = build_result(counts, p)
    assert r.selection == [0, 1]
    assert r.value == 30 and r.weight == 9
    assert r.optimum == 30 and r.is_optimal
    assert r.shots == 100

    out = capsys.readouterr().out
    assert "KNAPSACK  -  RESULT SUMMARY" in out
    assert "Solution quality" in out
    assert "100.0%" in out


def test_build_result_quiet(capsys):
    from qiskit_qaoa_knapsack_domi import build_result

    p = KnapsackProblem([10, 20, 25], [5, 4, 6], capacity=9)
    build_result({"011": 5}, p, verbose=False)
    assert capsys.readouterr().out == ""


def test_build_result_reports_no_feasible_sample(capsys):
    from qiskit_qaoa_knapsack_domi import build_result

    p = KnapsackProblem([10, 20, 25], [5, 4, 6], capacity=9)
    r = build_result({"111": 10}, p)  # weight 15 > capacity
    assert r.selection == [] and r.value == 0
    assert "No feasible solution" in capsys.readouterr().out


def test_build_result_reports_logical_not_chip_qubits(capsys):
    """n_qubits must be the logical count, never the whole device."""
    from qiskit_qaoa_knapsack_domi import build_result

    class FakeTranspiled:
        method = "seed-scan"
        twoq = 766
        final_layout = [0, 1, 2]  # three logical qubits

    class FakeBackend:
        name = "ibm_kingston"

    p = KnapsackProblem([10, 20, 25], [5, 4, 6], capacity=9)
    r = build_result(
        {"011": 1}, p, backend=FakeBackend(), transpiled=FakeTranspiled()
    )
    assert r.n_qubits == 3
    assert r.backend_name == "ibm_kingston"
    assert r.method == "seed-scan"
    assert r.twoq == 766
    out = capsys.readouterr().out
    assert "Logical qubits : 3" in out
    assert "ibm_kingston" in out


def test_solve_still_prints_summary_once(capsys):
    from qiskit_qaoa_knapsack_domi import solve

    p = KnapsackProblem.random(5, seed=5)
    cfg = RunConfig(
        dry_run=True, draw_circuit=False, make_plots=False, num_transpiles=1
    )
    solve(
        p,
        config=cfg,
        tune_kwargs=dict(
            schedule_grid=[(1, 2.0)], l1_range=(0.0, 25.0), l2_range=(0.1, 30.0),
            n_l2=8, n_coarse=1, verbose=False,
        ),
    )
    assert capsys.readouterr().out.count("RESULT SUMMARY") == 1
