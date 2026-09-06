"""The shortest path: instance -> result.

Runs for free on a local fake backend:
    python examples/quickstart.py
"""

from qiskit_qaoa_knapsack_domi import KnapsackProblem, PlotConfig, RunConfig, solve

# The inputs are values, weights and capacity; N is derived from len(values).
#
# Equivalent of the notebook code:
#     random.seed(38)
#     values   = [random.randint(5, 60) for _ in range(15)]
#     weights  = [random.randint(1, 20) for _ in range(15)]
#     CAPACITY = sum(weights) // 3
problem = KnapsackProblem.random(15, seed=38)

# The capacity can also be set by hand:
#     problem = KnapsackProblem.random(15, seed=38, capacity=100)
#     problem = KnapsackProblem(values, weights, 100)

config = RunConfig(
    dry_run=True,           # True = local fake backend, False = real hardware
    opt_level=3,
    num_transpiles=10,
    use_mapomatic=False,
    make_plots=True,
    draw_circuit=False,     # the circuit diagram at N=15 is very wide
    fold=100,
    layout_view="physical",
    prune_keep=0.5,
    shots=300_000,
    plots=PlotConfig(save_dir="out"),
)

session = solve(
    problem,
    config=config,
    tune_kwargs=dict(
        schedule_grid=[(3, 6.0), (5, 10.0), (7, 15.0)],
        max_rzz=500,
        alpha_out=2.0,
        l1_range=(2.0, 25.0),
        l2_range=(0.1, 5.0),
    ),
)

r = session.result
print(f"\nfound {r.value} / {r.optimum} = {100 * r.ratio:.1f} % of the optimum")
