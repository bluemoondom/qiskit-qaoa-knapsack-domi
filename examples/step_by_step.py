"""The same run step by step -- when you want to reach into the intermediates."""

from qiskit_qaoa_knapsack_domi import (
    KnapsackProblem,
    PlotConfig,
    RunConfig,
    build_annealing_circuit,
    build_distribution,
    build_result,
    plot_bitstrings,
    plot_tail,
    run_circuit,
    select_backend,
    transpile_best,
    tune_penalties,
)

problem = KnapsackProblem.random(12, seed=38)
problem.log()

cfg = RunConfig(
    dry_run=True,
    prune_keep=0.5,
    num_transpiles=5,
    shots="auto",
    draw_circuit=True,                 # logical and ISA circuit diagrams
    make_plots=True,                   # coupling map
    plots=PlotConfig(save_dir="out"),  # where to save the result PNGs
)

# 1) penalties: ALPHA, LAM1, LAM2, STEPS, T + recommended SHOTS
pen = tune_penalties(
    problem,
    schedule_grid=[(3, 6.0), (5, 10.0)],
    max_rzz=500,
    l1_range=(2.0, 25.0),
    l2_range=(0.1, 5.0),
)
print("\npenalties:", pen.as_tuple())

# 2) QUBO -> pruning -> Ising -> circuit
#    draws the circuit diagram itself when draw_circuit=True
qc, qubo, ising = build_annealing_circuit(problem, penalties=pen, config=cfg)
print(f"kept {qubo.kept_couplings}/{qubo.raw_couplings} couplings")

# 3) backend + noise-aware transpilation
#    draws the coupling map and ISA diagram per make_plots / draw_circuit
backend = select_backend(cfg, num_qubits=qubo.n)
tr = transpile_best(qc, backend, cfg, num_items=problem.n)

# 4) run
counts = run_circuit(tr.circuit, backend, cfg, pen)

# 5) results -- build_result decodes the counts and prints RESULT SUMMARY
result = build_result(counts, problem, backend=backend, transpiled=tr)

# 6) step by step, the result plots are called by hand;
#    `plots=` is required for them to be saved as PNG
dist = build_distribution(counts, problem, good=0.90)
plot_bitstrings(dist, backend_name=backend.name, plots=cfg.plots)
plot_tail(dist, plots=cfg.plots)
