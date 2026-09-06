"""Running on real IBM Quantum hardware.

Prerequisite: the account is saved (once is enough)::

    from qiskit_ibm_runtime import QiskitRuntimeService

    QiskitRuntimeService.save_account(
        channel="ibm_quantum_platform",
        token="...",
        instance="crn:v1:bluemix:public:quantum-computing:...",
        overwrite=True,
        set_as_default=True,
    )
"""

from qiskit_qaoa_knapsack_domi import KnapsackProblem, PlotConfig, RunConfig, solve

problem = KnapsackProblem.random(15, seed=38)

config = RunConfig(
    dry_run=False,          # real hardware
    num_transpiles=10,      # more seeds = better layout
    use_mapomatic=True,     # requires `pip install mapomatic`
    prune_keep=0.5,         # the main lever on circuit depth
    shots="auto",           # take the recommendation from tune_penalties
    dynamical_decoupling=True,
    dd_sequence="XY4",
    monitor=True,
    poll_interval=2,
    plots=PlotConfig(save_dir="out"),
)

session = solve(
    problem,
    config=config,
    tune_kwargs=dict(
        schedule_grid=[(3, 6.0), (5, 10.0), (7, 15.0)],
        max_rzz=500,
        gate_error=8e-3,    # score P(opt) after hardware attenuation
        l1_range=(2.0, 25.0),
        l2_range=(0.1, 5.0),
    ),
)

r = session.result
print(f"\nfound {r.value} / {r.optimum} = {100 * r.ratio:.1f} % of the optimum")
