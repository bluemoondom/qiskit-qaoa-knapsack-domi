"""Command line: ``python -m qiskit_qaoa_knapsack_domi``."""

from __future__ import annotations

import argparse
import json
import sys

from .config import PlotConfig, RunConfig
from .pipeline import solve
from .problem import KnapsackProblem

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qiskit-qaoa-knapsack-domi",
        description="Knapsack by digitized quantum annealing on IBM Quantum.",
    )
    inst = p.add_argument_group("instance")
    inst.add_argument("-n", "--n", type=int, default=15, help="number of items (N)")
    inst.add_argument("--seed", type=int, default=38, help="random instance seed")
    inst.add_argument("--values", type=str, help="comma-separated values")
    inst.add_argument("--weights", type=str, help="comma-separated weights")
    inst.add_argument(
        "--capacity", type=int, help="capacity (defaults to total weight / 3)"
    )

    run = p.add_argument_group("run")
    run.add_argument(
        "--dry-run", action="store_true", help="local fake backend instead of real HW"
    )
    run.add_argument("--backend", type=str, help="force a specific backend")
    run.add_argument(
        "--shots", type=str, default="300000", help='number of shots, or "auto"'
    )
    run.add_argument("--opt-level", type=int, default=3, choices=[0, 1, 2, 3])
    run.add_argument("--num-transpiles", type=int, default=10)
    run.add_argument("--mapomatic", action="store_true", help="also try mapomatic")
    run.add_argument("--prune-keep", type=float, default=0.5)

    tune = p.add_argument_group("penalty tuning")
    tune.add_argument("--no-tune", action="store_true", help="skip tune_penalties")
    tune.add_argument("--max-rzz", type=int, default=500)
    tune.add_argument(
        "--schedule",
        type=str,
        default="3:6.0,5:10.0,7:15.0",
        help="comma-separated STEPS:T schedules",
    )
    tune.add_argument("--alpha", type=float, default=2.0)
    tune.add_argument("--l1-range", type=str, default="2.0,25.0")
    tune.add_argument("--l2-range", type=str, default="0.1,5.0")

    out = p.add_argument_group("output")
    out.add_argument("--no-plots", action="store_true")
    out.add_argument("--no-draw", action="store_true")
    out.add_argument("--save-dir", type=str, help="where to save the plot PNGs")
    out.add_argument("--quiet", action="store_true", help="suppress the log")
    out.add_argument("--json", type=str, help="write the result to a JSON file")
    return p


def _pair(s: str) -> tuple[float, float]:
    a, b = s.split(",")
    return float(a), float(b)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.values or args.weights:
        if not (args.values and args.weights):
            print(
                "error: --values and --weights must be given together", file=sys.stderr
            )
            return 2
        values = [int(x) for x in args.values.split(",")]
        weights = [int(x) for x in args.weights.split(",")]
        problem = KnapsackProblem(values, weights, args.capacity)
    else:
        problem = KnapsackProblem.random(args.n, seed=args.seed, capacity=args.capacity)

    shots = "auto" if args.shots == "auto" else int(args.shots)
    cfg = RunConfig(
        dry_run=args.dry_run,
        backend_name=args.backend,
        shots=shots,
        opt_level=args.opt_level,
        num_transpiles=args.num_transpiles,
        use_mapomatic=args.mapomatic,
        prune_keep=args.prune_keep,
        make_plots=not args.no_plots,
        draw_circuit=not args.no_draw,
        verbose=not args.quiet,
        plots=PlotConfig(save_dir=args.save_dir),
    )

    schedule = [
        (int(s.split(":")[0]), float(s.split(":")[1])) for s in args.schedule.split(",") if s
    ]
    tune_kwargs = dict(
        schedule_grid=schedule,
        max_rzz=args.max_rzz,
        alpha_out=args.alpha,
        l1_range=_pair(args.l1_range),
        l2_range=_pair(args.l2_range),
    )

    session = solve(problem, config=cfg, tune=not args.no_tune, tune_kwargs=tune_kwargs)

    if args.json and session.result:
        payload = {
            "values": list(problem.values),
            "weights": list(problem.weights),
            "capacity": problem.capacity,
            "selection": session.result.selection,
            "value": session.result.value,
            "weight": session.result.weight,
            "optimum": session.result.optimum,
            "ratio": session.result.ratio,
            "backend": session.result.backend_name,
            "shots": session.result.shots,
            "penalties": {
                "ALPHA": session.penalties.ALPHA,
                "LAM1": session.penalties.LAM1,
                "LAM2": session.penalties.LAM2,
                "STEPS": session.penalties.STEPS,
                "T": session.penalties.T,
            },
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    return 0 if session.result and session.result.selection else 1


if __name__ == "__main__":
    raise SystemExit(main())
