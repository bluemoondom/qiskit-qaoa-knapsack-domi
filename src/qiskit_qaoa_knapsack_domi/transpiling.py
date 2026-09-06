"""Noise-aware transpilation: finding the lowest-error layout.

This corresponds to steps 4A and 4B from the notebook.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from .backends import error_score, two_qubit_gates, twoq_count
from .config import RunConfig

__all__ = ["TranspileResult", "transpile_best", "seed_scan", "mapomatic_candidate"]


@dataclass
class TranspileResult:
    """Result of the layout selection."""

    circuit: QuantumCircuit
    """The ISA circuit that will run on the chip."""

    method: str
    """"seed-scan" or "mapomatic"."""

    error_sum: float
    twoq: int
    depth: int
    physical_qubits: list[int]
    final_layout: list[int]
    """logical qubit k -> physical qubit."""

    scan: list[tuple] = field(default_factory=list, repr=False)
    candidates: list[tuple[str, QuantumCircuit]] = field(default_factory=list, repr=False)


def seed_scan(
    qc: QuantumCircuit,
    backend,
    config: RunConfig | None = None,
) -> list[tuple]:
    """Transpile the circuit with several seeds and sort them by error.

    Returns:
        A list of (error_sum, twoq, depth, seed, circuit) sorted ascending.
    """
    cfg = config or RunConfig()
    two_q = two_qubit_gates(backend)
    scan = []
    for seed in range(cfg.num_transpiles):
        pm = generate_preset_pass_manager(
            backend=backend,
            optimization_level=cfg.opt_level,
            seed_transpiler=seed,
        )
        cand = pm.run(qc)
        scan.append(
            (
                error_score(cand, backend, two_q),
                twoq_count(cand, two_q=two_q),
                cand.depth(),
                seed,
                cand,
            )
        )
    scan.sort(key=lambda r: r[0])
    return scan


def mapomatic_candidate(
    qc: QuantumCircuit,
    backend,
    config: RunConfig | None = None,
) -> QuantumCircuit | None:
    """Layout via mapomatic: transpile -> deflate -> score every placement.

    Returns ``None`` when mapomatic is not installed or fails.
    """
    cfg = config or RunConfig()
    try:
        import mapomatic as mm

        base = generate_preset_pass_manager(
            backend=backend, optimization_level=cfg.opt_level, seed_transpiler=0
        ).run(qc)
        small = mm.deflate_circuit(base)
        layouts = mm.matching_layouts(small, backend)
        scored = mm.evaluate_layouts(small, layouts, backend)  # sorted, lower = better
        if cfg.verbose:
            print(
                f"\n[B] mapomatic: {len(scored)} matching layouts on "
                f"{backend.name}, top 5 by cost:"
            )
            print(f"     {'rank':>4}  {'mm_cost':>9}  physical qubits")
            for rank, (lay, cost) in enumerate(scored[:5], 1):
                print(f"     {rank:>4}  {cost:>9.4f}  {sorted(lay)}")
        best_layout = scored[0][0]
        return transpile(
            small,
            backend,
            initial_layout=best_layout,
            optimization_level=cfg.opt_level,
            seed_transpiler=0,
        )
    except Exception as e:  # noqa: BLE001 - mapomatic is optional
        if cfg.verbose:
            print(f"\n[B] mapomatic skipped ({e!r})")
        return None


def transpile_best(
    qc: QuantumCircuit,
    backend,
    config: RunConfig | None = None,
    *,
    num_items: int | None = None,
) -> TranspileResult:
    """Pick the lowest-error circuit from the seed scan (and mapomatic if enabled).

    Also draws the coupling map and the ISA circuit diagram when
    ``make_plots`` / ``draw_circuit`` are set.

    Args:
        qc: the logical circuit.
        backend: the target backend.
        config: settings (OPT_LEVEL, NUM_TRANSPILES, USE_MAPOMATIC).
        num_items: N -- how many of the first logical qubits are items (the
            rest would be slack variables). None = all of them.
    """
    cfg = config or RunConfig()
    two_q = two_qubit_gates(backend)
    n = qc.num_qubits
    N = num_items if num_items is not None else n

    if cfg.verbose:
        print("\n" + "=" * 64)
        print(" TRANSPILATION  (noise-aware layout selection)")
        print("=" * 64)

    scan = seed_scan(qc, backend, cfg)
    if cfg.verbose:
        print(f"\n[A] Seed scan ({cfg.num_transpiles} transpiles), top 5 by error:")
        print(f"     {'rank':>4}  {'err_sum':>9}  {'2q':>5}  {'depth':>6}  {'seed':>4}")
        for rank, (sc, tq, dp_, sd, _) in enumerate(scan[:5], 1):
            print(f"     {rank:>4}  {sc:>9.4f}  {tq:>5}  {dp_:>6}  {sd:>4}")
        worst = scan[-1][0]
        drop = 100 * (1 - scan[0][0] / worst) if worst else 0.0
        print(
            f"     best seed error_sum = {scan[0][0]:.4f}  "
            f"(worst = {worst:.4f}, i.e. {drop:.0f}% lower)"
        )

    candidates: list[tuple[str, QuantumCircuit]] = [("seed-scan", scan[0][4])]

    if cfg.use_mapomatic:
        isa_mm = mapomatic_candidate(qc, backend, cfg)
        if isa_mm is not None:
            candidates.append(("mapomatic", isa_mm))

    if cfg.verbose:
        print("\n" + "-" * 64)
        print(f" {'method':<12} {'err_sum':>9} {'2q gates':>9} {'depth':>7}")
        print("-" * 64)
        for name, circ in candidates:
            print(
                f" {name:<12} {error_score(circ, backend, two_q):>9.4f} "
                f"{twoq_count(circ, two_q=two_q):>9} {circ.depth():>7}"
            )

    chosen, isa = min(candidates, key=lambda kv: error_score(kv[1], backend, two_q))
    if cfg.verbose:
        print("-" * 64)
        print(f" chosen: {chosen}")

    phys_qubits = sorted(
        {
            isa.find_bit(q).index
            for inst in isa.data
            for q in inst.qubits
            if inst.operation.num_qubits == 2
        }
    )
    tq = twoq_count(isa, two_q=two_q)
    if cfg.verbose:
        print(f" physical qubits used ({len(phys_qubits)}): {phys_qubits}")
        if tq > cfg.max_twoq_warn:
            print(
                f" WARNING: >{cfg.max_twoq_warn} two-qubit gates -> expect noise. "
                "Lower N or STEPS."
            )

    # Reconcile the two numbering schemes: the layout plot (view='virtual') would
    # label the used qubits 0..n-1 (logical), while the list above is physical.
    # Here is exactly which physical qubit each logical qubit (item / slack) runs on.
    final_layout = list(isa.layout.final_index_layout())
    if cfg.verbose:
        print("\n logical -> physical qubit map:")
        print(f"   {'logical':>7}  {'role':<10} {'physical':>8}")
        for k in range(n):
            role = f"item {k}" if k < N else f"slack {k - N}"
            print(f"   {k:>7}  {role:<10} {final_layout[k]:>8}")

    # Device-layout picture (the coupling map with used qubits highlighted).
    if cfg.make_plots:
        from .plotting import plot_layout

        plot_layout(isa, backend, cfg)

    # Diagram of the transpiled (ISA) circuit - what actually runs on the chip
    # (ecr / sx / rz). This is BIG (hundreds of gates); expect a very wide figure.
    if cfg.draw_circuit:
        from .plotting import draw_circuit_diagram

        draw_circuit_diagram(isa, cfg)

    if cfg.verbose:
        print(isa.count_ops())

    return TranspileResult(
        circuit=isa,
        method=chosen,
        error_sum=error_score(isa, backend, two_q),
        twoq=tq,
        depth=isa.depth(),
        physical_qubits=phys_qubits,
        final_layout=final_layout,
        scan=[(a, b, c, d) for a, b, c, d, _ in scan],
        candidates=candidates,
    )
