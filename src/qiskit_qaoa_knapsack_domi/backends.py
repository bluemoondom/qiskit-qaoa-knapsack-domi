"""Backend selection and metrics for comparing layouts."""

from __future__ import annotations

from collections.abc import Sequence

from qiskit.circuit import QuantumCircuit

from .config import RunConfig

__all__ = ["select_backend", "two_qubit_gates", "error_score", "twoq_count"]

_TWO_Q_CANDIDATES = ("ecr", "cx", "cz", "rzz")


def select_backend(config: RunConfig | None = None, num_qubits: int = 1):
    """Return a backend according to the configuration.

    ``dry_run=True`` -> local FakeSherbrooke (free, no account needed).
    ``dry_run=False`` -> the least busy real backend with enough qubits, or the
    backend named in ``config.backend_name``.

    Args:
        config: run settings.
        num_qubits: how many logical qubits the circuit needs.
    """
    cfg = config or RunConfig()
    if cfg.dry_run:
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

        backend = FakeSherbrooke()
        if cfg.verbose:
            print(f"[DRY_RUN] fake backend: {backend.name} ({backend.num_qubits} qubits)")
        return backend

    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService()
    if cfg.backend_name:
        backend = service.backend(
            cfg.backend_name, use_fractional_gates=cfg.use_fractional_gates
        )
    else:
        backend = service.least_busy(
            operational=True,
            simulator=False,
            min_num_qubits=num_qubits,
            use_fractional_gates=cfg.use_fractional_gates,
        )
    if cfg.verbose:
        print(f"Real backend: {backend.name} ({backend.num_qubits} qubits)")
        basis = list(getattr(backend, "basis_gates", []) or [])
        print("basis:", basis)
        print("rzz in basis :", "rzz" in basis)
        print("rzz in target:", "rzz" in backend.target.operation_names)
    return backend


def two_qubit_gates(backend) -> list[str]:
    """Which two-qubit gates the backend actually has."""
    names = backend.target.operation_names
    return [g for g in _TWO_Q_CANDIDATES if g in names]


def error_score(
    circ: QuantumCircuit, backend, two_q: Sequence[str] | None = None
) -> float:
    """Sum of two-qubit gate and readout errors on the qubits the circuit uses.

    Lower = better. One consistent metric for comparing layouts.
    """
    target = backend.target
    two_q = list(two_q) if two_q is not None else two_qubit_gates(backend)
    s = 0.0
    for instr in circ.data:
        op = instr.operation
        qs = tuple(circ.find_bit(q).index for q in instr.qubits)
        if op.num_qubits == 2 and op.name in two_q:
            pr = target[op.name].get(qs)
            if pr and pr.error is not None:
                s += pr.error
        elif op.name == "measure":
            pr = target["measure"].get(qs)
            if pr and pr.error is not None:
                s += pr.error
    return s


def twoq_count(
    circ: QuantumCircuit, backend=None, two_q: Sequence[str] | None = None
) -> int:
    """Number of two-qubit gates in the circuit."""
    if two_q is None:
        two_q = two_qubit_gates(backend) if backend is not None else _TWO_Q_CANDIDATES
    two_q = set(two_q)
    return sum(v for k, v in circ.count_ops().items() if k in two_q)
