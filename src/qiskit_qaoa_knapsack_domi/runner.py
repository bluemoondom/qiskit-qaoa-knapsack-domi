"""Running the circuit and monitoring the job."""

from __future__ import annotations

import time

from qiskit.circuit import QuantumCircuit

from .config import Penalties, RunConfig

__all__ = ["monitor_job", "run_circuit"]

FINAL_STATES = {"DONE", "FAILED", "CANCELLED"}


def monitor_job(job, poll_interval: float = 10) -> float:
    """Follow the job status and print a breakdown of time per state.

    Args:
        job: the running RuntimeJob.
        poll_interval: polling period in seconds.

    Returns:
        Total wall-clock time in seconds.
    """
    status_times: dict[str, float] = {}
    current_status = None
    start_time = time.time()
    last_progress = -1

    print(f" Job ID   : {job.job_id()}")
    print(f" Backend  : {job.backend().name if hasattr(job, 'backend') else 'N/A'}")
    print(f" Started  : {time.strftime('%H:%M:%S', time.localtime(start_time))}")
    print("-" * 64)

    try:
        while True:
            status = job.status()
            now = time.time()

            if status != current_status:
                if current_status is not None:
                    elapsed = now - status_times[current_status]
                    print(f"  [←] {current_status:<15} : {elapsed:>6.1f}s")
                status_times[status] = now
                current_status = status
                print(
                    f"  [→] {status:<15} : "
                    f"{time.strftime('%H:%M:%S', time.localtime(now))}"
                )

            try:
                progress = None
                if hasattr(job, "progress") and callable(job.progress):
                    progress = job.progress()
                elif hasattr(job, "percent_complete") and callable(job.percent_complete):
                    progress = job.percent_complete()
                if progress is not None and progress != last_progress:
                    last_progress = progress
                    bar = "█" * int(progress * 20) + " " * (20 - int(progress * 20))
                    print(f"  [▓] Progress: |{bar}| {progress*100:>5.1f}%")
            except Exception:  # noqa: BLE001 - progress is an optional API
                pass

            if status in FINAL_STATES:
                elapsed = now - status_times[status]
                print(f"  [←] {status:<15} : {elapsed:>6.1f}s")
                break

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\n  [!] Monitoring stopped by user")

    print("-" * 64)
    print(" TIME BREAKDOWN:")
    total_time = time.time() - start_time
    for state, start in sorted(status_times.items(), key=lambda x: x[1]):
        end = next((v for k, v in status_times.items() if v > start), total_time)
        if state == current_status:
            end = time.time()
        duration = end - start
        pct = 100 * duration / total_time if total_time > 0 else 0
        print(f"   {state:<15}: {duration:>6.1f}s ({pct:>5.1f}%)")

    # --- Billed usage (Pay-As-You-Go only) ---
    try:
        usage = job.usage()
        if usage:
            print(f"\n   {'Billed usage':<15}: {usage:>6.1f}s")
    except Exception:  # noqa: BLE001 - usage is not on every plan
        pass

    print(f"   {'TOTAL WALL':<15}: {total_time:>6.1f}s")
    print("=" * 64)
    return total_time


def run_circuit(
    isa: QuantumCircuit,
    backend,
    config: RunConfig | None = None,
    penalties: Penalties | None = None,
) -> dict[str, int]:
    """Submit the ISA circuit to the Sampler and return the counts.

    Enables dynamical decoupling (XY4) and gate/measurement twirling according
    to the configuration.

    Args:
        isa: the transpiled circuit.
        backend: the backend it will run on.
        config: settings (SHOTS, DD, twirling, monitoring).
        penalties: used when ``config.shots="auto"``.

    Returns:
        A dictionary of {bitstring: count}.
    """
    from qiskit_ibm_runtime import SamplerV2 as Sampler

    cfg = config or RunConfig()
    shots = cfg.resolve_shots(penalties)

    sampler = Sampler(mode=backend)
    sampler.options.dynamical_decoupling.enable = cfg.dynamical_decoupling
    sampler.options.dynamical_decoupling.sequence_type = cfg.dd_sequence
    sampler.options.twirling.enable_gates = cfg.twirling_gates
    sampler.options.twirling.enable_measure = cfg.twirling_measure

    job = sampler.run([isa], shots=shots)
    if cfg.verbose:
        print("\nJob ID:", job.job_id())

    if cfg.monitor and not cfg.dry_run:
        monitor_job(job, poll_interval=cfg.poll_interval)

    return job.result()[0].data.meas.get_counts()
