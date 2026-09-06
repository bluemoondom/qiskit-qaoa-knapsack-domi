"""Run configuration.

`RunConfig` holds the settings that lived in the notebook's SETTINGS cell.
`Penalties` holds what :func:`tune_penalties` determines (ALPHA/LAM1/LAM2/STEPS/T).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

__all__ = ["RunConfig", "Penalties", "PlotConfig"]

LayoutView = Literal["physical", "virtual"]


@dataclass
class PlotConfig:
    """Settings for the result plots (notebook cells 10/11)."""

    good: float = 0.90
    """Blue = solutions within (1-good) of the optimum."""

    top_k: int = 300
    """How many of the most probable bitstrings to plot."""

    dpi: int = 110
    save_dir: str | None = None
    """Where to save the PNGs. None = don't save, only display."""

    bitstrings_filename: str = "bitstrings.png"
    tail_filename: str = "tail.png"


@dataclass
class RunConfig:
    """Run settings -- the equivalent of the notebook's SETTINGS block.

    Parameters that used to be upper-case globals in the notebook are
    attributes here. The original names are given in parentheses.
    """

    dry_run: bool = False
    """DRY_RUN: True = local fake backend (free), False = real hardware."""

    opt_level: int = 3
    """OPT_LEVEL: transpiler aggressiveness (0-3)."""

    num_transpiles: int = 10
    """NUM_TRANSPILES: how many transpile seeds to try and score."""

    use_mapomatic: bool = False
    """USE_MAPOMATIC: additionally try mapomatic layout selection."""

    make_plots: bool = True
    """MAKE_PLOTS: draw the device-layout figure (coupling map)."""

    draw_circuit: bool = True
    """DRAW_CIRCUIT: draw circuit diagrams (logical + transpiled)."""

    fold: int = 100
    """FOLD: gates per row in diagrams (-1 = one very long row)."""

    layout_view: LayoutView = "physical"
    """LAYOUT_VIEW: "physical" = label the map with chip qubit numbers,
    "virtual" = label with logical qubit numbers (0..n-1)."""

    prune_keep: float = 0.5
    """PRUNE_KEEP: fraction of the strongest ZZ couplings to keep
    (1.0 = prune nothing)."""

    shots: int | Literal["auto"] = 300_000
    """SHOTS: number of shots on real hardware. "auto" = take the
    recommendation from :func:`tune_penalties` (shots_recommended)."""

    dry_run_shots: int = 512
    """Number of shots in DRY_RUN (the notebook used 512)."""

    # --- backend -----------------------------------------------------------
    backend_name: str | None = None
    """Force a specific backend by name. None = least_busy."""

    use_fractional_gates: bool = True
    """Allow fractional gates (native rzz) when selecting the backend."""

    # --- Sampler options ---------------------------------------------------
    dynamical_decoupling: bool = True
    dd_sequence: str = "XY4"
    twirling_gates: bool = True
    twirling_measure: bool = True

    # --- monitoring --------------------------------------------------------
    monitor: bool = True
    """Follow job progress (outside DRY_RUN only)."""

    poll_interval: float = 2.0
    """Job status polling interval in seconds."""

    # --- misc --------------------------------------------------------------
    max_twoq_warn: int = 3000
    """Print a warning above this number of two-qubit gates."""

    max_exact_n: int = 24
    """Above this N, skip the checks that cost 2^N."""

    verbose: bool = True
    """Print the log (same content as the notebook)."""

    plots: PlotConfig = field(default_factory=PlotConfig)

    def replace(self, **kwargs: Any) -> RunConfig:
        """Return a copy with the given fields changed."""
        return replace(self, **kwargs)

    def resolve_shots(self, penalties: Penalties | None = None) -> int:
        """How many shots will actually be used."""
        if self.dry_run:
            return self.dry_run_shots
        if self.shots == "auto":
            if penalties is None or penalties.shots_recommended is None:
                raise ValueError('shots="auto" requires penalties from tune_penalties()')
            return int(penalties.shots_recommended)
        return int(self.shots)

    def __post_init__(self) -> None:
        if not 0.0 < self.prune_keep <= 1.0:
            raise ValueError(f"prune_keep must be in (0, 1], got {self.prune_keep}")
        if self.layout_view not in ("physical", "virtual"):
            raise ValueError(
                f'layout_view must be "physical" or "virtual", got {self.layout_view!r}'
            )
        if self.opt_level not in (0, 1, 2, 3):
            raise ValueError(f"opt_level must be 0-3, got {self.opt_level}")


@dataclass
class Penalties:
    """Output of :func:`tune_penalties` -- cost function and schedule parameters.

    Also supports dictionary-style access (``cfg["LAM1"]``) so that notebook
    code can be carried over unchanged.
    """

    ALPHA: float = 2.0
    LAM1: float = 6.20
    LAM2: float = 0.179
    STEPS: int = 3
    T: float = 6.0

    # --- diagnostics -------------------------------------------------------
    p_opt: float | None = None
    """P(optimum) from a statevector simulation of the circuit that will run."""

    p_feas: float | None = None
    rank: int | None = None
    score: float | None = None
    F: float | None = None
    """Attenuation exp(-gate_error * overhead * RZZ), 1.0 when no gate_error given."""

    rzz: int | None = None
    optimum: int | None = None
    x_uniform: float | None = None
    lp_dual: float | None = None
    shots_recommended: int | None = None
    plateau_LAM1: tuple[float, float] | None = None
    plateau_LAM2: tuple[float, float] | None = None
    all_results: list[dict] = field(default_factory=list, repr=False)

    # backwards compatibility with the notebook's dict API
    _ALIASES = {"shots_for_10_hits": "shots_recommended"}

    def __getitem__(self, key: str) -> Any:
        key = self._ALIASES.get(key, key)
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def as_tuple(self) -> tuple[float, float, float, int, float]:
        """(ALPHA, LAM1, LAM2, STEPS, T) -- for unpacking as in the notebook."""
        return (self.ALPHA, self.LAM1, self.LAM2, self.STEPS, self.T)
