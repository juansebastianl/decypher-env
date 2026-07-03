"""Backend protocol and data export helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ...circuit import Assignment, Circuit, ConstraintKind, EvaluationResult
from ...constraints import ConstraintClass, classify_constraints  # re-exported for compatibility


@dataclass(frozen=True)
class ParallelTemperingConfig:
    replicas: int = 16
    t_min: float = 0.5
    t_max: float = 20.0
    sweeps_per_epoch: int = 1000
    mu: float = 0.1
    dual_eta: float = 0.01
    goal_weight: float = 0.1
    consistency_weight: float = 1.0
    ascii_weight: float = 1.0
    seed: int | None = None
    threads: int = 0
    profile: bool = False
    validate_python_epoch: bool = False
    ladder_mode: str = "geometric"
    ladder_adapt_interval_epochs: int = 5
    ladder_burn_in_epochs: int = 5
    ladder_min_round_trips: int = 1
    dual_mode: str = "fixed"
    scheduled_rho_initial: float = 0.1
    scheduled_eta_tol: float = 0.5
    scheduled_eta_target: float = 1e-3
    scheduled_rho_growth: float = 2.0
    scheduled_violation_shrink: float = 0.75
    diagnostics: bool = True
    algebra_diagnostics: bool = False
    bp_diagnostics: bool = False
    alternative_diagnostics: bool = False
    marginal_burn_in_epochs: int = 0
    marginal_window: int = 4096
    marginal_alpha: float = 0.5
    marginal_min_distinct_keys: int = 25
    lambda_scale_cold: float = 1.0
    lambda_scale_hot: float = 1.0
    aes_rounds: int = 14
    # Multiplier applied to inter-round consistency (DEFINE8) residuals. The RL
    # curriculum ramps this from small (rounds drift independently, each round
    # solvable on its own) to large (rounds forced to match) via the
    # method-of-multipliers schedule in src/decoder/rl/curriculum.py.
    coupling_weight: float = 1.0
    repair_move_prob: float = 0.0
    key_gibbs_prob: float = 0.0
    bp_iterations: int = 20
    bp_damping: float = 0.35
    bp_tolerance: float = 1e-4
    bp_proposal_weight: float = 0.25
    bethe_weight: float = 0.0
    algebraic_newton_prob: float = 0.0
    survey_restarts: int = 0


@dataclass(frozen=True)
class SharedKeySample:
    plaintext: bytes
    key1: bytes
    key2: bytes
    wires: bytes | None
    energy: float
    residual_sum: float


@dataclass
class ParallelTemperingMetrics:
    epochs: int = 0
    sweeps: int = 0
    feasible_count: int = 0
    swap_attempts: list[int] = field(default_factory=list)
    swap_accepts: list[int] = field(default_factory=list)
    residuals_by_class: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CircuitBuffers:
    op_rows: tuple[tuple[int, int, int, int, int, int, int, int], ...]
    constraint_rows: tuple[tuple[int, int, int], ...]
    value_widths: tuple[int, ...]
    constants: tuple[bytes, ...]


def export_circuit_buffers(circuit: Circuit) -> CircuitBuffers:
    return CircuitBuffers(
        op_rows=tuple(circuit.op_table()),
        constraint_rows=tuple(circuit.constraint_table()),
        value_widths=tuple(value.width for value in circuit.values),
        constants=circuit.constants,
    )


class SamplerEngine(Protocol):
    circuit: Circuit
    assignment: Assignment

    @classmethod
    def is_available(cls) -> bool: ...

    def set_multipliers(self, multipliers: list[float] | tuple[float, ...]) -> None: ...

    def set_temperature(self, temperature: float) -> None: ...

    def evaluate(self) -> EvaluationResult: ...

    def residuals(self) -> tuple[float, ...]: ...

    def run_epoch(self, sweeps: int) -> EvaluationResult: ...

    def drain_feasible(self) -> list[Assignment]: ...
