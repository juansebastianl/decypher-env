"""Relaxation protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ...circuit import Assignment, Circuit, EvaluationResult


@dataclass
class ContinuousState:
    """Probability vectors over byte values, keyed by circuit value ID."""

    probabilities: dict[int, list[float]]


class Relaxation(Protocol):
    uses_base_result: bool

    def residuals_from_result(self, result: EvaluationResult) -> tuple[float, ...]: ...

    def residuals(self, circuit: Circuit, assignment: Assignment) -> tuple[float, ...]: ...

    def affected_constraints(self, circuit: Circuit, value_id: int) -> tuple[int, ...]: ...

    def gradients(self, circuit: Circuit, state: ContinuousState) -> dict[int, list[float]]: ...
