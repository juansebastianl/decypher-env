"""Discrete byte-valued local residuals."""

from __future__ import annotations

from ...circuit import Assignment, Circuit, EvaluationResult


class DiscreteLocalRelaxation:
    uses_base_result = True

    def residuals_from_result(self, result: EvaluationResult) -> tuple[float, ...]:
        return result.residuals

    def residuals(self, circuit: Circuit, assignment: Assignment) -> tuple[float, ...]:
        return tuple(float(residual) for residual in circuit.evaluate(assignment).residuals)

    def affected_constraints(self, circuit: Circuit, value_id: int) -> tuple[int, ...]:
        return circuit.dependencies.get(value_id, ())

    def gradients(self, circuit, state):  # noqa: ANN001
        raise NotImplementedError("discrete relaxation does not expose continuous gradients")
