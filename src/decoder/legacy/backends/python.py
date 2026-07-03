"""Pure-Python reference sampler engine."""

from __future__ import annotations

import random
from dataclasses import replace

from ...ascii_constraints import TEXT_ASCII
from ...circuit import Assignment, Circuit, EvaluationResult
from ..relaxations import get_relaxation
from ..relaxations.base import Relaxation


class PythonSamplerEngine:
    def __init__(
        self,
        circuit: Circuit,
        assignment: Assignment,
        *,
        fixed_key: bool = True,
        seed: int | None = None,
        relaxation: str | Relaxation = "discrete",
    ) -> None:
        self.circuit = circuit
        self.assignment = assignment
        self.fixed_key = fixed_key
        self.relaxation = get_relaxation(relaxation) if isinstance(relaxation, str) else relaxation
        self.random = random.Random(seed)
        self._multipliers = [1.0] * len(circuit.constraints)
        self._temperature = 1.0
        self._feasible: list[Assignment] = []
        self._last_result = self.evaluate()

    @classmethod
    def is_available(cls) -> bool:
        return True

    def set_multipliers(self, multipliers: list[float] | tuple[float, ...]) -> None:
        if len(multipliers) != len(self.circuit.constraints):
            raise ValueError("multiplier count must match constraint count")
        self._multipliers = list(multipliers)

    def set_temperature(self, temperature: float) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self._temperature = temperature

    def evaluate(self) -> EvaluationResult:
        self._last_result = self._evaluate_assignment(self.assignment)
        if self._last_result.satisfied:
            self._feasible.append(self._copy_assignment(self.assignment))
        return self._last_result

    def residuals(self) -> tuple[float, ...]:
        return self._last_result.residuals

    def run_epoch(self, sweeps: int) -> EvaluationResult:
        for _ in range(sweeps):
            candidate = self._mutated_assignment(self.assignment)
            current = self._last_result
            candidate_result = self._evaluate_assignment(candidate)
            if self._accept(current, candidate_result):
                self.assignment = candidate
                self._last_result = candidate_result
                if candidate_result.satisfied:
                    self._feasible.append(self._copy_assignment(candidate))
        return self._last_result

    def drain_feasible(self) -> list[Assignment]:
        feasible = self._feasible
        self._feasible = []
        return feasible

    def _accept(self, current: EvaluationResult, candidate: EvaluationResult) -> bool:
        current_energy = self._energy(current)
        candidate_energy = self._energy(candidate)
        if candidate_energy <= current_energy:
            return True
        probability = pow(2.718281828459045, (current_energy - candidate_energy) / self._temperature)
        return self.random.random() < probability

    def _energy(self, result: EvaluationResult) -> float:
        return sum(multiplier * residual for multiplier, residual in zip(self._multipliers, result.residuals))

    def _evaluate_assignment(self, assignment: Assignment) -> EvaluationResult:
        base = self.circuit.evaluate(assignment)
        if getattr(self.relaxation, "uses_base_result", False):
            residuals = self.relaxation.residuals_from_result(base)
        else:
            residuals = tuple(float(residual) for residual in self.relaxation.residuals(self.circuit, assignment))
        if residuals == base.residuals:
            return base
        failing_indices = tuple(index for index, residual in enumerate(residuals) if residual != 0.0)
        failing_labels = tuple(
            self.circuit.constraints[index].label
            for index in failing_indices
            if self.circuit.constraints[index].label
        )
        return replace(
            base,
            violations=len(failing_indices),
            hamming_score=sum(residuals),
            residuals=residuals,
            failing_indices=failing_indices,
            failing_labels=failing_labels,
        )

    def _mutated_assignment(self, assignment: Assignment) -> Assignment:
        mutated = self._copy_assignment(assignment)
        if mutated.wires is not None and self.random.random() < 0.5:
            offset = self.random.randrange(len(mutated.wires))
            mutated.wires[offset] ^= 1 << self.random.randrange(8)
            return mutated

        if not self.fixed_key and self.random.random() < 0.25:
            key = mutated.key1 if self.random.random() < 0.5 else mutated.key2
            key[self.random.randrange(len(key))] ^= 1 << self.random.randrange(8)
            return mutated

        offset = self.random.randrange(len(mutated.plaintext))
        mutated.plaintext[offset] = self.random.choice(TEXT_ASCII)
        return mutated

    @staticmethod
    def _copy_assignment(assignment: Assignment) -> Assignment:
        return Assignment(
            plaintext=bytearray(assignment.plaintext),
            key1=bytearray(assignment.key1),
            key2=bytearray(assignment.key2),
            wires=bytearray(assignment.wires) if assignment.wires is not None else None,
        )
