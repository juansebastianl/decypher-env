"""Continuous byte-distribution relaxation for local factors."""

from __future__ import annotations

from ...aes_primitives import SBOX
from ...circuit import Assignment, Circuit, ConstraintKind, OpCode
from .base import ContinuousState


class ByteDistributionRelaxation:
    """Experimental local-factor relaxation over byte probability vectors.

    This is not a full AES differentiable solver. It provides distribution-space
    residuals and gradients for local byte equality factors, plus S-box
    pushforward/backprop hooks that future continuous engines can consume.
    """

    uses_base_result = False

    def residuals(self, circuit: Circuit, assignment: Assignment) -> tuple[float, ...]:
        values = circuit.evaluate_values(assignment)
        residuals = []
        for constraint in circuit.constraints:
            if constraint.kind == ConstraintKind.ASCII_PRINTABLE:
                value = values[constraint.left][0]
                residuals.append(0.0 if value in (0x09, 0x0A, 0x0D) or 0x20 <= value <= 0x7E else 1.0)
            elif constraint.kind in (ConstraintKind.EQ8, ConstraintKind.DEFINE8) and constraint.right is not None:
                residuals.append(0.0 if values[constraint.left] == values[constraint.right] else 1.0)
            elif constraint.kind == ConstraintKind.EQ128 and constraint.right is not None:
                left = values[constraint.left]
                right = values[constraint.right]
                residuals.append(sum(a != b for a, b in zip(left, right)) / max(1, len(left)))
            else:
                raise ValueError(f"unsupported constraint for continuous relaxation: {constraint.kind}")
        return tuple(residuals)

    def affected_constraints(self, circuit: Circuit, value_id: int) -> tuple[int, ...]:
        return circuit.dependencies.get(value_id, ())

    def gradients(self, circuit: Circuit, state: ContinuousState) -> dict[int, list[float]]:
        probabilities, sbox_sources = self._propagated_probabilities(circuit, state)
        gradients: dict[int, list[float]] = {value_id: [0.0] * 256 for value_id in probabilities}
        for constraint in circuit.constraints:
            if constraint.kind not in (ConstraintKind.EQ8, ConstraintKind.DEFINE8) or constraint.right is None:
                continue
            left_probs = probabilities.get(constraint.left)
            right_probs = probabilities.get(constraint.right)
            if left_probs is None or right_probs is None:
                continue
            for byte in range(256):
                delta = left_probs[byte] - right_probs[byte]
                gradients[constraint.left][byte] += delta
                gradients[constraint.right][byte] -= delta

        for output_id, input_id in sbox_sources.items():
            if input_id not in gradients:
                gradients[input_id] = [0.0] * 256
            output_gradient = gradients.get(output_id, [0.0] * 256)
            for byte in range(256):
                gradients[input_id][byte] += output_gradient[SBOX[byte]]

        return {
            value_id: gradient
            for value_id, gradient in gradients.items()
            if value_id in state.probabilities
        }

    def expected_sbox_distribution(self, input_distribution: list[float]) -> list[float]:
        if len(input_distribution) != 256:
            raise ValueError("byte distribution must have 256 probabilities")
        output = [0.0] * 256
        for byte, probability in enumerate(input_distribution):
            output[SBOX[byte]] += probability
        return output

    def initialize_from_assignment(self, circuit: Circuit, assignment: Assignment) -> ContinuousState:
        probabilities: dict[int, list[float]] = {}
        for value in circuit.values:
            byte = _assignment_byte(circuit, assignment, value.id)
            if byte is not None:
                probabilities[value.id] = _one_hot(byte)
        return ContinuousState(probabilities)

    def local_factor_expectations(self, circuit: Circuit, state: ContinuousState) -> dict[int, list[float]]:
        probabilities, _ = self._propagated_probabilities(circuit, state)
        return {
            op.output: probabilities[op.output]
            for op in circuit.ops
            if op.opcode == OpCode.SBOX8 and op.output in probabilities
        }

    def _propagated_probabilities(
        self,
        circuit: Circuit,
        state: ContinuousState,
    ) -> tuple[dict[int, list[float]], dict[int, int]]:
        probabilities = dict(state.probabilities)
        sbox_sources: dict[int, int] = {}
        # One pass is enough for the local S-box factors currently exposed.
        # Later continuous engines can replace this with graph-level AD.
        expectations: dict[int, list[float]] = {}
        for op in circuit.ops:
            if op.opcode == OpCode.SBOX8 and op.inputs[0] in state.probabilities:
                probabilities[op.output] = self.expected_sbox_distribution(state.probabilities[op.inputs[0]])
                sbox_sources[op.output] = op.inputs[0]
        return probabilities, sbox_sources


def _one_hot(byte: int) -> list[float]:
    probabilities = [0.0] * 256
    probabilities[byte] = 1.0
    return probabilities


def _assignment_byte(circuit: Circuit, assignment: Assignment, value_id: int) -> int | None:
    if value_id in circuit.wire_offsets:
        if assignment.wires is None:
            return None
        return assignment.wires[circuit.wire_offsets[value_id]]
    for name, (start, count) in circuit.input_ranges.items():
        if not (start <= value_id < start + count):
            continue
        offset = value_id - start
        if name == "plaintext":
            return assignment.plaintext[offset]
        if name == "key1":
            return assignment.key1[offset]
        if name == "key2":
            return assignment.key2[offset]
    return None
