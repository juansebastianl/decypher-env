"""Compact circuit IR for AES-XTS constraint sampling.

The IR is intentionally array-friendly: values are integer IDs, operations are
small records, and assignments keep plaintext/key data in bytearrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable

from .aes_primitives import AES_BLOCK_BYTES, SBOX, aes256_encrypt_block, aes_xts_encrypt_block, expand_key_256, gf_mul, xts_mul_x


class OpCode(IntEnum):
    INPUT = 1
    CONST = 2
    XOR8 = 3
    XOR128 = 4
    AES_XTS_BLOCK = 5
    EQ8 = 6
    EQ128 = 7
    ASCII_PRINTABLE = 8
    SBOX8 = 9
    MIX_COLUMN_BYTE = 10
    AES256_ROUND_KEY_BYTE = 11
    XTS_MUL_X_BYTE = 12
    AES256_BLOCK = 13
    EXTRACT_BYTE = 14


class ConstraintKind(IntEnum):
    EQ8 = 1
    EQ128 = 2
    ASCII_PRINTABLE = 3
    DEFINE8 = 4


@dataclass(frozen=True)
class Value:
    id: int
    width: int
    name: str


@dataclass(frozen=True)
class Op:
    opcode: OpCode
    output: int
    inputs: tuple[int, ...] = ()
    immediates: tuple[int, ...] = ()


@dataclass(frozen=True)
class Constraint:
    kind: ConstraintKind
    left: int
    right: int | None = None
    label: str = ""


@dataclass(frozen=True)
class XtsBlockInfo:
    """Structural description of one lowered AES-XTS block.

    Lets the native solver reconstruct a block's plaintext by decrypting its
    target ciphertext under a candidate key (the constraint-repair move).
    """

    sector_number: int
    block_index_in_sector: int
    plaintext_offset: int
    ciphertext: bytes


@dataclass
class Assignment:
    """Mutable candidate buffers consumed by the Python and C++ solver paths."""

    plaintext: bytearray
    key1: bytearray
    key2: bytearray
    wires: bytearray | None = None

    @classmethod
    def empty(cls, plaintext_bytes: int, wire_bytes: int = 0) -> "Assignment":
        return cls(
            plaintext=bytearray(plaintext_bytes),
            key1=bytearray(32),
            key2=bytearray(32),
            wires=bytearray(wire_bytes) if wire_bytes else None,
        )


@dataclass(frozen=True)
class EvaluationResult:
    constraints: int
    violations: int
    hamming_score: float
    residuals: tuple[float, ...]
    failing_indices: tuple[int, ...]
    failing_labels: tuple[str, ...]

    @property
    def satisfied(self) -> bool:
        return self.hamming_score == 0


@dataclass(frozen=True)
class Circuit:
    values: tuple[Value, ...]
    ops: tuple[Op, ...]
    constraints: tuple[Constraint, ...]
    constants: tuple[bytes, ...]
    input_ranges: dict[str, tuple[int, int]]
    dependencies: dict[int, tuple[int, ...]]
    wire_offsets: dict[int, int]
    xts_blocks: tuple[XtsBlockInfo, ...] = ()

    def evaluate(self, assignment: Assignment) -> EvaluationResult:
        evaluator = CircuitEvaluator(self, assignment)
        return evaluator.evaluate()

    def evaluate_values(self, assignment: Assignment) -> dict[int, bytes]:
        evaluator = CircuitEvaluator(self, assignment)
        return evaluator.evaluate_values()

    def derive_wire_values(self, assignment: Assignment) -> Assignment:
        """Populate internal wires by following local equality constraints."""

        if assignment.wires is None:
            assignment.wires = bytearray(len(self.wire_offsets))
        constraints_by_right: dict[int, list[Constraint]] = {}
        for constraint in self.constraints:
            if (
                constraint.kind == ConstraintKind.DEFINE8
                and constraint.right is not None
                and constraint.left in self.wire_offsets
            ):
                constraints_by_right.setdefault(constraint.right, []).append(constraint)

        evaluator = CircuitEvaluator(self, assignment)

        def assign_wire(value_id: int, value: bytes) -> None:
            if value_id not in self.wire_offsets:
                return
            assignment.wires[self.wire_offsets[value_id]] = value[0]
            evaluator.values[value_id] = value
            for dependent in constraints_by_right.get(value_id, ()):
                assign_wire(dependent.left, value)

        for op in self.ops:
            evaluator._eval_op(op)
            for constraint in constraints_by_right.get(op.output, ()):
                assign_wire(constraint.left, evaluator.values[op.output])
        return assignment

    def op_table(self) -> list[tuple[int, int, int, int, int, int, int, int]]:
        """Return fixed-width op rows suitable for Cython/C++ export."""

        rows = []
        for op in self.ops:
            inputs = tuple(op.inputs[:4]) + (-1,) * max(0, 4 - len(op.inputs))
            immediate = op.immediates[0] if op.immediates else -1
            rows.append(
                (
                    int(op.opcode),
                    op.output,
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    immediate,
                    len(op.inputs),
                )
            )
        return rows

    def constraint_table(self) -> list[tuple[int, int, int]]:
        return [
            (int(constraint.kind), constraint.left, -1 if constraint.right is None else constraint.right)
            for constraint in self.constraints
        ]


class CircuitBuilder:
    def __init__(self) -> None:
        self._values: list[Value] = []
        self._ops: list[Op] = []
        self._constraints: list[Constraint] = []
        self._constants: list[bytes] = []
        self._input_ranges: dict[str, tuple[int, int]] = {}
        self._wire_offsets: dict[int, int] = {}
        self._xts_blocks: list[XtsBlockInfo] = []

    def value(self, width: int, name: str) -> int:
        value_id = len(self._values)
        self._values.append(Value(value_id, width, name))
        return value_id

    def input_range(self, name: str, width: int, count: int) -> list[int]:
        start = len(self._values)
        ids = [self.value(width, f"{name}_{index}") for index in range(count)]
        self._input_ranges[name] = (start, count)
        for value_id in ids:
            self._ops.append(Op(OpCode.INPUT, value_id))
        return ids

    def constant(self, data: bytes, name: str) -> int:
        value_id = self.value(len(data) * 8, name)
        const_id = len(self._constants)
        self._constants.append(bytes(data))
        self._ops.append(Op(OpCode.CONST, value_id, immediates=(const_id,)))
        return value_id

    def op(self, opcode: OpCode, output_width: int, inputs: Iterable[int], name: str, *immediates: int) -> int:
        output = self.value(output_width, name)
        self._ops.append(Op(opcode, output, tuple(inputs), tuple(immediates)))
        return output

    def constrain(self, kind: ConstraintKind, left: int, right: int | None = None, label: str = "") -> None:
        self._constraints.append(Constraint(kind, left, right, label))

    def define(self, left: int, right: int, label: str = "") -> None:
        self.constrain(ConstraintKind.DEFINE8, left, right, label)

    def register_xts_block(
        self,
        sector_number: int,
        block_index_in_sector: int,
        plaintext_offset: int,
        ciphertext: bytes,
    ) -> None:
        self._xts_blocks.append(
            XtsBlockInfo(
                sector_number=sector_number,
                block_index_in_sector=block_index_in_sector,
                plaintext_offset=plaintext_offset,
                ciphertext=bytes(ciphertext),
            )
        )

    def build(self) -> Circuit:
        return Circuit(
            values=tuple(self._values),
            ops=tuple(self._ops),
            constraints=tuple(self._constraints),
            constants=tuple(self._constants),
            input_ranges=dict(self._input_ranges),
            dependencies=self._build_dependencies(),
            wire_offsets=dict(self._wire_offsets),
            xts_blocks=tuple(self._xts_blocks),
        )

    def internal_range(self, name: str, width: int, count: int) -> list[int]:
        ids = []
        for index in range(count):
            value_id = self.value(width, f"{name}_{index}")
            self._wire_offsets[value_id] = len(self._wire_offsets)
            self._ops.append(Op(OpCode.INPUT, value_id))
            ids.append(value_id)
        return ids

    def internal_value(self, name: str, width: int = 8) -> int:
        return self.internal_range(name, width, 1)[0]

    def _build_dependencies(self) -> dict[int, tuple[int, ...]]:
        dependencies: dict[int, list[int]] = {}
        producer_inputs = {op.output: op.inputs for op in self._ops}
        for constraint_index, constraint in enumerate(self._constraints):
            value_ids = [constraint.left]
            if constraint.right is not None:
                value_ids.append(constraint.right)
            expanded = set(value_ids)
            for value_id in value_ids:
                expanded.update(producer_inputs.get(value_id, ()))
            for value_id in expanded:
                dependencies.setdefault(value_id, []).append(constraint_index)
        return {value_id: tuple(indices) for value_id, indices in dependencies.items()}


class CircuitEvaluator:
    def __init__(self, circuit: Circuit, assignment: Assignment) -> None:
        self.circuit = circuit
        self.assignment = assignment
        self.values: dict[int, bytes] = {}

    def evaluate(self) -> EvaluationResult:
        self.evaluate_values()

        violations = 0
        hamming_score = 0.0
        failing_labels: list[str] = []
        residuals: list[float] = []
        failing_indices: list[int] = []
        for constraint_index, constraint in enumerate(self.circuit.constraints):
            failed, score = self._score_constraint(constraint)
            residuals.append(float(score))
            if failed:
                violations += 1
                hamming_score += score
                failing_indices.append(constraint_index)
                failing_labels.append(constraint.label)

        return EvaluationResult(
            constraints=len(self.circuit.constraints),
            violations=violations,
            hamming_score=hamming_score,
            residuals=tuple(residuals),
            failing_indices=tuple(failing_indices),
            failing_labels=tuple(label for label in failing_labels if label),
        )

    def evaluate_values(self) -> dict[int, bytes]:
        for op in self.circuit.ops:
            self._eval_op(op)
        return self.values

    def _eval_op(self, op: Op) -> None:
        if op.opcode == OpCode.INPUT:
            self.values[op.output] = self._input_value(op.output)
        elif op.opcode == OpCode.CONST:
            self.values[op.output] = self.circuit.constants[op.immediates[0]]
        elif op.opcode in (OpCode.XOR8, OpCode.XOR128):
            left, right = (self.values[input_id] for input_id in op.inputs)
            self.values[op.output] = bytes(a ^ b for a, b in zip(left, right))
        elif op.opcode == OpCode.AES_XTS_BLOCK:
            plaintext = self._concat_inputs(op.inputs[:AES_BLOCK_BYTES])
            key1 = bytes(self.assignment.key1)
            key2 = bytes(self.assignment.key2)
            sector_number, block_index = op.immediates
            self.values[op.output] = aes_xts_encrypt_block(
                key1,
                key2,
                plaintext,
                sector_number,
                block_index,
            )
        elif op.opcode == OpCode.AES256_BLOCK:
            plaintext = self._concat_inputs(op.inputs[:AES_BLOCK_BYTES])
            key_selector = op.immediates[0]
            key = bytes(self.assignment.key1 if key_selector == 1 else self.assignment.key2)
            self.values[op.output] = aes256_encrypt_block(key, plaintext)
        elif op.opcode == OpCode.EXTRACT_BYTE:
            byte_index = op.immediates[0]
            self.values[op.output] = bytes([self.values[op.inputs[0]][byte_index]])
        elif op.opcode == OpCode.SBOX8:
            self.values[op.output] = bytes([SBOX[self.values[op.inputs[0]][0]]])
        elif op.opcode == OpCode.MIX_COLUMN_BYTE:
            column = [self.values[input_id][0] for input_id in op.inputs]
            coefficients = op.immediates
            value = 0
            for coefficient, byte in zip(coefficients, column):
                value ^= gf_mul(byte, coefficient)
            self.values[op.output] = bytes([value])
        elif op.opcode == OpCode.AES256_ROUND_KEY_BYTE:
            key_selector, round_index, byte_index = op.immediates
            key = bytes(self.assignment.key1 if key_selector == 1 else self.assignment.key2)
            self.values[op.output] = bytes([expand_key_256(key)[round_index][byte_index]])
        elif op.opcode == OpCode.XTS_MUL_X_BYTE:
            byte_index = op.immediates[0]
            tweak = self._concat_inputs(op.inputs[:AES_BLOCK_BYTES])
            self.values[op.output] = bytes([xts_mul_x(tweak)[byte_index]])
        elif op.opcode in (OpCode.EQ8, OpCode.EQ128, OpCode.ASCII_PRINTABLE):
            return
        else:
            raise ValueError(f"unsupported opcode: {op.opcode}")

    def _input_value(self, value_id: int) -> bytes:
        if value_id in self.circuit.wire_offsets:
            if self.assignment.wires is None:
                raise ValueError("assignment does not include internal wire values")
            return bytes([self.assignment.wires[self.circuit.wire_offsets[value_id]]])
        for name, (start, count) in self.circuit.input_ranges.items():
            if start <= value_id < start + count:
                offset = value_id - start
                if name == "plaintext":
                    return bytes([self.assignment.plaintext[offset]])
                if name == "key1":
                    return bytes([self.assignment.key1[offset]])
                if name == "key2":
                    return bytes([self.assignment.key2[offset]])
        raise KeyError(f"value {value_id} is not an input")

    def _concat_inputs(self, input_ids: Iterable[int]) -> bytes:
        return b"".join(self.values[input_id] for input_id in input_ids)

    def _score_constraint(self, constraint: Constraint) -> tuple[bool, int]:
        left = self.values[constraint.left]
        if constraint.kind == ConstraintKind.ASCII_PRINTABLE:
            value = left[0]
            failed = not (value in (0x09, 0x0A, 0x0D) or 0x20 <= value <= 0x7E)
            return failed, 0 if not failed else 1

        if constraint.right is None:
            raise ValueError("equality constraint requires a right value")
        right = self.values[constraint.right]
        if constraint.kind == ConstraintKind.DEFINE8:
            score = (left[0] ^ right[0]).bit_count()
            return score != 0, score
        score = sum((a ^ b).bit_count() for a, b in zip(left, right))
        return score != 0, score
