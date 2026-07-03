"""Lowered AES-256 encoding with local per-byte constraints."""

from __future__ import annotations

from ..aes_primitives import AES_BLOCK_BYTES
from ..circuit import CircuitBuilder, OpCode
from .base import KeySelector


MIX_COLUMN_COEFFICIENTS = (
    (2, 3, 1, 1),
    (1, 2, 3, 1),
    (1, 1, 2, 3),
    (3, 1, 1, 2),
)


class LoweredAesEncoding:
    def __init__(self, rounds: int = 14) -> None:
        if rounds < 1 or rounds > 14:
            raise ValueError("AES rounds must be between 1 and 14")
        self.rounds = rounds
        self._round_key_cache: dict[tuple[int, KeySelector], list[list[int]]] = {}

    def encrypt_block(
        self,
        builder: CircuitBuilder,
        block: list[int],
        key_selector: KeySelector,
        label: str,
    ) -> list[int]:
        round_keys = self._round_keys(builder, key_selector)

        state = self._add_round_key(builder, block, round_keys[0], f"{label}:round_0")
        for round_index in range(1, self.rounds):
            state = self._sub_bytes(builder, state, f"{label}:round_{round_index}:sub")
            state = self._shift_rows(builder, state, f"{label}:round_{round_index}:shift")
            state = self._mix_columns(builder, state, f"{label}:round_{round_index}:mix")
            state = self._add_round_key(builder, state, round_keys[round_index], f"{label}:round_{round_index}:ark")

        state = self._sub_bytes(builder, state, f"{label}:round_{self.rounds}:sub")
        state = self._shift_rows(builder, state, f"{label}:round_{self.rounds}:shift")
        return self._add_round_key(builder, state, round_keys[self.rounds], f"{label}:round_{self.rounds}:ark")

    def _round_keys(self, builder: CircuitBuilder, key_selector: KeySelector) -> list[list[int]]:
        cache_key = (id(builder), key_selector)
        cached = self._round_key_cache.get(cache_key)
        if cached is not None:
            return cached
        label = f"shared_key_{int(key_selector)}"
        round_keys = [
            [
                self._round_key_byte(builder, key_selector, round_index, byte_index, label)
                for byte_index in range(AES_BLOCK_BYTES)
            ]
            for round_index in range(15)
        ]
        self._round_key_cache[cache_key] = round_keys
        return round_keys

    def _round_key_byte(
        self,
        builder: CircuitBuilder,
        key_selector: KeySelector,
        round_index: int,
        byte_index: int,
        label: str,
    ) -> int:
        wire = builder.internal_value(f"{label}:rk_{round_index}_{byte_index}")
        expected = builder.op(
            OpCode.AES256_ROUND_KEY_BYTE,
            8,
            [],
            f"{label}:rk_expected_{round_index}_{byte_index}",
            int(key_selector),
            round_index,
            byte_index,
        )
        builder.define(wire, expected, label=f"{label}:rk_consistent:{round_index}:{byte_index}")
        return wire

    def _add_round_key(
        self,
        builder: CircuitBuilder,
        state: list[int],
        round_key: list[int],
        label: str,
    ) -> list[int]:
        out = []
        for index, (state_byte, key_byte) in enumerate(zip(state, round_key)):
            wire = builder.internal_value(f"{label}:ark_{index}")
            expected = builder.op(OpCode.XOR8, 8, [state_byte, key_byte], f"{label}:ark_expected_{index}")
            builder.define(wire, expected, label=f"{label}:ark_consistent:{index}")
            out.append(wire)
        return out

    def _sub_bytes(self, builder: CircuitBuilder, state: list[int], label: str) -> list[int]:
        out = []
        for index, state_byte in enumerate(state):
            wire = builder.internal_value(f"{label}:sbox_{index}")
            expected = builder.op(OpCode.SBOX8, 8, [state_byte], f"{label}:sbox_expected_{index}")
            builder.define(wire, expected, label=f"{label}:sbox_consistent:{index}")
            out.append(wire)
        return out

    def _shift_rows(self, builder: CircuitBuilder, state: list[int], label: str) -> list[int]:
        out = [0] * AES_BLOCK_BYTES
        for row in range(4):
            for col in range(4):
                output_index = row + 4 * col
                input_index = row + 4 * ((col + row) % 4)
                wire = builder.internal_value(f"{label}:shift_{output_index}")
                builder.define(wire, state[input_index], label=f"{label}:shift_consistent:{output_index}")
                out[output_index] = wire
        return out

    def _mix_columns(self, builder: CircuitBuilder, state: list[int], label: str) -> list[int]:
        out = [0] * AES_BLOCK_BYTES
        for col in range(4):
            column = state[4 * col : 4 * col + 4]
            for row, coefficients in enumerate(MIX_COLUMN_COEFFICIENTS):
                output_index = 4 * col + row
                wire = builder.internal_value(f"{label}:mix_{output_index}")
                expected = builder.op(
                    OpCode.MIX_COLUMN_BYTE,
                    8,
                    column,
                    f"{label}:mix_expected_{output_index}",
                    *coefficients,
                )
                builder.define(wire, expected, label=f"{label}:mix_consistent:{output_index}")
                out[output_index] = wire
        return out
