"""AES-XTS block circuit builders."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from .aes_primitives import (
    AES_BLOCK_BYTES,
    AES_256_KEY_BYTES,
    aes_xts_encrypt_block,
    sector_tweak_input,
    split_xts_key,
)

AES_FULL_ROUNDS = 14
from .circuit import Assignment, Circuit, CircuitBuilder, ConstraintKind, OpCode
from .encodings import KeySelector, LoweredAesEncoding


@dataclass(frozen=True)
class Fixture:
    metadata_path: Path
    ciphertext_path: Path
    plaintext_path: Path | None
    ciphertext: bytes
    plaintext: bytes | None
    key1: bytes | None
    key2: bytes | None
    sector_bytes: int
    start_sector_number: int

    @property
    def blocks_per_sector(self) -> int:
        return self.sector_bytes // AES_BLOCK_BYTES


@dataclass(frozen=True)
class BlockSpec:
    ciphertext: bytes
    sector_number: int
    block_index_in_sector: int
    global_block_index: int


def load_fixture(metadata_path: Path) -> Fixture:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    base_dir = metadata_path.parent
    ciphertext_path = base_dir / metadata["ciphertext_file"]
    plaintext_file = metadata.get("plaintext_file")
    plaintext_path = base_dir / plaintext_file if plaintext_file else None
    xts_key = bytes.fromhex(metadata["key_hex"]) if "key_hex" in metadata else None
    key1, key2 = split_xts_key(xts_key) if xts_key is not None else (None, None)
    return Fixture(
        metadata_path=metadata_path,
        ciphertext_path=ciphertext_path,
        plaintext_path=plaintext_path,
        ciphertext=ciphertext_path.read_bytes(),
        plaintext=plaintext_path.read_bytes() if plaintext_path and plaintext_path.exists() else None,
        key1=key1,
        key2=key2,
        sector_bytes=int(metadata["sector_bytes"]),
        start_sector_number=int(metadata["start_sector_number"]),
    )


def block_specs_for_fixture(
    fixture: Fixture,
    *,
    start_block: int = 0,
    block_count: int | None = None,
) -> list[BlockSpec]:
    total_blocks = len(fixture.ciphertext) // AES_BLOCK_BYTES
    if block_count is None:
        block_count = total_blocks - start_block
    if start_block < 0 or block_count < 1 or start_block + block_count > total_blocks:
        raise ValueError("requested block window is outside the ciphertext")

    specs: list[BlockSpec] = []
    for global_block_index in range(start_block, start_block + block_count):
        sector_offset = global_block_index // fixture.blocks_per_sector
        block_index = global_block_index % fixture.blocks_per_sector
        sector_number = fixture.start_sector_number + sector_offset
        offset = global_block_index * AES_BLOCK_BYTES
        specs.append(
            BlockSpec(
                ciphertext=fixture.ciphertext[offset : offset + AES_BLOCK_BYTES],
                sector_number=sector_number,
                block_index_in_sector=block_index,
                global_block_index=global_block_index,
            )
        )
    return specs


def block_specs_for_model(
    fixture: Fixture,
    specs: list[BlockSpec],
    *,
    aes_rounds: int = AES_FULL_ROUNDS,
) -> list[BlockSpec]:
    """Return specs whose ciphertext is the correct target for ``aes_rounds``.

    The fixture ciphertext is produced by full 14-round AES-XTS, so a model that
    lowers fewer rounds computes a different cipher and can never match it. For a
    reduced-round positive control the target must be ciphertext generated with
    the same number of rounds, recomputed here from the fixture key and
    plaintext. At full rounds the original fixture ciphertext is returned
    unchanged so the model is still validated against real library output.
    """

    if aes_rounds >= AES_FULL_ROUNDS:
        return list(specs)
    if fixture.plaintext is None or fixture.key1 is None or fixture.key2 is None:
        raise ValueError("reduced-round targets require fixture plaintext and key metadata")

    rebuilt: list[BlockSpec] = []
    for spec in specs:
        offset = spec.global_block_index * AES_BLOCK_BYTES
        plaintext_block = fixture.plaintext[offset : offset + AES_BLOCK_BYTES]
        ciphertext = aes_xts_encrypt_block(
            fixture.key1,
            fixture.key2,
            plaintext_block,
            spec.sector_number,
            spec.block_index_in_sector,
            rounds=aes_rounds,
        )
        rebuilt.append(replace(spec, ciphertext=ciphertext))
    return rebuilt


def build_xts_block_collection(
    specs: list[BlockSpec],
    *,
    ascii_plaintext: bool = True,
    encoding: str = "opaque",
    relaxation: str = "discrete",
    aes_rounds: int = 14,
) -> Circuit:
    """Build a composed known-position AES-XTS model with shared key buffers."""

    if relaxation not in ("discrete", "continuous"):
        raise ValueError("relaxation must be 'discrete' or 'continuous'")
    if encoding not in ("opaque", "lowered"):
        raise ValueError("encoding must be 'opaque' or 'lowered'")
    if encoding == "lowered":
        return _build_lowered_xts_block_collection(specs, ascii_plaintext=ascii_plaintext, aes_rounds=aes_rounds)

    builder = CircuitBuilder()
    plaintext = builder.input_range("plaintext", 8, len(specs) * AES_BLOCK_BYTES)
    builder.input_range("key1", 8, AES_256_KEY_BYTES)
    builder.input_range("key2", 8, AES_256_KEY_BYTES)

    for block_number, spec in enumerate(specs):
        pt_start = block_number * AES_BLOCK_BYTES
        pt_values = plaintext[pt_start : pt_start + AES_BLOCK_BYTES]
        if ascii_plaintext:
            for byte_index, value_id in enumerate(pt_values):
                builder.constrain(
                    ConstraintKind.ASCII_PRINTABLE,
                    value_id,
                    label=f"ascii:block={spec.global_block_index}:byte={byte_index}",
                )

        ciphertext_value = builder.constant(
            spec.ciphertext,
            f"ciphertext_block_{spec.global_block_index}",
        )
        predicted = builder.op(
            OpCode.AES_XTS_BLOCK,
            AES_BLOCK_BYTES * 8,
            pt_values,
            f"predicted_ciphertext_block_{spec.global_block_index}",
            spec.sector_number,
            spec.block_index_in_sector,
        )
        builder.constrain(
            ConstraintKind.EQ128,
            predicted,
            ciphertext_value,
            label=f"ciphertext:block={spec.global_block_index}",
        )

    return builder.build()


def _build_lowered_xts_block_collection(specs: list[BlockSpec], *, ascii_plaintext: bool, aes_rounds: int = 14) -> Circuit:
    builder = CircuitBuilder()
    plaintext = builder.input_range("plaintext", 8, len(specs) * AES_BLOCK_BYTES)
    builder.input_range("key1", 8, AES_256_KEY_BYTES)
    builder.input_range("key2", 8, AES_256_KEY_BYTES)
    aes = LoweredAesEncoding(rounds=aes_rounds)
    sector_tweaks: dict[int, list[int]] = {}
    tweak_steps: dict[tuple[int, int], list[int]] = {}

    for block_number, spec in enumerate(specs):
        pt_start = block_number * AES_BLOCK_BYTES
        pt_values = plaintext[pt_start : pt_start + AES_BLOCK_BYTES]
        if ascii_plaintext:
            for byte_index, value_id in enumerate(pt_values):
                builder.constrain(
                    ConstraintKind.ASCII_PRINTABLE,
                    value_id,
                    label=f"ascii:block={spec.global_block_index}:byte={byte_index}",
                )

        if spec.sector_number not in sector_tweaks:
            sector_input = [
                builder.constant(bytes([byte]), f"sector_{spec.sector_number}_byte_{index}")
                for index, byte in enumerate(sector_tweak_input(spec.sector_number))
            ]
            sector_tweaks[spec.sector_number] = aes.encrypt_block(
                builder,
                sector_input,
                KeySelector.KEY2,
                f"sector_{spec.sector_number}:base_tweak",
            )
            tweak_steps[(spec.sector_number, 0)] = sector_tweaks[spec.sector_number]
        tweak = tweak_steps[(spec.sector_number, 0)]
        for tweak_step in range(spec.block_index_in_sector):
            next_key = (spec.sector_number, tweak_step + 1)
            if next_key not in tweak_steps:
                tweak_steps[next_key] = _xts_mul_x_wires(
                    builder,
                    tweak,
                    f"sector_{spec.sector_number}:tweak_step_{tweak_step}",
                )
            tweak = tweak_steps[next_key]

        pre_whitened = [
            _xor_wire(builder, pt_byte, tweak_byte, f"block_{spec.global_block_index}:prewhite_{index}")
            for index, (pt_byte, tweak_byte) in enumerate(zip(pt_values, tweak))
        ]
        aes_out = aes.encrypt_block(
            builder,
            pre_whitened,
            KeySelector.KEY1,
            f"block_{spec.global_block_index}:data",
        )
        for index, (aes_byte, tweak_byte, ciphertext_byte) in enumerate(zip(aes_out, tweak, spec.ciphertext)):
            cipher_wire = _xor_wire(
                builder,
                aes_byte,
                tweak_byte,
                f"block_{spec.global_block_index}:cipher_{index}",
            )
            expected = builder.constant(bytes([ciphertext_byte]), f"ciphertext_{spec.global_block_index}_{index}")
            builder.constrain(
                ConstraintKind.EQ8,
                cipher_wire,
                expected,
                label=f"ciphertext:block={spec.global_block_index}:byte={index}",
            )

        builder.register_xts_block(
            sector_number=spec.sector_number,
            block_index_in_sector=spec.block_index_in_sector,
            plaintext_offset=pt_start,
            ciphertext=spec.ciphertext,
        )

    return builder.build()


def _xor_wire(builder: CircuitBuilder, left: int, right: int, label: str) -> int:
    wire = builder.internal_value(label)
    expected = builder.op(OpCode.XOR8, 8, [left, right], f"{label}:expected")
    builder.define(wire, expected, label=f"{label}:consistent")
    return wire


def _xts_mul_x_wires(builder: CircuitBuilder, tweak: list[int], label: str) -> list[int]:
    out = []
    for index in range(AES_BLOCK_BYTES):
        wire = builder.internal_value(f"{label}:byte_{index}")
        expected = builder.op(OpCode.XTS_MUL_X_BYTE, 8, tweak, f"{label}:expected_{index}", index)
        builder.define(wire, expected, label=f"{label}:consistent:{index}")
        out.append(wire)
    return out


def assignment_from_fixture_plaintext(fixture: Fixture, specs: list[BlockSpec]) -> Assignment:
    if fixture.plaintext is None or fixture.key1 is None or fixture.key2 is None:
        raise ValueError("fixture does not contain plaintext and key metadata")
    plaintext = bytearray()
    for spec in specs:
        offset = spec.global_block_index * AES_BLOCK_BYTES
        plaintext.extend(fixture.plaintext[offset : offset + AES_BLOCK_BYTES])
    return Assignment(
        plaintext=plaintext,
        key1=bytearray(fixture.key1),
        key2=bytearray(fixture.key2),
    )
