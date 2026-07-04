"""Task-builder tests: AES primitives, circuit IR, and the XTS constraint model.

These cover the substrate the RL environment builds tasks from — no solver or
search machinery is involved.
"""

from __future__ import annotations

from pathlib import Path

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from src.decoder.aes_primitives import (
    AES_BLOCK_BYTES,
    aes256_encrypt_block,
    aes_xts_encrypt_block,
    sector_tweak_input,
    split_xts_key,
)
from src.decoder.ascii_constraints import is_text_ascii
from src.decoder.circuit import Assignment, CircuitBuilder, ConstraintKind
from src.decoder.constraints import ConstraintClass, classify_constraints
from src.decoder.encodings import KeySelector, LoweredAesEncoding
from src.decoder.xts_model import (
    assignment_from_fixture_plaintext,
    block_specs_for_fixture,
    block_specs_for_model,
    build_xts_block_collection,
    load_fixture,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_METADATA = ROOT / "src" / "testing_setup" / "lorem_ipsum_aes_xts.json"


def test_aes256_known_answer_vector() -> None:
    key = bytes.fromhex("603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4")
    plaintext = bytes.fromhex("6bc1bee22e409f96e93d7e117393172a")
    expected = bytes.fromhex("f3eed1bdb5d2a03c064b5a7e3db181f8")
    assert aes256_encrypt_block(key, plaintext) == expected


def test_lowered_aes256_encoding_accepts_known_answer_vector() -> None:
    key = bytes.fromhex("603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4")
    plaintext = bytes.fromhex("6bc1bee22e409f96e93d7e117393172a")
    expected = bytes.fromhex("f3eed1bdb5d2a03c064b5a7e3db181f8")

    builder = CircuitBuilder()
    plaintext_values = builder.input_range("plaintext", 8, AES_BLOCK_BYTES)
    builder.input_range("key1", 8, 32)
    builder.input_range("key2", 8, 32)
    output = LoweredAesEncoding().encrypt_block(builder, plaintext_values, KeySelector.KEY1, "kat")
    for index, byte in enumerate(expected):
        expected_value = builder.constant(bytes([byte]), f"expected_{index}")
        builder.constrain(ConstraintKind.EQ8, output[index], expected_value, label=f"output:{index}")
    circuit = builder.build()
    assignment = Assignment(bytearray(plaintext), bytearray(key), bytearray(32))
    circuit.derive_wire_values(assignment)
    assert circuit.evaluate(assignment).satisfied


def test_lowered_reduced_round_encoding_matches_reference() -> None:
    key = bytes(range(32))
    plaintext = bytes.fromhex("6bc1bee22e409f96e93d7e117393172a")
    expected = aes256_encrypt_block(key, plaintext, rounds=2)

    builder = CircuitBuilder()
    plaintext_values = builder.input_range("plaintext", 8, AES_BLOCK_BYTES)
    builder.input_range("key1", 8, 32)
    builder.input_range("key2", 8, 32)
    output = LoweredAesEncoding(rounds=2).encrypt_block(builder, plaintext_values, KeySelector.KEY1, "reduced")
    for index, byte in enumerate(expected):
        expected_value = builder.constant(bytes([byte]), f"expected_{index}")
        builder.constrain(ConstraintKind.EQ8, output[index], expected_value, label=f"output:{index}")
    circuit = builder.build()
    assignment = Assignment(bytearray(plaintext), bytearray(key), bytearray(32))
    circuit.derive_wire_values(assignment)

    assert circuit.evaluate(assignment).satisfied


def test_xts_block_matches_generated_fixture() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    assert fixture.key1 is not None
    assert fixture.key2 is not None
    assert fixture.plaintext is not None
    expected = fixture.ciphertext[:16]
    actual = aes_xts_encrypt_block(fixture.key1, fixture.key2, fixture.plaintext[:16], 0, 0)
    assert actual == expected


def test_xts_matches_every_generated_fixture_block() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    assert fixture.key1 is not None
    assert fixture.key2 is not None
    assert fixture.plaintext is not None

    for spec in block_specs_for_fixture(fixture):
        offset = spec.global_block_index * AES_BLOCK_BYTES
        actual = aes_xts_encrypt_block(
            fixture.key1,
            fixture.key2,
            fixture.plaintext[offset : offset + AES_BLOCK_BYTES],
            spec.sector_number,
            spec.block_index_in_sector,
        )
        assert actual == spec.ciphertext


def test_xts_matches_cryptography_for_deterministic_sectors() -> None:
    keys = [
        bytes(range(64)),
        bytes((index * 17 + 23) % 256 for index in range(64)),
    ]
    sector_numbers = [0, 1, 31, 2**32 + 7]
    sector_bytes = 512

    for key in keys:
        key1, key2 = split_xts_key(key)
        for sector_number in sector_numbers:
            plaintext = bytes((index * 13 + sector_number) % 256 for index in range(sector_bytes))
            encryptor = Cipher(
                algorithms.AES(key),
                modes.XTS(sector_tweak_input(sector_number)),
            ).encryptor()
            expected = encryptor.update(plaintext) + encryptor.finalize()
            actual = b"".join(
                aes_xts_encrypt_block(
                    key1,
                    key2,
                    plaintext[offset : offset + AES_BLOCK_BYTES],
                    sector_number,
                    offset // AES_BLOCK_BYTES,
                )
                for offset in range(0, sector_bytes, AES_BLOCK_BYTES)
            )
            assert actual == expected


def test_block_circuit_accepts_known_fixture_assignment() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=2)
    circuit = build_xts_block_collection(specs)
    assignment = assignment_from_fixture_plaintext(fixture, specs)
    result = circuit.evaluate(assignment)
    assert result.satisfied


def test_lowered_xts_block_accepts_known_fixture_assignment() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=1)
    circuit = build_xts_block_collection(specs, encoding="lowered")
    assignment = assignment_from_fixture_plaintext(fixture, specs)
    circuit.derive_wire_values(assignment)
    result = circuit.evaluate(assignment)
    assert result.satisfied
    assert circuit.wire_offsets


def test_lowered_xts_block_rejects_corrupted_internal_wire() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=1)
    circuit = build_xts_block_collection(specs, encoding="lowered")
    assignment = assignment_from_fixture_plaintext(fixture, specs)
    circuit.derive_wire_values(assignment)
    assert assignment.wires is not None
    assignment.wires[0] ^= 1
    result = circuit.evaluate(assignment)
    assert not result.satisfied
    assert result.failing_indices


def test_block_specs_for_model_preserves_fixture_ciphertext_at_full_rounds() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=3)
    model_specs = block_specs_for_model(fixture, specs, aes_rounds=14)
    assert [spec.ciphertext for spec in model_specs] == [spec.ciphertext for spec in specs]


def test_block_specs_for_model_reduced_rounds_match_reduced_round_reference() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=3)
    model_specs = block_specs_for_model(fixture, specs, aes_rounds=2)

    for spec, model_spec in zip(specs, model_specs):
        offset = spec.global_block_index * AES_BLOCK_BYTES
        expected = aes_xts_encrypt_block(
            fixture.key1,
            fixture.key2,
            fixture.plaintext[offset : offset + AES_BLOCK_BYTES],
            spec.sector_number,
            spec.block_index_in_sector,
            rounds=2,
        )
        assert model_spec.ciphertext == expected
        # Reduced rounds compute a different cipher than the 14-round fixture.
        assert model_spec.ciphertext != spec.ciphertext


def test_reduced_round_target_against_raw_fixture_is_infeasible() -> None:
    # Documents the bug this fix addresses: lowering rounds while keeping the
    # full-round fixture ciphertext leaves the goal constraints unsatisfiable
    # even for the true key and plaintext.
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=1)
    circuit = build_xts_block_collection(specs, encoding="lowered", aes_rounds=1)
    assignment = assignment_from_fixture_plaintext(fixture, specs)
    circuit.derive_wire_values(assignment)
    assert not circuit.evaluate(assignment).satisfied


def test_reduced_round_positive_control_accepts_true_key() -> None:
    # The reduced-round targets must remain satisfiable by the true key and
    # plaintext at every rung of the curriculum ladder.
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=1)
    for rounds in (1, 2, 6):
        model_specs = block_specs_for_model(fixture, specs, aes_rounds=rounds)
        circuit = build_xts_block_collection(model_specs, encoding="lowered", aes_rounds=rounds)
        assignment = assignment_from_fixture_plaintext(fixture, model_specs)
        circuit.derive_wire_values(assignment)
        result = circuit.evaluate(assignment)
        assert result.satisfied, f"true key should solve a {rounds}-round positive control"


def test_wire_derivation_uses_definitions_not_labels() -> None:
    builder = CircuitBuilder()
    wire = builder.internal_value("wire")
    definition_value = builder.constant(bytes([0x12]), "definition")
    target_value = builder.constant(bytes([0xFF]), "target")
    builder.define(wire, definition_value, label="arbitrary label")
    builder.constrain(ConstraintKind.EQ8, wire, target_value, label="goal equality")
    circuit = builder.build()
    assignment = Assignment(bytearray(), bytearray(32), bytearray(32))

    circuit.derive_wire_values(assignment)

    assert assignment.wires == bytearray([0x12])
    assert not circuit.evaluate(assignment).satisfied


def test_block_circuit_rejects_corrupted_plaintext() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=1)
    circuit = build_xts_block_collection(specs)
    assignment = assignment_from_fixture_plaintext(fixture, specs)
    assignment.plaintext[0] ^= 1
    result = circuit.evaluate(assignment)
    assert not result.satisfied
    assert result.hamming_score > 0


def test_composed_blocks_share_key_ranges() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=3)
    circuit = build_xts_block_collection(specs)
    assert circuit.input_ranges["key1"] == (48, 32)
    assert circuit.input_ranges["key2"] == (80, 32)


def test_lowered_21_block_circuit_uses_one_shared_key_buffer() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=21)
    circuit = build_xts_block_collection(specs, encoding="lowered")

    assert circuit.input_ranges["plaintext"] == (0, 21 * AES_BLOCK_BYTES)
    assert circuit.input_ranges["key1"] == (21 * AES_BLOCK_BYTES, 32)
    assert circuit.input_ranges["key2"] == (21 * AES_BLOCK_BYTES + 32, 32)
    assert len(circuit.wire_offsets) > 21 * 100


def test_constraint_classification_is_structural() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=1)
    circuit = build_xts_block_collection(specs, encoding="lowered")
    classes = classify_constraints(circuit)

    assert ConstraintClass.ASCII in classes
    assert ConstraintClass.CONSISTENCY in classes
    assert ConstraintClass.GOAL in classes
    for constraint, constraint_class in zip(circuit.constraints, classes):
        if constraint.kind == ConstraintKind.DEFINE8:
            assert constraint_class == ConstraintClass.CONSISTENCY
        elif constraint.kind == ConstraintKind.ASCII_PRINTABLE:
            assert constraint_class == ConstraintClass.ASCII
        else:
            assert constraint_class == ConstraintClass.GOAL


def test_ascii_constraints() -> None:
    assert is_text_ascii(ord("A"))
    assert is_text_ascii(0x0A)
    assert not is_text_ascii(0x00)
    # A bytes/sequence argument checks that every byte is text ASCII.
    assert is_text_ascii(b"Lorem ipsum dolo")
    assert is_text_ascii(bytearray(b"tab\tnewline\n"))
    assert not is_text_ascii(b"with null\x00byte")
    assert is_text_ascii(b"")


def test_xts_key_split() -> None:
    key = bytes(range(64))
    key1, key2 = split_xts_key(key)
    assert key1 == bytes(range(32))
    assert key2 == bytes(range(32, 64))
