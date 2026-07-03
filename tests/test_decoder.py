from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from src.decoder.aes_primitives import (
    AES_BLOCK_BYTES,
    SBOX,
    aes256_encrypt_block,
    aes_xts_encrypt_block,
    sector_tweak_input,
    split_xts_key,
)
from src.decoder.ascii_constraints import is_text_ascii
from src.decoder.legacy.backends import available_backends, get_backend
from src.decoder.legacy.backends.base import ConstraintClass, ParallelTemperingConfig, classify_constraints, export_circuit_buffers
from src.decoder.legacy.backends.native import NativeAlgebraicRelaxedEngine, NativeContinuousRelaxedEngine, NativeParallelTemperingEngine
from src.decoder.legacy.backends.python import PythonSamplerEngine
from src.decoder.circuit import Assignment, CircuitBuilder, ConstraintKind
from src.decoder.encodings import KeySelector, LoweredAesEncoding
from src.decoder.legacy.relaxations import ByteDistributionRelaxation, ContinuousState, DiscreteLocalRelaxation
from src.decoder.legacy.sampler import SamplerConfig, XtsSampler
from src.testing_setup.run_shared_key_sampler import build_parser as build_shared_key_runner_parser
from src.testing_setup.run_window_fusion_sampler import (
    BlockMarginal,
    _add_native_key_marginals_to_blocks,
    _add_sample_to_block_marginals,
    _bytes_to_bits,
    _native_engine_class,
    _parse_args as parse_fusion_runner_args,
    build_parser as build_fusion_runner_parser,
)
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


def test_reduced_round_sampler_accepts_true_key() -> None:
    for rounds in (1, 2, 6):
        sampler = XtsSampler(
            SamplerConfig(
                metadata_path=FIXTURE_METADATA,
                block_count=1,
                encoding="lowered",
                aes_rounds=rounds,
            )
        )
        assignment = sampler.known_fixture_assignment()
        result = sampler.circuit.evaluate(assignment)
        assert result.satisfied, f"true key should solve a {rounds}-round positive control"


def test_repair_move_recovers_true_plaintext_from_key() -> None:
    # With the true key fixed, the constraint-repair move should reconstruct the
    # block plaintext exactly (decrypting the target ciphertext), yielding a
    # feasible sample within a single epoch even from a corrupted plaintext start.
    for rounds in (1, 2, 6):
        sampler = XtsSampler(
            SamplerConfig(
                metadata_path=FIXTURE_METADATA,
                block_count=1,
                encoding="lowered",
                aes_rounds=rounds,
            )
        )
        assignment = sampler.known_fixture_assignment()
        # Corrupt the plaintext so the goal constraints start violated; keep the true key.
        for index in range(len(assignment.plaintext)):
            assignment.plaintext[index] = 0x00
        sampler.circuit.derive_wire_values(assignment)
        assert not sampler.circuit.evaluate(assignment).satisfied

        engine = NativeParallelTemperingEngine(
            sampler.circuit,
            assignment,
            config=ParallelTemperingConfig(
                replicas=2,
                seed=7,
                sweeps_per_epoch=50,
                aes_rounds=rounds,
                repair_move_prob=1.0,
            ),
        )
        engine.native_engine.run_epoch(1)
        feasible = engine.native_engine.drain_feasible(8)
        assert feasible, f"repair move should harvest a feasible sample at {rounds} rounds"
        recovered = feasible[0]
        expected = sampler.known_fixture_assignment()
        assert bytes(recovered.plaintext) == bytes(expected.plaintext)


def test_repair_move_disabled_without_xts_metadata() -> None:
    # A plain (non-XTS) circuit carries no block metadata, so enabling the repair
    # move must be a safe no-op rather than reading out-of-bounds.
    builder = CircuitBuilder()
    builder.input_range("plaintext", 8, AES_BLOCK_BYTES)
    builder.input_range("key1", 8, 32)
    builder.input_range("key2", 8, 32)
    wire = builder.internal_value("wire")
    target = builder.constant(bytes([0x41]), "target")
    builder.define(wire, target, label="define")
    circuit = builder.build()
    assert circuit.xts_blocks == ()
    assignment = Assignment(bytearray(AES_BLOCK_BYTES), bytearray(32), bytearray(32))
    circuit.derive_wire_values(assignment)
    engine = NativeParallelTemperingEngine(
        circuit,
        assignment,
        config=ParallelTemperingConfig(replicas=2, seed=1, sweeps_per_epoch=10, repair_move_prob=1.0),
    )
    engine.native_engine.run_epoch(1)


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


def test_lowered_sampler_score_derives_missing_wire_buffer() -> None:
    sampler = XtsSampler(
        SamplerConfig(
            metadata_path=FIXTURE_METADATA,
            block_count=1,
            encoding="lowered",
        )
    )
    assignment = assignment_from_fixture_plaintext(sampler.fixture, sampler.specs)
    assert assignment.wires is None

    result = sampler.score(assignment)

    assert assignment.wires is not None
    assert result.satisfied


def test_lowered_sampler_random_assignment_allocates_wires() -> None:
    sampler = XtsSampler(
        SamplerConfig(
            metadata_path=FIXTURE_METADATA,
            block_count=1,
            encoding="lowered",
            seed=7,
        )
    )

    assignment = sampler.random_assignment()

    assert assignment.wires is not None
    assert len(assignment.wires) == len(sampler.circuit.wire_offsets)


def test_lowered_sampler_run_keeps_wire_buffer() -> None:
    sampler = XtsSampler(
        SamplerConfig(
            metadata_path=FIXTURE_METADATA,
            block_count=1,
            encoding="lowered",
            seed=7,
        )
    )

    state = sampler.run(iterations=2)

    assert state.assignment.wires is not None
    assert len(state.assignment.wires) == len(sampler.circuit.wire_offsets)


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


def test_bridge_exports_stable_tables() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=1)
    circuit = build_xts_block_collection(specs)
    buffers = export_circuit_buffers(circuit)
    assert len(buffers.op_rows) == len(circuit.ops)
    assert len(buffers.constraint_rows) == len(circuit.constraints)
    assert len(buffers.value_widths) == len(circuit.values)
    assert buffers.constants[0] == fixture.ciphertext[:16]


def test_discrete_relaxation_reports_local_dependencies() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=1)
    circuit = build_xts_block_collection(specs, encoding="lowered")
    assignment = assignment_from_fixture_plaintext(fixture, specs)
    circuit.derive_wire_values(assignment)
    relaxation = DiscreteLocalRelaxation()
    wire_id = next(iter(circuit.wire_offsets))
    affected = relaxation.affected_constraints(circuit, wire_id)
    assert affected
    assert len(affected) < len(circuit.constraints)
    assert sum(relaxation.residuals(circuit, assignment)) == 0.0


def test_continuous_relaxation_sbox_distribution() -> None:
    relaxation = ByteDistributionRelaxation()
    input_distribution = [0.0] * 256
    input_distribution[0x53] = 1.0
    output_distribution = relaxation.expected_sbox_distribution(input_distribution)
    assert output_distribution[SBOX[0x53]] == 1.0
    assert sum(output_distribution) == 1.0


def test_continuous_relaxation_gradient_shape() -> None:
    builder = CircuitBuilder()
    left = builder.input_range("plaintext", 8, 1)[0]
    right = builder.constant(bytes([0x42]), "expected")
    builder.constrain(ConstraintKind.EQ8, left, right, label="eq")
    circuit = builder.build()
    state = ContinuousState(
        {
            left: [1.0 if byte == 0x41 else 0.0 for byte in range(256)],
            right: [1.0 if byte == 0x42 else 0.0 for byte in range(256)],
        }
    )
    gradients = ByteDistributionRelaxation().gradients(circuit, state)
    assert set(gradients) == {left, right}
    assert len(gradients[left]) == 256


def test_continuous_relaxation_uses_distribution_divergence() -> None:
    builder = CircuitBuilder()
    left = builder.input_range("plaintext", 8, 1)[0]
    right = builder.constant(bytes([0x42]), "expected")
    builder.constrain(ConstraintKind.EQ8, left, right, label="eq")
    circuit = builder.build()
    assignment = Assignment(bytearray([0x41]), bytearray(32), bytearray(32))
    discrete = DiscreteLocalRelaxation().residuals(circuit, assignment)
    continuous = ByteDistributionRelaxation().residuals(circuit, assignment)
    assert discrete == (2.0,)
    assert continuous == (1.0,)


def test_discrete_engine_reuses_base_evaluation(monkeypatch) -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=1)
    circuit = build_xts_block_collection(specs)
    assignment = assignment_from_fixture_plaintext(fixture, specs)

    def fail_if_called(self, circuit, assignment):  # noqa: ANN001
        raise AssertionError("discrete relaxation should reuse base evaluation")

    monkeypatch.setattr(DiscreteLocalRelaxation, "residuals", fail_if_called)
    engine = PythonSamplerEngine(circuit, assignment, relaxation="discrete")

    assert engine.evaluate().satisfied


def test_python_engine_preserves_and_mutates_wires() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=1)
    circuit = build_xts_block_collection(specs, encoding="lowered")
    assignment = assignment_from_fixture_plaintext(fixture, specs)
    circuit.derive_wire_values(assignment)
    engine = PythonSamplerEngine(circuit, assignment, seed=0)
    copied = engine._copy_assignment(assignment)
    assert copied.wires == assignment.wires
    mutated = engine._mutated_assignment(assignment)
    assert mutated.wires is not None


def test_backend_registry_auto_returns_available_engine() -> None:
    assert "python" in available_backends()
    engine_cls = get_backend("auto")
    assert engine_cls.is_available()
    assert getattr(engine_cls, "supports_native_residuals", True)


def test_continuous_relaxed_backend_is_separate_from_native_pt() -> None:
    assert "native-pt" in available_backends()
    assert "native-continuous-relaxed" in available_backends()
    assert get_backend("native-pt") is NativeParallelTemperingEngine
    assert get_backend("native-continuous-relaxed") is NativeContinuousRelaxedEngine


def test_algebraic_relaxed_backend_is_registered() -> None:
    assert "native-algebraic-relaxed" in available_backends()
    assert get_backend("native-algebraic-relaxed") is NativeAlgebraicRelaxedEngine


def _tiny_continuous_relaxed_circuit():
    builder = CircuitBuilder()
    plaintext = builder.input_range("plaintext", 8, 1)
    builder.input_range("key1", 8, 32)
    builder.input_range("key2", 8, 32)
    wire = builder.internal_value("wire")
    target = builder.constant(bytes([0x41]), "target")
    builder.define(wire, target, label="define target")
    builder.constrain(ConstraintKind.ASCII_PRINTABLE, plaintext[0], label="ascii plaintext")
    builder.constrain(ConstraintKind.EQ8, wire, target, label="goal")
    return builder.build()


def test_continuous_relaxed_engine_smoke_and_diagnostics() -> None:
    circuit = _tiny_continuous_relaxed_circuit()
    assignment = Assignment(bytearray([0x00]), bytearray(32), bytearray(32))
    circuit.derive_wire_values(assignment)
    engine = NativeContinuousRelaxedEngine(
        circuit,
        assignment,
        config=ParallelTemperingConfig(replicas=2, seed=101, dual_eta=0.1),
    )

    score = engine.run_epoch(1)
    metrics = engine.metrics()

    assert score["residuals"]
    assert metrics["sweeps"] == 1
    assert metrics["langevin_available"]
    assert metrics["alternative_available"]
    assert metrics["algebra_summary"]["gf2_elimination_implemented"]
    assert sum(metrics["proposal_attempts"].values()) >= 1


def test_continuous_relaxed_residuals_match_python_hard_score() -> None:
    circuit = _tiny_continuous_relaxed_circuit()
    assignment = Assignment(bytearray([0x41]), bytearray(32), bytearray(32))
    circuit.derive_wire_values(assignment)
    engine = NativeContinuousRelaxedEngine(
        circuit,
        assignment,
        config=ParallelTemperingConfig(replicas=1, seed=102),
    )

    native = engine.native_engine.score_assignment(assignment)
    python = circuit.evaluate(assignment)

    assert tuple(native["residuals"]) == tuple(python.residuals)
    assert native["hamming_score"] == python.hamming_score


def test_algebraic_relaxed_engine_hard_score_and_bp_metrics() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=1)
    model_specs = block_specs_for_model(fixture, specs, aes_rounds=1)
    circuit = build_xts_block_collection(model_specs, encoding="lowered", aes_rounds=1)
    assignment = assignment_from_fixture_plaintext(fixture, model_specs)
    circuit.derive_wire_values(assignment)
    engine = NativeAlgebraicRelaxedEngine(
        circuit,
        assignment,
        config=ParallelTemperingConfig(
            replicas=1,
            seed=103,
            aes_rounds=1,
            bp_iterations=1,
            algebra_diagnostics=True,
            bp_diagnostics=True,
            survey_restarts=2,
        ),
    )

    native = engine.native_engine.score_assignment(assignment)
    python = circuit.evaluate(assignment)
    metrics = engine.metrics()

    assert tuple(native["residuals"]) == tuple(python.residuals)
    assert native["hamming_score"] == python.hamming_score
    assert metrics["bp_available"]
    assert len(metrics["bp_key_marginals"]) == 64 * 256
    assert metrics["algebra_summary"]["sbox_lift_count"] > 0
    assert metrics["algebra_summary"]["gf256_newton_rank"] > 0
    assert metrics["bethe_free_energy"] <= metrics["bp_entropy"]
    assert metrics["survey_restarts"] == 2
    assert "bp_guided_key_byte" in metrics["proposal_attempts"]


def test_continuous_relaxed_parallel_threads_and_marginals_populate_without_feasible_samples() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=1)
    circuit = build_xts_block_collection(specs, encoding="lowered")
    assignment = Assignment.empty(AES_BLOCK_BYTES, len(circuit.wire_offsets))
    assignment.plaintext[:] = b"A" * AES_BLOCK_BYTES
    assignment.key1[:] = bytes(range(32))
    assignment.key2[:] = bytes(range(32, 64))
    assert assignment.wires is not None
    assignment.wires[:] = bytes((index * 17) % 256 for index in range(len(assignment.wires)))
    engine = NativeContinuousRelaxedEngine(
        circuit,
        assignment,
        config=ParallelTemperingConfig(replicas=4, threads=2, seed=141),
    )

    assert engine.drain_feasible() == []
    # Cold-replica marginal recording only starts after the first sweep, so a
    # 2-sweep run leaves a single recording opportunity whose outcome hinges on
    # the exact energy landscape (and thus the fixture's ciphertext constants).
    # Use enough sweeps that at least one key visit is recorded for any fixture.
    engine.run_epoch(8)
    metrics = engine.metrics()

    assert metrics["native_summary"]["graph_counts"]["threads"] == 2
    assert metrics["key_visit_count"] >= 1
    assert "key_distinct_count" in metrics


def test_continuous_relaxed_two_round_profile_gibbs_runs_on_realistic_window(monkeypatch) -> None:
    monkeypatch.setenv("CR_PROFILE_K1_SWEEPS", "1")
    monkeypatch.setenv("CR_PROFILE_K2_PROB", "1.0")
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=10)
    model_specs = block_specs_for_model(fixture, specs, aes_rounds=2)
    circuit = build_xts_block_collection(model_specs, encoding="lowered", aes_rounds=2)
    assignment = Assignment.empty(AES_BLOCK_BYTES * len(model_specs), len(circuit.wire_offsets))
    assignment.plaintext[:] = b"A" * len(assignment.plaintext)
    assignment.key1[:] = bytes((index * 17 + 5) % 256 for index in range(32))
    assignment.key2[:] = bytes((index * 29 + 11) % 256 for index in range(32))
    assert assignment.wires is not None
    assignment.wires[:] = bytes((index * 7) % 256 for index in range(len(assignment.wires)))
    circuit.derive_wire_values(assignment)

    engine = NativeContinuousRelaxedEngine(
        circuit,
        assignment,
        config=ParallelTemperingConfig(
            replicas=1,
            seed=210,
            sweeps_per_epoch=1,
            aes_rounds=2,
            key_gibbs_prob=1.0,
            repair_move_prob=0.0,
        ),
    )

    score = engine.run_epoch(1)
    metrics = engine.metrics()
    current = engine.native_engine.current_assignment()
    rescored = engine.native_engine.score_assignment(current)

    assert metrics["proposal_attempts"]["key_tweak_profile_gibbs"] >= 1
    assert score["hamming_score"] == rescored["hamming_score"]
    assert tuple(score["residuals"]) == tuple(rescored["residuals"])


def test_continuous_relaxed_key_word_swap_proposal_is_counted() -> None:
    circuit = _tiny_continuous_relaxed_circuit()
    assignment = Assignment(bytearray([0x00]), bytearray(range(32)), bytearray(range(32, 64)))
    circuit.derive_wire_values(assignment)
    engine = NativeContinuousRelaxedEngine(
        circuit,
        assignment,
        config=ParallelTemperingConfig(replicas=1, seed=211, sweeps_per_epoch=200, key_gibbs_prob=0.0),
    )

    engine.run_epoch(200)
    metrics = engine.metrics()

    assert metrics["proposal_attempts"]["key_word_swap"] > 0


def test_continuous_relaxed_two_round_inversion_smoke_attempts_realistic_window(monkeypatch) -> None:
    monkeypatch.setenv("CR_PROFILE_K1_SWEEPS", "1")
    monkeypatch.setenv("CR_PROFILE_K2_PROB", "0.0")
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=10)
    model_specs = block_specs_for_model(fixture, specs, aes_rounds=2)
    circuit = build_xts_block_collection(model_specs, encoding="lowered", aes_rounds=2)
    assignment = Assignment.empty(AES_BLOCK_BYTES * len(model_specs), len(circuit.wire_offsets))
    assignment.plaintext[:] = b"?" * len(assignment.plaintext)
    assignment.key1[:] = bytes((index * 37 + 19) % 256 for index in range(32))
    assignment.key2[:] = bytes((index * 41 + 23) % 256 for index in range(32))
    assert assignment.wires is not None
    assignment.wires[:] = bytes((index * 13 + 3) % 256 for index in range(len(assignment.wires)))
    circuit.derive_wire_values(assignment)
    truth = assignment_from_fixture_plaintext(fixture, model_specs)

    engine = NativeContinuousRelaxedEngine(
        circuit,
        assignment,
        config=ParallelTemperingConfig(
            replicas=2,
            threads=1,
            seed=212,
            sweeps_per_epoch=4,
            aes_rounds=2,
            key_gibbs_prob=0.75,
        ),
    )

    initial = engine.native_engine.score_assignment(assignment)
    for _ in range(3):
        engine.run_epoch(4)
    current = engine.native_engine.current_assignment()
    final = engine.native_engine.score_assignment(current)
    metrics = engine.metrics()
    feasible = engine.drain_feasible()
    key_distance = sum((a ^ b).bit_count() for a, b in zip(bytes(current.key1) + bytes(current.key2), bytes(truth.key1) + bytes(truth.key2)))
    diagnostics = {
        "initial_hamming": initial["hamming_score"],
        "final_hamming": final["hamming_score"],
        "key_distance": key_distance,
        "feasible_count": len(feasible),
        "proposal_attempts": metrics["proposal_attempts"],
    }

    assert metrics["proposal_attempts"]["key_tweak_profile_gibbs"] > 0, diagnostics
    assert final["hamming_score"] >= 0, diagnostics
    assert tuple(final["residuals"]) == tuple(engine.native_engine.score_assignment(current)["residuals"])
    assert len(metrics["key_ones"]) == 512
    assert metrics["key_marginal_max_deviation"] is not None
    assert metrics["key_information_bits"] >= 0.0
    assert metrics["key_information_bits_raw"] >= metrics["key_information_bits"]
    assert metrics["key_information_null_bits"] >= 0.0


def test_backends_agree_with_python_reference() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=2)
    circuit = build_xts_block_collection(specs)
    assignment = assignment_from_fixture_plaintext(fixture, specs)
    reference = PythonSamplerEngine(circuit, assignment).evaluate()

    for backend_name in available_backends():
        engine_cls = get_backend(backend_name)
        if not getattr(engine_cls, "supports_native_residuals", True):
            continue
        engine = engine_cls(circuit, assignment_from_fixture_plaintext(fixture, specs))
        result = engine.evaluate()
        assert result.residuals == reference.residuals
        assert result.failing_indices == reference.failing_indices
        assert result.hamming_score == reference.hamming_score


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


def test_native_pt_residuals_match_python_for_random_lowered_assignments() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=1)
    circuit = build_xts_block_collection(specs, encoding="lowered")
    base = assignment_from_fixture_plaintext(fixture, specs)
    circuit.derive_wire_values(base)
    engine = NativeParallelTemperingEngine(
        circuit,
        base,
        config=ParallelTemperingConfig(replicas=1, seed=3),
    )

    for case in range(6):
        assignment = Assignment.empty(AES_BLOCK_BYTES, len(circuit.wire_offsets))
        assignment.plaintext[:] = bytes((case * 17 + index) % 95 + 0x20 for index in range(AES_BLOCK_BYTES))
        assignment.key1[:] = bytes((case * 13 + index) % 256 for index in range(32))
        assignment.key2[:] = bytes((case * 19 + index) % 256 for index in range(32))
        assert assignment.wires is not None
        assignment.wires[:] = bytes((case * 23 + index) % 256 for index in range(len(assignment.wires)))
        native = engine.native_engine.score_assignment(assignment)
        python = circuit.evaluate(assignment)
        assert tuple(native["residuals"]) == tuple(python.residuals)
        assert tuple(native["failing_indices"]) == tuple(python.failing_indices)


def test_native_pt_hot_path_does_not_call_python_evaluator(monkeypatch) -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=1)
    circuit = build_xts_block_collection(specs, encoding="lowered")
    assignment = assignment_from_fixture_plaintext(fixture, specs)
    circuit.derive_wire_values(assignment)
    engine = NativeParallelTemperingEngine(
        circuit,
        assignment,
        config=ParallelTemperingConfig(replicas=1, seed=13),
    )

    def fail_evaluate(_self, _assignment):  # noqa: ANN001
        raise AssertionError("native PT hot path called Python circuit.evaluate")

    monkeypatch.setattr(type(circuit), "evaluate", fail_evaluate)
    native = engine.native_engine.score_assignment(assignment)
    engine.native_engine.run_epoch(1)

    assert native["residuals"]


def test_native_pt_wrapper_epoch_skips_python_evaluator_by_default(monkeypatch) -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=1)
    circuit = build_xts_block_collection(specs, encoding="lowered")
    assignment = assignment_from_fixture_plaintext(fixture, specs)
    circuit.derive_wire_values(assignment)
    engine = NativeParallelTemperingEngine(
        circuit,
        assignment,
        config=ParallelTemperingConfig(replicas=1, seed=17),
    )

    def fail_evaluate(_self, _assignment):  # noqa: ANN001
        raise AssertionError("native PT wrapper epoch called Python circuit.evaluate")

    monkeypatch.setattr(type(circuit), "evaluate", fail_evaluate)
    result = engine.run_epoch(1)

    assert result["residuals"]


def test_native_pt_wrapper_epoch_can_validate_with_python() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=1)
    circuit = build_xts_block_collection(specs, encoding="lowered")
    assignment = assignment_from_fixture_plaintext(fixture, specs)
    circuit.derive_wire_values(assignment)
    engine = NativeParallelTemperingEngine(
        circuit,
        assignment,
        config=ParallelTemperingConfig(replicas=1, seed=19, validate_python_epoch=True),
    )

    result = engine.run_epoch(1)
    native = engine.native_engine.score_assignment(engine.assignment)

    assert tuple(native["residuals"]) == tuple(result.residuals)


def test_native_pt_derives_wires_like_python_for_random_inputs() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=1)
    circuit = build_xts_block_collection(specs, encoding="lowered")
    base = assignment_from_fixture_plaintext(fixture, specs)
    circuit.derive_wire_values(base)
    engine = NativeParallelTemperingEngine(
        circuit,
        base,
        config=ParallelTemperingConfig(replicas=1, seed=21),
    )
    consistency_indices = [
        index for index, constraint in enumerate(circuit.constraints)
        if constraint.kind == ConstraintKind.DEFINE8
    ]

    for case in range(4):
        assignment = Assignment.empty(AES_BLOCK_BYTES, len(circuit.wire_offsets))
        assignment.plaintext[:] = bytes((case * 29 + index) % 95 + 0x20 for index in range(AES_BLOCK_BYTES))
        assignment.key1[:] = bytes((case * 31 + index) % 256 for index in range(32))
        assignment.key2[:] = bytes((case * 37 + index) % 256 for index in range(32))
        assert assignment.wires is not None
        assignment.wires[:] = bytes((case * 41 + index) % 256 for index in range(len(assignment.wires)))

        native_derived = engine.native_engine.derive_wires(assignment)
        python_derived = Assignment(
            bytearray(assignment.plaintext),
            bytearray(assignment.key1),
            bytearray(assignment.key2),
            bytearray(assignment.wires),
        )
        circuit.derive_wire_values(python_derived)

        assert native_derived.wires == python_derived.wires
        native_score = engine.native_engine.score_assignment(native_derived)
        assert sum(native_score["residuals"][index] for index in consistency_indices) == 0


def test_native_pt_graph_model_metrics_match_circuit_shape() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=1)
    circuit = build_xts_block_collection(specs, encoding="lowered")
    assignment = assignment_from_fixture_plaintext(fixture, specs)
    circuit.derive_wire_values(assignment)
    engine = NativeParallelTemperingEngine(
        circuit,
        assignment,
        config=ParallelTemperingConfig(replicas=2, seed=61),
    )

    graph_counts = engine.metrics()["native_summary"]["graph_counts"]

    assert graph_counts["values"] == len(circuit.values)
    assert graph_counts["ops"] == len(circuit.ops)
    assert graph_counts["constraints"] == len(circuit.constraints)
    assert graph_counts["wire_bytes"] == len(circuit.wire_offsets)
    assert graph_counts["replicas"] == 2


def test_native_pt_proposal_kernel_counts_one_attempt_per_replica_step() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=1)
    circuit = build_xts_block_collection(specs, encoding="lowered")
    assignment = assignment_from_fixture_plaintext(fixture, specs)
    circuit.derive_wire_values(assignment)
    engine = NativeParallelTemperingEngine(
        circuit,
        assignment,
        config=ParallelTemperingConfig(replicas=3, seed=62, repair_move_prob=0.0),
    )

    engine.run_epoch(4)
    attempts = engine.metrics()["proposal_attempts"]

    assert sum(attempts.values()) == 12
    assert set(attempts) == {"wire_flip", "plaintext_ascii", "key_bit_flip", "key_word_swap"}


def test_native_pt_key_word_swap_proposal_is_counted() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=1)
    circuit = build_xts_block_collection(specs, encoding="lowered")
    assignment = assignment_from_fixture_plaintext(fixture, specs)
    circuit.derive_wire_values(assignment)
    engine = NativeParallelTemperingEngine(
        circuit,
        assignment,
        config=ParallelTemperingConfig(replicas=1, seed=64, repair_move_prob=0.0),
    )

    engine.run_epoch(200)
    attempts = engine.metrics()["proposal_attempts"]

    assert attempts["key_word_swap"] > 0


def test_native_pt_profile_diagnostics_are_collected_by_label() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=1)
    circuit = build_xts_block_collection(specs, encoding="lowered")
    assignment = assignment_from_fixture_plaintext(fixture, specs)
    circuit.derive_wire_values(assignment)
    engine = NativeParallelTemperingEngine(
        circuit,
        assignment,
        config=ParallelTemperingConfig(replicas=2, seed=63, profile=True),
    )

    engine.run_epoch(2)
    metrics = engine.metrics()

    assert metrics["profile_counts"]["run_epoch_total"] >= 1
    assert metrics["profile_seconds"]["run_epoch_total"] >= 0.0
    assert metrics["profile_counts"].keys() == metrics["profile_seconds"].keys()


def test_native_pt_drains_known_feasible_shared_key_sample() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=21)
    circuit = build_xts_block_collection(specs, encoding="lowered")
    assignment = assignment_from_fixture_plaintext(fixture, specs)
    circuit.derive_wire_values(assignment)
    engine = NativeParallelTemperingEngine(
        circuit,
        assignment,
        config=ParallelTemperingConfig(replicas=1, sweeps_per_epoch=0, seed=5),
    )

    samples = engine.drain_feasible()

    assert len(samples) == 1
    assert samples[0].key1 == assignment.key1
    assert samples[0].key2 == assignment.key2


def test_native_pt_accepted_state_marginals_populate_without_feasible_samples() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=1)
    circuit = build_xts_block_collection(specs, encoding="lowered")
    assignment = Assignment.empty(AES_BLOCK_BYTES, len(circuit.wire_offsets))
    assignment.plaintext[:] = b"A" * AES_BLOCK_BYTES
    assignment.key1[:] = bytes(range(32))
    assignment.key2[:] = bytes(range(32, 64))
    assert assignment.wires is not None
    assignment.wires[:] = bytes((index * 17) % 256 for index in range(len(assignment.wires)))
    engine = NativeParallelTemperingEngine(
        circuit,
        assignment,
        config=ParallelTemperingConfig(replicas=1, seed=41),
    )

    assert engine.drain_feasible() == []
    engine.run_epoch(2)
    metrics = engine.metrics()

    assert metrics["key_visit_count"] >= 1
    assert "key_distinct_count" in metrics
    assert len(metrics["key_ones"]) == 512
    assert metrics["key_marginal_max_deviation"] is not None
    assert metrics["key_information_bits"] >= 0.0
    assert metrics["key_information_bits_raw"] >= metrics["key_information_bits"]
    assert metrics["key_information_null_bits"] >= 0.0
    assert "full-key Hamming weight" in metrics["marginal_diagnostic_note"]


def test_native_pt_frozen_cold_chain_marginals_are_untrusted() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=1)
    circuit = build_xts_block_collection(specs, encoding="lowered")
    assignment = assignment_from_fixture_plaintext(fixture, specs)
    circuit.derive_wire_values(assignment)
    engine = NativeParallelTemperingEngine(
        circuit,
        assignment,
        config=ParallelTemperingConfig(replicas=1, seed=42),
    )

    for _ in range(60):
        engine.native_engine.run_epoch(0)
    metrics = engine.metrics()

    assert metrics["key_visit_count"] == 0
    assert metrics["key_distinct_count"] == 0
    assert not metrics["marginal_trusted"]
    assert metrics["marginal_ess"] == 0.0
    assert metrics["marginal_rhat"] == 0.0
    assert metrics["key_information_bits"] == 0.0


def test_native_pt_low_distinct_key_window_is_untrusted() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=1)
    circuit = build_xts_block_collection(specs, encoding="lowered")
    assignment = assignment_from_fixture_plaintext(fixture, specs)
    circuit.derive_wire_values(assignment)
    engine = NativeParallelTemperingEngine(
        circuit,
        assignment,
        config=ParallelTemperingConfig(replicas=1, seed=45, marginal_min_distinct_keys=25),
    )

    engine.run_epoch(60)
    metrics = engine.metrics()

    if metrics["key_distinct_count"] < 25:
        assert not metrics["marginal_trusted"]
        assert metrics["key_information_bits"] == 0.0


def test_native_pt_energy_stats_survive_dual_update_until_metrics() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=1)
    circuit = build_xts_block_collection(specs, encoding="lowered")
    assignment = assignment_from_fixture_plaintext(fixture, specs)
    circuit.derive_wire_values(assignment)
    engine = NativeParallelTemperingEngine(
        circuit,
        assignment,
        config=ParallelTemperingConfig(replicas=3, seed=44),
    )

    engine.run_epoch(3)
    metrics = engine.metrics()

    assert metrics["energy_sample_counts"] == [3, 3, 3]
    assert any(value != 0.0 for value in metrics["energy_mean_by_rung"])
    assert metrics["relative_log_z_estimate"] is not None
    assert metrics["log_z_estimate"] is None
    assert metrics["log_feasible_count_estimate"] is None
    assert metrics["log_z_state_bits"] == (AES_BLOCK_BYTES + 64) * 8


def test_native_pt_epoch_and_swap_stats_are_deterministic() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=1)
    circuit = build_xts_block_collection(specs, encoding="lowered")
    assignment = assignment_from_fixture_plaintext(fixture, specs)
    circuit.derive_wire_values(assignment)
    config = ParallelTemperingConfig(replicas=2, sweeps_per_epoch=1, seed=11)

    first = NativeParallelTemperingEngine(circuit, assignment, config=config)
    first.run_epoch(1)
    second = NativeParallelTemperingEngine(circuit, assignment, config=config)
    second.run_epoch(1)

    assert first.metrics()["swap_attempts"] == second.metrics()["swap_attempts"]
    assert first.metrics()["swap_accepts"] == second.metrics()["swap_accepts"]
    assert first.metrics()["sweeps"] == 1


def test_native_pt_feedback_ladder_exports_flow_diagnostics() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=1)
    circuit = build_xts_block_collection(specs, encoding="lowered")
    assignment = assignment_from_fixture_plaintext(fixture, specs)
    circuit.derive_wire_values(assignment)
    engine = NativeParallelTemperingEngine(
        circuit,
        assignment,
        config=ParallelTemperingConfig(
            replicas=4,
            seed=43,
            ladder_mode="feedback",
            ladder_burn_in_epochs=0,
            ladder_adapt_interval_epochs=1,
            ladder_min_round_trips=0,
        ),
    )

    engine.run_epoch(2)
    metrics = engine.metrics()

    assert len(metrics["temperatures"]) == 4
    assert metrics["temperatures"] == sorted(metrics["temperatures"])
    assert len(metrics["replica_up_counts"]) == 4
    assert "total_round_trips" in metrics


def test_native_pt_protocol_setters_reach_cpp_engine() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=1)
    circuit = build_xts_block_collection(specs, encoding="lowered")
    assignment = assignment_from_fixture_plaintext(fixture, specs)
    circuit.derive_wire_values(assignment)
    engine = NativeParallelTemperingEngine(
        circuit,
        assignment,
        config=ParallelTemperingConfig(replicas=2, seed=23),
    )

    engine.configure_temperatures([1.0, 2.0])
    engine.set_temperature(1.5)
    engine.set_constraint_classes(classify_constraints(circuit))
    engine.set_multipliers([0.5] * len(circuit.constraints))
    engine.run_epoch(1)

    assert engine.metrics()["sweeps"] == 1


def test_native_pt_dual_update_does_not_raise_goal_multipliers() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=1)
    circuit = build_xts_block_collection(specs, encoding="lowered")
    assignment = assignment_from_fixture_plaintext(fixture, specs)
    circuit.derive_wire_values(assignment)
    assignment.wires[0] ^= 0xFF
    engine = NativeParallelTemperingEngine(
        circuit,
        assignment,
        config=ParallelTemperingConfig(replicas=2, dual_eta=1.0, goal_weight=0.8, seed=37),
    )
    before = engine.metrics()["multiplier_mean_by_class"]

    engine.run_epoch(3)
    after = engine.metrics()["multiplier_mean_by_class"]

    assert after["goal"] == before["goal"]


def test_native_pt_scheduled_dual_exports_penalty_diagnostics() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=1)
    circuit = build_xts_block_collection(specs, encoding="lowered")
    assignment = assignment_from_fixture_plaintext(fixture, specs)
    circuit.derive_wire_values(assignment)
    assert assignment.wires is not None
    assignment.wires[0] ^= 0xFF
    engine = NativeParallelTemperingEngine(
        circuit,
        assignment,
        config=ParallelTemperingConfig(
            replicas=2,
            seed=47,
            dual_mode="scheduled",
            scheduled_rho_initial=0.2,
        ),
    )

    engine.run_epoch(2)
    metrics = engine.metrics()

    assert metrics["rho_by_class"]["consistency"] >= 0.2
    assert sum(metrics["lambda_update_counts_by_class"].values()) >= 1
    assert "infeasibility_suspected" in metrics


def test_native_pt_algebra_bp_and_alternative_diagnostics_are_exported() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=1)
    circuit = build_xts_block_collection(specs, encoding="lowered")
    assignment = assignment_from_fixture_plaintext(fixture, specs)
    circuit.derive_wire_values(assignment)
    engine = NativeParallelTemperingEngine(
        circuit,
        assignment,
        config=ParallelTemperingConfig(
            replicas=2,
            seed=53,
            algebra_diagnostics=True,
            bp_diagnostics=True,
            alternative_diagnostics=True,
        ),
    )

    engine.run_epoch(1)
    metrics = engine.metrics()

    assert metrics["algebra_summary"]["variable_bits"] > 0
    assert metrics["algebra_summary"]["factor_count"] > 0
    assert metrics["algebra_summary"]["rank_estimate"] is None
    assert not metrics["algebra_summary"]["gf2_elimination_implemented"]
    assert not metrics["bp_available"]
    assert not metrics["bp_converged"]
    assert metrics["bp_key_marginals"] == []
    assert not metrics["alternative_available"]
    assert metrics["alternative_log_z_estimates"] == []
    assert not metrics["langevin_available"]
    assert metrics["langevin_seed_score"] is None


def test_native_pt_epoch_result_matches_python_after_delta_path() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=1)
    circuit = build_xts_block_collection(specs, encoding="lowered")
    assignment = assignment_from_fixture_plaintext(fixture, specs)
    circuit.derive_wire_values(assignment)
    engine = NativeParallelTemperingEngine(
        circuit,
        assignment,
        config=ParallelTemperingConfig(replicas=1, seed=29),
    )

    engine.run_epoch(5)
    native = engine.native_engine.score_assignment(engine.assignment)
    python = circuit.evaluate(engine.assignment)

    assert tuple(native["residuals"]) == tuple(python.residuals)


def test_native_pt_parallel_threads_keep_residuals_consistent() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=1)
    circuit = build_xts_block_collection(specs, encoding="lowered")
    assignment = assignment_from_fixture_plaintext(fixture, specs)
    circuit.derive_wire_values(assignment)
    engine = NativeParallelTemperingEngine(
        circuit,
        assignment,
        config=ParallelTemperingConfig(replicas=4, threads=2, seed=31),
    )

    engine.run_epoch(3)
    native = engine.native_engine.score_assignment(engine.assignment)
    python = circuit.evaluate(engine.assignment)

    assert tuple(native["residuals"]) == tuple(python.residuals)
    assert engine.metrics()["sweeps"] == 3


def test_ascii_constraints() -> None:
    assert is_text_ascii(ord("A"))
    assert is_text_ascii(0x0A)
    assert not is_text_ascii(0x00)
    # A bytes/sequence argument checks that every byte is text ASCII.
    assert is_text_ascii(b"Lorem ipsum dolo")
    assert is_text_ascii(bytearray(b"tab\tnewline\n"))
    assert not is_text_ascii(b"with null\x00byte")
    assert is_text_ascii(b"")


def test_sampler_does_not_seed_key_by_default() -> None:
    sampler = XtsSampler(
        SamplerConfig(
            metadata_path=FIXTURE_METADATA,
            block_count=1,
            encoding="lowered",
            seed=7,
        )
    )
    assert sampler.config.fixed_key is False
    assignment = sampler.random_assignment()
    assert bytes(assignment.key1) != bytes(sampler.fixture.key1)
    assert bytes(assignment.key2) != bytes(sampler.fixture.key2)

    seeded = XtsSampler(
        SamplerConfig(
            metadata_path=FIXTURE_METADATA,
            block_count=1,
            encoding="lowered",
            seed=7,
            fixed_key=True,
        )
    )
    seeded_assignment = seeded.random_assignment()
    assert bytes(seeded_assignment.key1) == bytes(seeded.fixture.key1)
    assert bytes(seeded_assignment.key2) == bytes(seeded.fixture.key2)


def test_xts_key_split() -> None:
    key = bytes(range(64))
    key1, key2 = split_xts_key(key)
    assert key1 == bytes(range(32))
    assert key2 == bytes(range(32, 64))


def test_cli_check_fixture_smoke() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.decoder.legacy.cli",
            str(FIXTURE_METADATA),
            "--block-count",
            "1",
            "--check-fixture",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "satisfied=True" in result.stdout


def test_cli_lowered_check_fixture_smoke() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.decoder.legacy.cli",
            str(FIXTURE_METADATA),
            "--block-count",
            "1",
            "--encoding",
            "lowered",
            "--check-fixture",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "satisfied=True" in result.stdout


def test_cli_lowered_sampler_smoke() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.decoder.legacy.cli",
            str(FIXTURE_METADATA),
            "--block-count",
            "1",
            "--encoding",
            "lowered",
            "--relaxation",
            "continuous",
            "--iterations",
            "1",
            "--seed",
            "1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "violations=" in result.stdout


def test_sampler_relaxation_flag_changes_residual_surface() -> None:
    discrete_sampler = XtsSampler(
        SamplerConfig(
            metadata_path=FIXTURE_METADATA,
            block_count=1,
            encoding="lowered",
            relaxation="discrete",
        )
    )
    continuous_sampler = XtsSampler(
        SamplerConfig(
            metadata_path=FIXTURE_METADATA,
            block_count=1,
            encoding="lowered",
            relaxation="continuous",
        )
    )
    assignment = discrete_sampler.known_fixture_assignment()
    assert assignment.wires is not None
    assignment.wires[0] ^= 0xFF
    discrete = discrete_sampler.score(assignment).residuals
    continuous = continuous_sampler.score(assignment).residuals
    assert discrete != continuous


def test_shared_key_sampler_cli_writes_jsonl(tmp_path: Path) -> None:
    samples_path = tmp_path / "samples.jsonl"
    metrics_path = tmp_path / "metrics.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.decoder.legacy.sample_blocks",
            str(FIXTURE_METADATA),
            "--block-count",
            "21",
            "--samples-target",
            "1",
            "--replicas",
            "1",
            "--epochs",
            "1",
            "--sweeps-per-epoch",
            "0",
            "--warm-start-fixture",
            "--output",
            str(samples_path),
            "--metrics",
            str(metrics_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    sample = json.loads(samples_path.read_text(encoding="utf-8").splitlines()[0])
    metric = json.loads(metrics_path.read_text(encoding="utf-8").splitlines()[0])
    assert sample["block_count"] == 21
    assert len(sample["plaintext_blocks_hex"]) == 21
    assert metric["samples_written"] == 1


def test_testing_setup_runners_do_not_warm_start_by_default() -> None:
    assert not build_shared_key_runner_parser().parse_args([]).warm_start_fixture
    assert build_shared_key_runner_parser().parse_args(["--warm-start-fixture"]).warm_start_fixture
    assert not build_fusion_runner_parser().parse_args([]).warm_start_fixture
    assert build_fusion_runner_parser().parse_args(["--warm-start-fixture"]).warm_start_fixture


def test_window_fusion_native_engine_cli_selects_implementation() -> None:
    parser = build_fusion_runner_parser()

    assert parser.parse_args([]).native_engine == "native-pt"
    assert parser.parse_args(["--native-engine", "native-continuous-relaxed"]).native_engine == "native-continuous-relaxed"
    assert parser.parse_args(["--native-engine", "native-algebraic-relaxed"]).native_engine == "native-algebraic-relaxed"
    assert _native_engine_class("native-pt") is NativeParallelTemperingEngine
    assert _native_engine_class("native-continuous-relaxed") is NativeContinuousRelaxedEngine
    assert _native_engine_class("native-algebraic-relaxed") is NativeAlgebraicRelaxedEngine


def test_window_fusion_benchmark_presets_apply_defaults() -> None:
    micro = parse_fusion_runner_args(["--benchmark-preset", "micro"])
    assert micro.max_windows == 1
    assert micro.epochs == 1
    assert micro.sweeps_per_epoch == 1
    assert micro.profile_sampler

    override = parse_fusion_runner_args(["--benchmark-preset", "micro", "--epochs", "3", "--no-profile-sampler"])
    assert override.epochs == 3
    assert not override.profile_sampler


def test_window_fusion_positive_control_preset_applies_solve_defaults() -> None:
    args = parse_fusion_runner_args(["--positive-control-solve-mode", "--aes-rounds", "2"])
    assert args.window_size == 1
    assert args.samples_per_window == 1
    assert args.goal_ramp == 5.0
    assert args.mu == 1.0
    assert args.dual_mode == "scheduled"
    assert args.t_min == 1.0
    assert args.t_max == 50.0

    override = parse_fusion_runner_args(["--positive-control-solve-mode", "--window-size", "3", "--goal-ramp", "2.0"])
    assert override.window_size == 3
    assert override.goal_ramp == 2.0


def test_window_fusion_marginal_updates_match_bit_order() -> None:
    marginals = [BlockMarginal(samples=0, ones=[0] * 512) for _ in range(3)]
    key = bytes((index * 7 + 3) % 256 for index in range(64))

    _add_sample_to_block_marginals(marginals, 1, 2, key)

    expected_bits = _bytes_to_bits(key)
    assert marginals[0].samples == 0
    for marginal in marginals[1:]:
        assert marginal.samples == 1
        assert marginal.ones == expected_bits


def test_window_fusion_accepts_native_key_marginal_fallback() -> None:
    marginals = [BlockMarginal(samples=0, ones=[0] * 512) for _ in range(4)]
    key = bytes((index * 5 + 11) % 256 for index in range(64))
    bits = _bytes_to_bits(key)
    metrics = {
        "key_visit_count": 7,
        "key_ones": [bit * 7 for bit in bits],
    }

    visits = _add_native_key_marginals_to_blocks(marginals, 1, 2, metrics)

    assert visits == 7
    assert marginals[0].samples == 0
    assert marginals[3].samples == 0
    for marginal in marginals[1:3]:
        assert marginal.samples == 7
        assert marginal.ones == [bit * 7 for bit in bits]
