"""Correctness tests for the augmented-Lagrangian energy and dual update.

The environment's energy is the augmented Lagrangian

    Energy = sum_i [ lambda_i * r_i + 1/2 * rho_i * r_i^2 ]

with per-class Lagrange multipliers (lambda) and quadratic penalties (rho), the
consistency class additionally scaled by the coupling weight. These tests assert
the native SDK computes exactly that (cross-checked against a Python reference),
that it is monotonic in rho, that feasibility yields zero energy, and that the
curriculum's method-of-multipliers dual update advances multipliers correctly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.decoder.constraints import classify_constraints
from src.decoder.rl.curriculum import Curriculum, TaskSpec
from src.decoder.rl.harness import (
    TaskWeights,
    harness_available,
    probe_assignment,
    reference_energy,
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

requires_toolchain = pytest.mark.skipif(
    not harness_available(), reason="C++ toolchain to build the harness is unavailable"
)


def _corrupted_circuit_and_assignment(aes_rounds: int = 2):
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=1)
    model_specs = block_specs_for_model(fixture, specs, aes_rounds=aes_rounds)
    circuit = build_xts_block_collection(model_specs, encoding="lowered", aes_rounds=aes_rounds)
    assignment = assignment_from_fixture_plaintext(fixture, model_specs)
    circuit.derive_wire_values(assignment)
    # Corrupt so residuals are nonzero and spread across classes/rounds.
    assignment.wires[0] ^= 0xFF
    assignment.key1[0] ^= 0x0F
    return circuit, assignment


# ---------------------------------------------------------------------------
# Pure-Python: the reference energy function itself
# ---------------------------------------------------------------------------


def test_reference_energy_linear_and_quadratic() -> None:
    # Two residuals: one ASCII (r=2), one goal (r=3); consistency none.
    residuals = [2.0, 3.0]
    classes = [1, 3]  # ASCII, GOAL
    w = TaskWeights(ascii_weight=1.0, goal_weight=2.0, ascii_rho=0.5, goal_rho=1.0)
    # ascii: 1*2 + 0.5*0.5*4 = 2 + 1 = 3 ; goal: 2*3 + 0.5*1*9 = 6 + 4.5 = 10.5
    assert reference_energy(residuals, classes, w) == pytest.approx(13.5)


def test_reference_energy_consistency_scaled_by_coupling() -> None:
    residuals = [4.0]
    classes = [2]  # CONSISTENCY
    w = TaskWeights(consistency_weight=1.0, consistency_rho=0.5, coupling_weight=3.0)
    # lambda_eff = 1*3 = 3 ; rho_eff = 0.5*3 = 1.5
    # energy = 3*4 + 0.5*1.5*16 = 12 + 12 = 24
    assert reference_energy(residuals, classes, w) == pytest.approx(24.0)


def test_reference_energy_zero_residual_is_zero() -> None:
    assert reference_energy([0.0, 0.0], [1, 2], TaskWeights(ascii_rho=9, consistency_rho=9)) == 0.0


# ---------------------------------------------------------------------------
# Native SDK cross-check (needs the compiler)
# ---------------------------------------------------------------------------


@requires_toolchain
def test_sdk_energy_matches_reference_linear_and_quadratic() -> None:
    circuit, a = _corrupted_circuit_and_assignment()
    classes = [int(c) for c in classify_constraints(circuit)]
    for w in [
        TaskWeights(),  # pure linear
        TaskWeights(
            ascii_weight=1.0, consistency_weight=2.0, goal_weight=1.5,
            ascii_rho=0.3, consistency_rho=0.7, goal_rho=0.5, coupling_weight=2.0,
        ),
    ]:
        probe = probe_assignment(circuit, 2, bytes(a.plaintext), bytes(a.key1), bytes(a.key2), weights=w)
        assert probe.energy == pytest.approx(reference_energy(probe.residuals, classes, w))


@requires_toolchain
def test_sdk_energy_monotonic_in_rho() -> None:
    circuit, a = _corrupted_circuit_and_assignment()
    lin = probe_assignment(
        circuit, 2, bytes(a.plaintext), bytes(a.key1), bytes(a.key2),
        weights=TaskWeights(),
    )
    quad = probe_assignment(
        circuit, 2, bytes(a.plaintext), bytes(a.key1), bytes(a.key2),
        weights=TaskWeights(ascii_rho=1.0, consistency_rho=1.0, goal_rho=1.0),
    )
    # With nonzero residuals, adding a positive quadratic penalty strictly
    # increases the energy.
    assert lin.hamming > 0
    assert quad.energy > lin.energy


@requires_toolchain
def test_feasible_assignment_has_zero_energy_regardless_of_rho() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=1)
    model_specs = block_specs_for_model(fixture, specs, aes_rounds=2)
    circuit = build_xts_block_collection(model_specs, encoding="lowered", aes_rounds=2)
    a = assignment_from_fixture_plaintext(fixture, model_specs)
    probe = probe_assignment(
        circuit, 2, bytes(a.plaintext), bytes(a.key1), bytes(a.key2),
        weights=TaskWeights(consistency_rho=5.0, goal_rho=5.0, coupling_weight=3.0),
    )
    assert probe.hamming == 0.0
    assert probe.energy == 0.0


# ---------------------------------------------------------------------------
# Method-of-multipliers dual update
# ---------------------------------------------------------------------------


def test_dual_update_raises_multipliers_by_rho_times_residual() -> None:
    spec = TaskSpec(
        metadata_path=FIXTURE_METADATA,
        ascii_weight=1.0, consistency_weight=1.0, goal_weight=1.0,
        ascii_rho=0.5, consistency_rho=2.0, goal_rho=1.0,
        coupling_weight=3.0,
    )
    # residuals [ascii, consistency, goal]
    updated = spec.dual_update([4.0, 5.0, 6.0])
    # ascii:       1 + 0.5*4 = 3
    # consistency: 1 + 2.0*coupling(3)*5 = 1 + 30 = 31
    # goal:        1 + 1.0*6 = 7
    assert updated.ascii_weight == pytest.approx(3.0)
    assert updated.consistency_weight == pytest.approx(31.0)
    assert updated.goal_weight == pytest.approx(7.0)
    # rho/coupling are unchanged by the dual update itself.
    assert updated.consistency_rho == pytest.approx(2.0)
    assert updated.coupling_weight == pytest.approx(3.0)


def test_dual_update_noop_when_rho_zero_or_no_residual() -> None:
    spec = TaskSpec(metadata_path=FIXTURE_METADATA)  # all rho default 0
    updated = spec.dual_update([9.0, 9.0, 9.0])
    assert (updated.ascii_weight, updated.consistency_weight, updated.goal_weight) == (
        spec.ascii_weight, spec.consistency_weight, spec.goal_weight
    )
    spec2 = TaskSpec(metadata_path=FIXTURE_METADATA, consistency_rho=2.0)
    updated2 = spec2.dual_update([0.0, 0.0, 0.0])
    assert updated2.consistency_weight == pytest.approx(spec2.consistency_weight)


def test_curriculum_ramps_penalty_and_advances_multipliers() -> None:
    cur = Curriculum(metadata_path=FIXTURE_METADATA, max_rounds=3)
    # Penalty ramp rises across stages.
    rhos = [cur.task_for_stage(s).consistency_rho for s in range(cur.num_stages())]
    assert rhos == sorted(rhos)
    assert rhos[-1] > rhos[0]
    # advance() carries a method-of-multipliers-updated multiplier into the next
    # stage's schedule-set rho/coupling.
    stage0 = cur.task_for_stage(0)
    nxt = cur.advance(0, [1.0, 2.0, 3.0])
    expected = stage0.dual_update([1.0, 2.0, 3.0])
    assert nxt.consistency_weight == pytest.approx(expected.consistency_weight)
    # And it picks up stage 1's penalty schedule.
    assert nxt.consistency_rho == pytest.approx(cur.task_for_stage(1).consistency_rho)
