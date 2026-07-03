"""Harness + reward correctness tests for the solver-authoring RL environment.

These assert the *environment* is correct -- the compile-and-run harness, the
reward plumbing, per-round aggregation, determinism and failure shaping -- and
deliberately make no claim about how well any example solver performs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.decoder.circuit import Assignment
from src.decoder.rl.curriculum import Curriculum, TaskSpec
from src.decoder.rl.harness import (
    HarnessOptions,
    constraint_round_indices,
    harness_available,
    probe_assignment,
)
from src.decoder.rl.reward import Rubric
from src.decoder.rl.solver_env import SolverAuthoringEnv, extract_source
from src.decoder.xts_model import (
    assignment_from_fixture_plaintext,
    block_specs_for_fixture,
    block_specs_for_model,
    build_xts_block_collection,
    load_fixture,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_METADATA = ROOT / "src" / "testing_setup" / "lorem_ipsum_aes_xts.json"
EXAMPLES = ROOT / "examples" / "solvers"

requires_toolchain = pytest.mark.skipif(
    not harness_available(), reason="C++ toolchain to build the harness is unavailable"
)

# Small budget keeps CI fast; correctness does not depend on search depth.
FAST = HarnessOptions(run_timeout=15.0, compile_timeout=60.0, epochs=1, sweeps_per_epoch=2000)


def _spec(rounds_ladder=(1, 2), block_count=1, seed=7) -> TaskSpec:
    return TaskSpec(
        metadata_path=FIXTURE_METADATA,
        block_count=block_count,
        rounds_ladder=rounds_ladder,
        seed=seed,
    )


def _example(name: str) -> str:
    return (EXAMPLES / name / "solver.cpp").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Pure-Python pieces (no toolchain needed)
# ---------------------------------------------------------------------------


def test_round_tagging_covers_only_data_path_rounds() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=1)
    model_specs = block_specs_for_model(fixture, specs, aes_rounds=3)
    circuit = build_xts_block_collection(model_specs, encoding="lowered", aes_rounds=3)

    rounds = constraint_round_indices(circuit, 3)
    assert len(rounds) == len(circuit.constraints)
    # Data-path rounds are tagged 1..3; everything else is the boundary bucket 0.
    assert set(rounds) <= {0, 1, 2, 3}
    assert 1 in rounds and 2 in rounds and 3 in rounds
    # No round is tagged beyond the configured round count.
    assert max(rounds) == 3


def test_extract_source_prefers_solver_block() -> None:
    completion = (
        "Here is my solver:\n```cpp\n#include \"solver_sdk.hpp\"\n"
        "extern \"C\" ISolver* create_solver(const SolverContext& c){return nullptr;}\n```\n"
    )
    source = extract_source(completion)
    assert "create_solver" in source
    assert "```" not in source


def test_rubric_orders_failures_below_running_solver() -> None:
    from src.decoder.rl.harness import HarnessRun, TaskResult

    rubric = Rubric()

    def run(**kwargs) -> HarnessRun:
        base = dict(aes_rounds=1, ran=True, feasible=False, timed_out=False, crashed=False,
                    energy=10.0, best_hamming=10.0, round_staircase=0.5, seconds=0.1,
                    per_round=[], error="", constraint_count=100)
        base.update(kwargs)
        return HarnessRun(True, 1.0, "", [TaskResult(**base)])

    spec = _spec(rounds_ladder=(1,))
    good = rubric.score(run(), spec).reward
    crashed = rubric.score(run(ran=False, crashed=True), spec).reward
    no_compile = rubric.score(HarnessRun(False, 0.0, "err", []), spec).reward
    assert no_compile < crashed < good


# ---------------------------------------------------------------------------
# Harness-backed tests (need a compiler)
# ---------------------------------------------------------------------------


@requires_toolchain
@pytest.mark.parametrize("name", ["baseline", "random_metropolis"])
def test_example_solvers_conform_to_interface(name: str) -> None:
    env = SolverAuthoringEnv(options=FAST)
    breakdown = env.score_source(_example(name), _spec())
    assert breakdown.compile_ok, breakdown.compile_stderr
    for task in breakdown.per_task:
        assert task["ran"] is not False
        assert not task["crashed"], task["error"]
        assert not task["timed_out"], task["error"]


@requires_toolchain
def test_energy_and_per_round_match_python_reference() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=1)
    model_specs = block_specs_for_model(fixture, specs, aes_rounds=2)
    circuit = build_xts_block_collection(model_specs, encoding="lowered", aes_rounds=2)

    # A corrupted-but-derived assignment so residuals are nonzero and spread out.
    assignment = assignment_from_fixture_plaintext(fixture, model_specs)
    circuit.derive_wire_values(assignment)
    assignment.wires[0] ^= 0xFF
    assignment.key1[0] ^= 0x0F

    probe = probe_assignment(
        circuit, 2, bytes(assignment.plaintext), bytes(assignment.key1), bytes(assignment.key2)
    )
    reference = circuit.evaluate(assignment)

    # The SDK derives wires from the same plaintext/keys, so compare against a
    # Python evaluation that also re-derives from those inputs.
    py_assignment = Assignment(
        bytearray(assignment.plaintext), bytearray(assignment.key1), bytearray(assignment.key2)
    )
    circuit.derive_wire_values(py_assignment)
    py_reference = circuit.evaluate(py_assignment)

    assert probe.hamming == py_reference.hamming_score
    assert probe.residuals == [float(r) for r in py_reference.residuals]

    # Per-round buckets recomputed in Python from the same tags must match.
    rounds = constraint_round_indices(circuit, 2)
    expected = [0.0] * (2 + 1)
    for residual, round_index in zip(py_reference.residuals, rounds):
        expected[round_index] += residual
    assert probe.per_round == expected
    assert reference.constraints == len(circuit.constraints)

    # The SDK's augmented-Lagrangian energy must match the Python reference for
    # the default (linear) weights on the same residuals/classes.
    from src.decoder.constraints import classify_constraints
    from src.decoder.rl.harness import TaskWeights, reference_energy

    classes = [int(c) for c in classify_constraints(circuit)]
    assert probe.energy == pytest.approx(
        reference_energy(probe.residuals, classes, TaskWeights())
    )


@requires_toolchain
def test_true_key_assignment_is_feasible_and_zero_energy() -> None:
    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=1)
    model_specs = block_specs_for_model(fixture, specs, aes_rounds=2)
    circuit = build_xts_block_collection(model_specs, encoding="lowered", aes_rounds=2)
    assignment = assignment_from_fixture_plaintext(fixture, model_specs)

    probe = probe_assignment(
        circuit, 2, bytes(assignment.plaintext), bytes(assignment.key1), bytes(assignment.key2)
    )
    assert probe.hamming == 0.0
    assert probe.energy == 0.0
    assert probe.per_round == [0.0, 0.0, 0.0]


@requires_toolchain
def test_repair_helper_reaches_feasibility_from_true_key() -> None:
    # random_metropolis uses RepairPlaintext; with enough sweeps at 1 round it
    # should harvest a feasible assignment, exercising the SDK repair path.
    env = SolverAuthoringEnv(options=HarnessOptions(run_timeout=20.0, epochs=3, sweeps_per_epoch=8000))
    breakdown = env.score_source(_example("random_metropolis"), _spec(rounds_ladder=(1,)))
    assert breakdown.compile_ok


@requires_toolchain
def test_reward_is_deterministic() -> None:
    env = SolverAuthoringEnv(options=FAST)
    spec = _spec(rounds_ladder=(1, 2))
    first = env.score_source(_example("random_metropolis"), spec)
    second = env.score_source(_example("random_metropolis"), spec)
    assert first.reward == second.reward
    assert [t["energy"] for t in first.per_task] == [t["energy"] for t in second.per_task]
    assert [t["round_staircase"] for t in first.per_task] == [t["round_staircase"] for t in second.per_task]


@requires_toolchain
def test_non_compiling_source_hits_compile_penalty() -> None:
    env = SolverAuthoringEnv(options=FAST)
    breakdown = env.score_source("this is not valid c++;", _spec())
    assert not breakdown.compile_ok
    assert breakdown.reward == Rubric().compile_penalty
    assert breakdown.compile_stderr


@requires_toolchain
def test_timeout_is_shaped_not_fatal() -> None:
    hang = (
        '#include "solver_sdk.hpp"\n'
        "using namespace aes_xts_decoder::sdk;\n"
        "namespace { class Hang : public ISolver {\n"
        " public:\n"
        "  explicit Hang(const SolverContext&) {}\n"
        "  ScoreData RunEpoch(std::size_t) override { for(;;){} }\n"
        "  AssignmentState CurrentAssignment() const override { return {}; }\n"
        "  std::vector<AssignmentState> DrainFeasible(std::size_t) override { return {}; }\n"
        "}; }\n"
        'extern "C" ISolver* create_solver(const SolverContext& c){ return new Hang(c); }\n'
    )
    env = SolverAuthoringEnv(options=HarnessOptions(run_timeout=2.0, epochs=1, sweeps_per_epoch=1))
    breakdown = env.score_source(hang, _spec(rounds_ladder=(1,)))
    assert breakdown.compile_ok
    assert breakdown.per_task[0]["timed_out"]
    assert breakdown.reward == Rubric().timeout_penalty


@requires_toolchain
def test_score_batch_runs_curriculum() -> None:
    curriculum = Curriculum(FIXTURE_METADATA, block_count=1, max_rounds=2)
    env = SolverAuthoringEnv.from_curriculum(curriculum, options=FAST)
    prompts = env.prompts()
    assert prompts and all("ISolver" in p for p in prompts)
    source = _example("baseline")
    results = env.score_batch([source] * len(env.task_specs), env.task_specs)
    assert len(results) == len(env.task_specs)
    for result in results:
        # RolloutResult is the structured object every adapter maps from.
        info = result.to_dict()
        assert "components" in info
        assert "source_diagnostics" in info
        assert result.source_diagnostics["has_create_solver"]
        assert result.failure_kind in {"ok", "timeout", "crash", "compile_error"}


@requires_toolchain
def test_artifacts_are_written_when_enabled(tmp_path) -> None:
    env = SolverAuthoringEnv(options=FAST, artifacts_dir=tmp_path)
    result = env.score_source(_example("baseline"), _spec(rounds_ladder=(1,)))
    assert result.artifacts["solver_source"]
    assert Path(result.artifacts["solver_source"]).exists()
    rollout_json = Path(result.artifacts["rollout_json"])
    assert rollout_json.exists()
    import json as _json
    saved = _json.loads(rollout_json.read_text())
    assert saved["reward"] == result.reward
    assert saved["reward_version"]
