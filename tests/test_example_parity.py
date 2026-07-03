"""Functional-parity tests for the ported example solvers.

The three "reference strategy" solvers -- parallel_tempering, continuous_relaxed
and algebraic_relaxed -- are re-implementations of the legacy native engines
against the ISolver plugin interface. Exact bit-for-bit parity is neither
possible nor meaningful (different RNG streams, a different circuit
representation, no OpenMP), so "functional parity" here means *capability
parity*: each ported solver must

  * conform to the interface (compile, run, not crash or time out),
  * do real work (drive constraint violations well below the no-search floor),
  * preserve the legacy capability ordering (key-conditional Gibbs solvers beat
    the pure-Metropolis one), and
  * match the legacy engine's ability to collapse a reduced-round instance,
    cross-checked against the actual native engine on the same circuit.

All rewards are deterministic given a seed, which the determinism test asserts.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from src.decoder.circuit import Assignment
from src.decoder.rl.curriculum import TaskSpec
from src.decoder.rl.harness import HarnessOptions, harness_available
from src.decoder.rl.solver_env import SolverAuthoringEnv
from src.decoder.xts_model import (
    block_specs_for_fixture,
    block_specs_for_model,
    build_xts_block_collection,
    load_fixture,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_METADATA = ROOT / "src" / "testing_setup" / "lorem_ipsum_aes_xts.json"
EXAMPLES = ROOT / "examples" / "solvers"

PORTED = ["parallel_tempering", "continuous_relaxed", "algebraic_relaxed"]
KEY_GIBBS = ["continuous_relaxed", "algebraic_relaxed"]

requires_toolchain = pytest.mark.skipif(
    not harness_available(), reason="C++ toolchain to build the harness is unavailable"
)

# Budgets. Capability tests need a few thousand sweeps to collapse an instance;
# conformance/determinism can be cheap. Timeouts are generous so a slow CI box
# does not turn a real result into a shaped timeout.
CONFORM = HarnessOptions(run_timeout=30.0, compile_timeout=90.0, epochs=1, sweeps_per_epoch=1200)
STRONG = HarnessOptions(run_timeout=45.0, compile_timeout=90.0, epochs=2, sweeps_per_epoch=1500)
DETERMINISTIC = HarnessOptions(run_timeout=30.0, compile_timeout=90.0, epochs=1, sweeps_per_epoch=500)


def _spec(rounds_ladder=(1,), block_count=1, seed=7) -> TaskSpec:
    return TaskSpec(
        metadata_path=FIXTURE_METADATA,
        block_count=block_count,
        rounds_ladder=rounds_ladder,
        seed=seed,
    )


def _example(name: str) -> str:
    return (EXAMPLES / name / "solver.cpp").read_text(encoding="utf-8")


def _single_task(name: str, spec: TaskSpec, options: HarnessOptions) -> dict:
    env = SolverAuthoringEnv(options=options)
    breakdown = env.score_source(_example(name), spec)
    assert breakdown.compile_ok, breakdown.compile_stderr
    assert len(breakdown.per_task) == 1
    return breakdown.per_task[0]


# ---------------------------------------------------------------------------
# Interface conformance
# ---------------------------------------------------------------------------


@requires_toolchain
@pytest.mark.parametrize("name", PORTED)
def test_ported_solver_conforms_to_interface(name: str) -> None:
    env = SolverAuthoringEnv(options=CONFORM)
    breakdown = env.score_source(_example(name), _spec((1, 2)))
    assert breakdown.compile_ok, breakdown.compile_stderr
    assert len(breakdown.per_task) == 2
    for task in breakdown.per_task:
        assert task["ran"] is True
        assert not task["crashed"], task["error"]
        assert not task["timed_out"], task["error"]


# ---------------------------------------------------------------------------
# Beats the no-search floor (baseline reports the initial assignment's score)
# ---------------------------------------------------------------------------


@requires_toolchain
@pytest.mark.parametrize("name", PORTED)
def test_ported_solver_beats_baseline(name: str) -> None:
    spec = _spec((1,))
    initial = _single_task("baseline", spec, CONFORM)["best_hamming"]
    task = _single_task(name, spec, STRONG)
    assert initial > 0
    assert task["best_hamming"] < initial, (name, task["best_hamming"], initial)


# ---------------------------------------------------------------------------
# Capability ordering: key-Gibbs solvers collapse an instance; PT reduces it
# ---------------------------------------------------------------------------


@requires_toolchain
def test_parallel_tempering_substantially_reduces_violations() -> None:
    spec = _spec((1,))
    initial = _single_task("baseline", spec, CONFORM)["best_hamming"]
    pt = _single_task("parallel_tempering", spec, STRONG)["best_hamming"]
    assert pt <= 0.5 * initial, (pt, initial)


@requires_toolchain
@pytest.mark.parametrize("name", KEY_GIBBS)
def test_key_gibbs_solvers_nearly_solve_one_round(name: str) -> None:
    spec = _spec((1,))
    initial = _single_task("baseline", spec, CONFORM)["best_hamming"]
    task = _single_task(name, spec, STRONG)
    # The exact key-conditional Gibbs move is what lets the legacy continuous /
    # algebraic engines collapse a one-round instance; the port must too.
    assert task["best_hamming"] <= 0.15 * initial, (name, task["best_hamming"], initial)


# ---------------------------------------------------------------------------
# Capability parity cross-checked against the real legacy native engine
# ---------------------------------------------------------------------------


@requires_toolchain
def test_sdk_continuous_matches_legacy_engine_capability() -> None:
    """Both the legacy native engine and the SDK port must collapse the same
    one-round instance well below its initial violation count."""

    from src.decoder.legacy.backends.base import ParallelTemperingConfig
    from src.decoder.legacy.backends.native import NativeContinuousRelaxedEngine

    fixture = load_fixture(FIXTURE_METADATA)
    specs = block_specs_for_fixture(fixture, start_block=0, block_count=1)
    model_specs = block_specs_for_model(fixture, specs, aes_rounds=1)
    circuit = build_xts_block_collection(model_specs, encoding="lowered", aes_rounds=1)

    rng = random.Random(1234)
    start = Assignment.empty(16, len(circuit.wire_offsets))
    start.plaintext[:] = bytes(rng.randrange(256) for _ in range(16))
    start.key1[:] = bytes(rng.randrange(256) for _ in range(32))
    start.key2[:] = bytes(rng.randrange(256) for _ in range(32))
    circuit.derive_wire_values(start)
    legacy_initial = circuit.evaluate(start).hamming_score
    assert legacy_initial > 0

    engine = NativeContinuousRelaxedEngine(
        circuit,
        start,
        config=ParallelTemperingConfig(replicas=4, seed=7, aes_rounds=1, key_gibbs_prob=0.75),
    )
    for _ in range(6):
        engine.run_epoch(50)
    legacy_final = engine.native_engine.score_assignment(
        engine.native_engine.current_assignment()
    )["hamming_score"]
    assert legacy_final <= 0.4 * legacy_initial, (legacy_final, legacy_initial)

    # The SDK port, on the same reduced-round task, must be at least as effective.
    spec = _spec((1,))
    sdk_initial = _single_task("baseline", spec, CONFORM)["best_hamming"]
    sdk_final = _single_task("continuous_relaxed", spec, STRONG)["best_hamming"]
    sdk_frac = sdk_final / sdk_initial
    legacy_frac = legacy_final / legacy_initial
    assert sdk_frac <= 0.4, (sdk_final, sdk_initial)
    assert sdk_frac <= legacy_frac + 0.1, (sdk_frac, legacy_frac)


# ---------------------------------------------------------------------------
# Determinism (RLVR requires reproducible rewards for a fixed seed)
# ---------------------------------------------------------------------------


@requires_toolchain
@pytest.mark.parametrize("name", PORTED)
def test_ported_solver_reward_is_deterministic(name: str) -> None:
    spec = _spec((1, 2))
    env = SolverAuthoringEnv(options=DETERMINISTIC)
    first = env.score_source(_example(name), spec)
    second = env.score_source(_example(name), spec)
    assert first.reward == second.reward
    assert [t["best_hamming"] for t in first.per_task] == [t["best_hamming"] for t in second.per_task]
    assert [t["energy"] for t in first.per_task] == [t["energy"] for t in second.per_task]
