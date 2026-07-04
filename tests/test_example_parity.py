"""Capability tests for the reference example solvers.

The three "reference strategy" solvers -- parallel_tempering, continuous_relaxed
and algebraic_relaxed -- implement classic search strategies against the ISolver
plugin interface. Each must

  * conform to the interface (compile, run, not crash or time out),
  * do real work (drive constraint violations well below the no-search floor),
  * preserve the expected capability ordering (key-conditional Gibbs solvers
    beat the pure-Metropolis one, collapsing a one-round instance).

All rewards are deterministic given a seed, which the determinism test asserts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.decoder.rl.curriculum import TaskSpec
from src.decoder.rl.harness import HarnessOptions, harness_available
from src.decoder.rl.solver_env import SolverAuthoringEnv

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
    # The exact key-conditional Gibbs move is what lets the continuous /
    # algebraic solvers collapse a one-round instance.
    assert task["best_hamming"] <= 0.15 * initial, (name, task["best_hamming"], initial)


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
