"""Verifiers (PrimeIntellect) adapter.

Verifiers is an in-process framework: an environment is a ``vf.SingleTurnEnv``
with a ``Dataset`` of prompts and a ``vf.Rubric`` of plain-Python reward
functions. This adapter exposes ``load_environment()`` -- the entry point the
Environments Hub / prime-rl call -- and wires the reward so that:

* the scalar ``reward`` drives training, and
* the full :class:`RolloutResult` (compile log, timeout/crash category,
  source-extraction diagnostics, per-task wall time) is attached to the
  verifiers ``state`` so nothing is lost, and
* the energy / round-staircase / feasibility components are exposed as
  zero-weight rubric functions so they show up in metrics without changing the
  optimised objective.

The rollout is computed once per completion and memoised on ``state`` so the
component functions do not recompile and re-run the solver.
"""

from __future__ import annotations

from pathlib import Path

from ._common import build_core_env, completion_text, missing_dependency, prompt_for


def load_environment(
    *,
    fixture: str | Path | None = None,
    block_count: int = 1,
    max_rounds: int = 3,
    seed: int = 0,
    artifacts_dir: str | Path | None = None,
    **_kwargs,
):
    """Return a ``verifiers`` ``SingleTurnEnv`` for the solver-authoring task."""
    try:
        import verifiers as vf
    except ImportError as exc:  # pragma: no cover - exercised only when installed
        raise missing_dependency("verifiers", "verifiers") from exc

    try:
        from datasets import Dataset
    except ImportError as exc:  # pragma: no cover
        raise missing_dependency("datasets", "verifiers") from exc

    env = build_core_env(
        fixture=fixture, block_count=block_count, max_rounds=max_rounds,
        seed=seed, artifacts_dir=artifacts_dir,
    )
    specs = env.task_specs
    rows = [
        {"question": prompt_for(spec), "task_index": index}
        for index, spec in enumerate(specs)
    ]
    dataset = Dataset.from_list(rows)

    def _result(completion, task_index, state):
        # Compute once per rollout; memoise on state for the component funcs.
        if isinstance(state, dict) and "_rollout_result" in state:
            return state["_rollout_result"]
        spec = specs[int(task_index)]
        result = env.score_completion(completion_text(completion), spec)
        if isinstance(state, dict):
            state["_rollout_result"] = result
            state["rollout_result"] = result.to_dict()
        return result

    def reward_func(completion, task_index=0, state=None, **_kw):
        return _result(completion, task_index, state).reward

    def energy_reward(completion, task_index=0, state=None, **_kw):
        return _result(completion, task_index, state).components.get("energy", 0.0)

    def round_reward(completion, task_index=0, state=None, **_kw):
        return _result(completion, task_index, state).components.get("round", 0.0)

    def feasible_reward(completion, task_index=0, state=None, **_kw):
        return 1.0 if _result(completion, task_index, state).feasible else 0.0

    # Only the first function contributes to the optimised reward; the rest are
    # zero-weight so they surface as inspectable metrics.
    rubric = vf.Rubric(
        funcs=[reward_func, energy_reward, round_reward, feasible_reward],
        weights=[1.0, 0.0, 0.0, 0.0],
    )
    return vf.SingleTurnEnv(dataset=dataset, rubric=rubric)
