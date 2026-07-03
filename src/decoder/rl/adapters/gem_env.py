"""GEM (Axon-RL) adapter.

GEM follows the Gymnasium API most closely: ``reset()`` returns
``(observation, info)`` and ``step(action)`` returns the 5-tuple
``(observation, reward, terminated, truncated, info)``. The solver-authoring
task is single-turn, so ``step`` always terminates; the action is the model
completion (the C++ solver), and the :class:`RolloutResult` is returned as the
Gymnasium ``info`` dict.

``load_environment()`` returns a ready instance; ``register()`` registers it
under a GEM id so ``gem.make("SolverAuthoring-v0")`` works.
"""

from __future__ import annotations

from pathlib import Path

from ._common import build_core_env, completion_text, missing_dependency, prompt_for

ENV_ID = "SolverAuthoring-v0"


def load_environment(
    *,
    fixture: str | Path | None = None,
    block_count: int = 1,
    max_rounds: int = 3,
    seed: int = 0,
    artifacts_dir: str | Path | None = None,
    **_kwargs,
):
    """Return a GEM ``Env`` instance for the solver-authoring task."""
    try:
        import gem
    except ImportError as exc:  # pragma: no cover - exercised only when installed
        raise missing_dependency("gem", "gem") from exc

    core = build_core_env(
        fixture=fixture, block_count=block_count, max_rounds=max_rounds,
        seed=seed, artifacts_dir=artifacts_dir,
    )

    class SolverAuthoringGemEnv(gem.Env):
        """Single-turn Gymnasium env: observe prompt, act with C++, get reward."""

        def __init__(self):
            super().__init__()
            self._core = core
            self._specs = core.task_specs
            self._cursor = 0

        def reset(self, *, seed=None, options=None):
            self._cursor = 0
            return prompt_for(self._specs[self._cursor]), {}

        def step(self, action):
            spec = self._specs[self._cursor]
            result = self._core.score_completion(completion_text(action), spec)
            # Single-turn: the episode terminates on the one submitted solver.
            return "", float(result.reward), True, False, result.to_dict()

    return SolverAuthoringGemEnv()


def register(env_id: str = ENV_ID, **kwargs) -> str:
    """Register the env with GEM's registry so ``gem.make(env_id)`` works."""
    try:
        import gem
    except ImportError as exc:  # pragma: no cover
        raise missing_dependency("gem", "gem") from exc
    gem.register(env_id, entry_point=lambda **kw: load_environment(**{**kwargs, **kw}))
    return env_id
