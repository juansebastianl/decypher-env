"""SkyRL Gym (NovaSky / Berkeley) adapter.

SkyRL Gym is an in-process, Gym-style framework built around ``BaseTextEnv``:
``init(prompt)`` seeds the episode and ``step(action)`` returns a
``BaseTextEnvStepOutput`` carrying ``observations``, ``reward``, ``done`` and
``metadata``. The task is single-turn, so ``step`` returns ``done=True`` with the
scalar reward and the full :class:`RolloutResult` in ``metadata``.

``load_environment()`` returns a ready instance; ``register()`` registers it with
``skyrl_gym`` so it can be constructed by id.
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
    """Return a SkyRL Gym ``BaseTextEnv`` instance for the solver-authoring task."""
    try:
        from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
    except ImportError as exc:  # pragma: no cover - exercised only when installed
        raise missing_dependency("skyrl_gym", "skyrl") from exc

    core = build_core_env(
        fixture=fixture, block_count=block_count, max_rounds=max_rounds,
        seed=seed, artifacts_dir=artifacts_dir,
    )

    class SolverAuthoringSkyRLEnv(BaseTextEnv):
        """Single-turn SkyRL text env for authoring an AES-XTS solver."""

        def __init__(self, env_config=None):
            super().__init__()
            self._core = core
            self._specs = core.task_specs
            self._cursor = 0

        def init(self, prompt=None):
            self._cursor = 0
            return prompt_for(self._specs[self._cursor]), {}

        def step(self, action):
            spec = self._specs[self._cursor]
            result = self._core.score_completion(completion_text(action), spec)
            return BaseTextEnvStepOutput(
                observations=[],
                reward=float(result.reward),
                done=True,
                metadata=result.to_dict(),
            )

    return SolverAuthoringSkyRLEnv()


def register(env_id: str = ENV_ID, **kwargs) -> str:
    """Register the env with ``skyrl_gym`` so it can be built by id."""
    try:
        import skyrl_gym
    except ImportError as exc:  # pragma: no cover
        raise missing_dependency("skyrl_gym", "skyrl") from exc
    skyrl_gym.register(
        id=env_id,
        entry_point="src.decoder.rl.adapters.skyrl_env:load_environment",
    )
    return env_id
