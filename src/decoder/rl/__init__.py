"""RL environment for authoring AES-XTS solver plugins.

The environment is the product: a model's policy emits C++ solver source that
implements the ``ISolver`` contract (see
``src/decoder/rl/native/include/solver_sdk.hpp`` and ``examples/``); the
harness compiles and runs it against a curriculum of reduced-round AES-XTS
instances; and the reward is the augmented-Lagrangian energy densified with a
per-round staircase and a feasibility bonus.

Public surface:

- :class:`~src.decoder.rl.curriculum.Curriculum` / :class:`TaskSpec` -- the
  reduced-round ladder and coupling schedule.
- :class:`~src.decoder.rl.reward.Rubric` -- Lagrangian + staircase + feasibility
  shaping with components exposed for GRPO analysis.
- :class:`~src.decoder.rl.reward.RolloutResult` -- the structured, framework-
  agnostic result every adapter maps into its native reward/state/info shape.
- :class:`~src.decoder.rl.solver_env.SolverAuthoringEnv` -- the framework-neutral
  core (``prompts`` / ``score_source`` / ``score_completion`` / ``score_batch``);
  per-framework adapters live in :mod:`src.decoder.rl.adapters`.
- :func:`~src.decoder.rl.harness.evaluate_solver_source` -- compile + score a
  solver source string, returning the full reward breakdown.
"""

from __future__ import annotations

from .curriculum import Curriculum, TaskSpec
from .harness import (
    HarnessOptions,
    HarnessUnavailable,
    Sandbox,
    SandboxUnavailable,
    evaluate_solver_source,
    harness_available,
)
from .reward import RewardBreakdown, RolloutResult, Rubric
from .solver_env import SolverAuthoringEnv

__all__ = [
    "Curriculum",
    "TaskSpec",
    "Rubric",
    "RewardBreakdown",
    "RolloutResult",
    "SolverAuthoringEnv",
    "HarnessOptions",
    "Sandbox",
    "SandboxUnavailable",
    "evaluate_solver_source",
    "harness_available",
    "HarnessUnavailable",
]
