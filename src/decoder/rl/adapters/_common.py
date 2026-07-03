"""Shared plumbing for the framework adapters.

Every adapter is a thin, per-framework mapping over the same core: build the
prompt, run :meth:`SolverAuthoringEnv.score_completion` (which compiles and runs
the model-authored solver inside the Phase 1 sandbox), and map the resulting
:class:`~src.decoder.rl.reward.RolloutResult` into that framework's native
reward/state/info shape. Keeping that logic here means the adapters differ only
in the framework glue, not in what a rollout *means*.
"""

from __future__ import annotations

from pathlib import Path

from ..curriculum import Curriculum, TaskSpec
from ..harness import HarnessOptions
from ..reward import RolloutResult
from ..solver_env import SolverAuthoringEnv, build_prompt

# The bundled fixture window the environment is scored on by default.
DEFAULT_FIXTURE = (
    Path(__file__).resolve().parents[3] / "testing_setup" / "lorem_ipsum_aes_xts.json"
)


def missing_dependency(framework: str, extra: str) -> ImportError:
    """A consistent, actionable error when a framework package is not installed."""
    return ImportError(
        f"the '{framework}' package is required for this adapter but is not "
        f"installed. Install it with `pip install aes-xts-decoder[{extra}]` "
        f"(or `pip install {framework}`)."
    )


def build_core_env(
    *,
    fixture: str | Path | None = None,
    block_count: int = 1,
    max_rounds: int = 3,
    seed: int = 0,
    options: HarnessOptions | None = None,
    artifacts_dir: str | Path | None = None,
) -> SolverAuthoringEnv:
    """Construct the framework-neutral :class:`SolverAuthoringEnv` adapters wrap."""
    fixture_path = Path(fixture) if fixture is not None else DEFAULT_FIXTURE
    curriculum = Curriculum(
        metadata_path=fixture_path,
        block_count=block_count,
        max_rounds=max_rounds,
        seed=seed,
    )
    env = SolverAuthoringEnv.from_curriculum(curriculum, options=options)
    if artifacts_dir is not None:
        env.artifacts_dir = Path(artifacts_dir)
    return env


def completion_text(completion) -> str:
    """Normalise a completion (str / OpenAI-style messages / dict) to plain text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, (list, tuple)) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
        return str(last)
    return str(completion)


def prompt_for(spec: TaskSpec) -> str:
    return build_prompt(spec)


def score(env: SolverAuthoringEnv, completion, spec: TaskSpec) -> RolloutResult:
    """Score one completion against one task spec, returning the structured result."""
    return env.score_completion(completion_text(completion), spec)
