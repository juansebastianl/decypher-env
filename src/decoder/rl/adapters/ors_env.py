"""Open Reward Standard (ORS) adapter.

ORS models an environment as a set of ``@tool`` functions returning a
``ToolOutput(blocks=..., reward=..., finished=...)``. The solver-authoring task
exposes one tool, ``submit_solver(source)``, that compiles+scores the C++ in the
Phase 1 sandbox and finishes the episode with the deterministic reward. Tasks
(splits) are backed by the benchmark suites.

Entry points:
* ``load_environment()`` -- build the ``ors.Environment``,
* ``list_tasks(split)`` -- task ids for a split,
* ``serve()`` -- run an ORS ``Server`` hosting the environment.

Deploy with the ``docker/ors`` scaffold (needs the C++ toolchain + bwrap).
"""

from __future__ import annotations

from pathlib import Path

from ..reward import RolloutResult
from ._common import build_core_env, completion_text, missing_dependency, prompt_for


def submit_solver_result(source, *, env=None, task_index: int = 0, **build_kwargs) -> RolloutResult:
    """Framework-neutral scoring call the ORS tool wraps (unit-testable)."""
    env = env or build_core_env(**build_kwargs)
    spec = env.task_specs[int(task_index)]
    return env.score_completion(completion_text(source), spec)


def list_tasks(split: str = "smoke"):
    """Return task ids for a split, backed by the benchmark suites (Phase 3)."""
    from ..benchmark import SUITES

    suite = SUITES.get(split)
    if suite is None:
        raise ValueError(f"unknown split {split!r}; choose one of {', '.join(SUITES)}")
    return [f"{split}:{index}" for index in range(len(suite.specs))]


def load_environment(
    *,
    fixture: str | Path | None = None,
    block_count: int = 1,
    max_rounds: int = 3,
    seed: int = 0,
    artifacts_dir: str | Path | None = None,
    **_kwargs,
):
    """Return an ``ors.Environment`` exposing ``submit_solver``."""
    try:
        import ors
        from ors import ToolOutput, tool
    except ImportError as exc:  # pragma: no cover - exercised only when installed
        raise missing_dependency("ors", "ors") from exc

    core = build_core_env(
        fixture=fixture, block_count=block_count, max_rounds=max_rounds,
        seed=seed, artifacts_dir=artifacts_dir,
    )

    class SolverAuthoringOrsEnv(ors.Environment):
        name = "aes-xts-solver-authoring"

        def prompt(self) -> str:
            return prompt_for(core.task_specs[0])

        @tool
        def submit_solver(self, source: str) -> "ToolOutput":
            """Submit and score a C++ solver; finishes the episode."""
            result = submit_solver_result(source, env=core)
            return ToolOutput(
                blocks=[{"type": "text", "text": f"reward={result.reward:.4f}"}],
                reward=float(result.reward),
                finished=True,
            )

        def list_tasks(self, split: str = "smoke"):
            return list_tasks(split)

    return SolverAuthoringOrsEnv()


def serve(**kwargs):
    """Run an ORS server hosting the solver-authoring environment."""
    try:
        from ors import Server
    except ImportError as exc:  # pragma: no cover
        raise missing_dependency("ors", "ors") from exc
    return Server([load_environment(**kwargs)]).run()
