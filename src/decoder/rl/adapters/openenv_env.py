"""OpenEnv (Meta PyTorch / Hugging Face) adapter.

OpenEnv serves an environment over HTTP using the MCP tool protocol: you
subclass ``openenv.core.env_server.mcp_environment.MCPEnvironment``, create a
``FastMCP`` server, and register tools with ``@mcp.tool``. The task is
single-turn: the model calls one tool, ``submit_solver(source)``, which compiles
and runs the C++ solver in the Phase 1 sandbox and returns the deterministic
reward + the full :class:`RolloutResult` as structured output.

Entry points:
* ``load_environment()`` -- build the ``MCPEnvironment`` (needs ``openenv-core``),
* ``submit_solver_result(source, ...)`` -- the framework-neutral scoring call the
  MCP tool wraps (unit-testable without OpenEnv installed).

Install with ``pip install aes-xts-decoder[openenv]`` (openenv-core, >= py3.11).
Deploy with the ``docker/Dockerfile.openenv`` scaffold (needs the C++ toolchain
+ bwrap in the image).
"""

from __future__ import annotations

from pathlib import Path

from ..reward import RolloutResult
from ._common import build_core_env, completion_text, missing_dependency, prompt_for


def submit_solver_result(source, *, env=None, task_index: int = 0, **build_kwargs) -> RolloutResult:
    """Compile+score a submitted solver, returning the structured result.

    Framework-independent so it can be unit-tested and reused by the MCP tool
    body. ``env`` may be a prebuilt core env; otherwise one is built from
    ``build_kwargs``.
    """
    env = env or build_core_env(**build_kwargs)
    spec = env.task_specs[int(task_index)]
    return env.score_completion(completion_text(source), spec)


def _import_fastmcp():
    try:  # FastMCP ships standalone and also under the mcp SDK.
        from fastmcp import FastMCP  # type: ignore

        return FastMCP
    except ImportError:
        from mcp.server.fastmcp import FastMCP  # type: ignore

        return FastMCP


def load_environment(
    *,
    fixture: str | Path | None = None,
    block_count: int = 1,
    max_rounds: int = 3,
    seed: int = 0,
    artifacts_dir: str | Path | None = None,
    **_kwargs,
):
    """Return an OpenEnv ``MCPEnvironment`` exposing the ``submit_solver`` tool."""
    try:
        from openenv.core.env_server.mcp_environment import MCPEnvironment
    except ImportError as exc:  # pragma: no cover - exercised only when installed
        raise missing_dependency("openenv-core", "openenv") from exc

    FastMCP = _import_fastmcp()

    core = build_core_env(
        fixture=fixture, block_count=block_count, max_rounds=max_rounds,
        seed=seed, artifacts_dir=artifacts_dir,
    )

    mcp = FastMCP("aes-xts-solver-authoring")
    last: dict = {"result": None}

    @mcp.tool
    def submit_solver(source: str) -> dict:
        """Submit a C++ solver translation unit; compiles+scores it, ends the episode.

        Returns the full RolloutResult dict (reward, compile log, per-task
        metrics, source-extraction diagnostics).
        """
        result = submit_solver_result(source, env=core)
        last["result"] = result
        return result.to_dict()

    class SolverAuthoringOpenEnv(MCPEnvironment):
        """Single-turn MCP environment for authoring an AES-XTS solver."""

        def __init__(self):
            super().__init__(mcp)
            self._core = core

        def prompt(self) -> str:
            return prompt_for(self._core.task_specs[0])

        def last_result(self) -> RolloutResult | None:
            return last["result"]

    return SolverAuthoringOpenEnv()


def create_app(**kwargs):  # pragma: no cover - requires openenv installed
    """Build the OpenEnv ASGI app (MCP/WebSocket transport) for serving."""
    env = load_environment(**kwargs)
    # MCPEnvironment exposes the ASGI app for its FastMCP server; serve it with
    # uvicorn (see docker/Dockerfile.openenv).
    if hasattr(env, "app"):
        return env.app
    return env
