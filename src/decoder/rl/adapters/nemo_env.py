"""NeMo Gym (NVIDIA) adapter.

NeMo Gym runs three async FastAPI services (agent / model / resources). Custom
tasks live in a *resources server*: you subclass ``SimpleResourcesServer``,
register tool endpoints in ``setup_webserver`` and implement ``verify(body)`` to
score the rollout with a reward. This task exposes ``POST /submit_solver`` (the
model's C++ solver) and the post-episode ``POST /verify`` (recompute + return
the reward and full :class:`RolloutResult` metrics); both route through the same
sandboxed scoring core.

Entry points:
* ``submit_solver_result(source, ...)`` -- framework-neutral scoring (unit-testable),
* ``create_app()`` -- a plain FastAPI app with the two routes (handy for local
  testing without the full NeMo stack),
* ``load_environment()`` -- the ``SimpleResourcesServer`` subclass NeMo drives,
* ``run_webserver()`` -- serve ``create_app()`` with uvicorn.

NeMo Gym requires Python >= 3.12. Install with ``pip install aes-xts-decoder[nemo]``.
Deploy with the ``docker/Dockerfile.nemo`` scaffold (needs the C++ toolchain +
bwrap in the image).
"""

from __future__ import annotations

from pathlib import Path

from ..reward import RolloutResult
from ._common import build_core_env, completion_text, missing_dependency, prompt_for


def submit_solver_result(source, *, env=None, task_index: int = 0, **build_kwargs) -> RolloutResult:
    """Framework-neutral scoring call the routes wrap (unit-testable)."""
    env = env or build_core_env(**build_kwargs)
    spec = env.task_specs[int(task_index)]
    return env.score_completion(completion_text(source), spec)


def _extract_source_from_response(response) -> str:
    """Best-effort pull of the model's C++ out of a NeMo response object/dict."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    # NeMoGymResponse / OpenAI-style: prefer an output_text, else stringify.
    for attr in ("output_text", "content", "text"):
        value = getattr(response, attr, None) or (
            response.get(attr) if isinstance(response, dict) else None
        )
        if value:
            return completion_text(value)
    return completion_text(response)


def create_app(
    *,
    fixture: str | Path | None = None,
    block_count: int = 1,
    max_rounds: int = 3,
    seed: int = 0,
    artifacts_dir: str | Path | None = None,
    **_kwargs,
):
    """Build a plain FastAPI app with ``/submit_solver`` and ``/verify`` routes."""
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
    except ImportError as exc:  # pragma: no cover - exercised only when installed
        raise missing_dependency("fastapi", "nemo") from exc

    core = build_core_env(
        fixture=fixture, block_count=block_count, max_rounds=max_rounds,
        seed=seed, artifacts_dir=artifacts_dir,
    )
    app = FastAPI(title="aes-xts-solver-authoring")

    class SolverSubmission(BaseModel):
        source: str
        task_index: int = 0

    @app.get("/prompt")
    def prompt():  # pragma: no cover - trivial
        return {"prompt": prompt_for(core.task_specs[0])}

    @app.post("/submit_solver")
    def submit_solver(body: SolverSubmission):  # pragma: no cover - needs server
        result = submit_solver_result(body.source, env=core, task_index=body.task_index)
        return result.to_dict()

    @app.post("/verify")
    def verify(body: SolverSubmission):  # pragma: no cover - needs server
        result = submit_solver_result(body.source, env=core, task_index=body.task_index)
        payload = result.to_dict()
        payload["reward"] = result.reward
        payload["verified"] = True
        return payload

    return app


def load_environment(
    *,
    fixture: str | Path | None = None,
    block_count: int = 1,
    max_rounds: int = 3,
    seed: int = 0,
    artifacts_dir: str | Path | None = None,
    **_kwargs,
):
    """Return a NeMo Gym ``SimpleResourcesServer`` subclass for the task."""
    try:
        from nemo_gym import SimpleResourcesServer
        from nemo_gym.base_resources_server import BaseVerifyRequest, BaseVerifyResponse
    except ImportError as exc:  # pragma: no cover - exercised only when installed
        raise missing_dependency("nemo-gym", "nemo") from exc

    core = build_core_env(
        fixture=fixture, block_count=block_count, max_rounds=max_rounds,
        seed=seed, artifacts_dir=artifacts_dir,
    )

    class SolverAuthoringResourcesServer(SimpleResourcesServer):
        """Single-step resources server: /submit_solver tool + /verify reward."""

        def setup_webserver(self):
            app = super().setup_webserver()
            app.post("/submit_solver")(self.submit_solver)
            return app

        async def submit_solver(self, body):  # pragma: no cover - needs server
            source = getattr(body, "source", "") or ""
            return submit_solver_result(source, env=core).to_dict()

        async def verify(self, body: "BaseVerifyRequest") -> "BaseVerifyResponse":  # pragma: no cover
            source = _extract_source_from_response(getattr(body, "response", None))
            result = submit_solver_result(source, env=core)
            return BaseVerifyResponse(**body.model_dump(), reward=float(result.reward))

    return SolverAuthoringResourcesServer


def run_webserver(host: str = "0.0.0.0", port: int = 8000, **kwargs):  # pragma: no cover
    """Serve the FastAPI app with uvicorn (local, framework-free serving)."""
    try:
        import uvicorn
    except ImportError as exc:
        raise missing_dependency("uvicorn", "nemo") from exc
    uvicorn.run(create_app(**kwargs), host=host, port=port)
